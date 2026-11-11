# Single-round STaR on GSM8K with GPT-J 6B

本仓库实现单轮 STaR 方法级复现和固定正确轨迹池上的子样本微调对照。旧的 MATH/Qwen3 流水线已删除；当前唯一实验契约是：

- 数据集：`openai/gsm8k`, config `main`
- 基模：`EleutherAI/gpt-j-6b`
- 生成：论文风格的纯文本 few-shot `Q:/A:` prompt，贪心解码，输出以 `#### answer` 结束
- 验证：只验证最终答案数学等价，不声称推理过程正确
- 微调：PEFT LoRA，prompt/padding loss 全部 mask，只监督模型生成的 rationale + final answer
- 对照：同一 `pool_hash` 下的 full、random 25/50/75%、exact dedup；另行比较 with/without rationalization

论文使用 GPT-J 6B 与 GSM8K，并在每轮从原始预训练模型重新训练。原论文是 TPU 上的全参数训练；本仓库为了单卡 AutoDL 可运行，采用 LoRA，因此属于现代方法级复现而非论文数值复现。

## 1. 本地提交到 GitHub

确认测试通过后，把新版放到独立分支：

```bash
git checkout -b star-gsm8k-single-round
python -m pytest -q
git add -A
git commit -m "replace MATH Qwen pipeline with GSM8K GPT-J STaR"
git push -u origin star-gsm8k-single-round
```

## 2. AutoDL 环境建议

当前实现不做 4-bit 量化。GPT-J 6B 的 fp16 LoRA：24GB 显存可先跑 32 题 pilot，但正式 4,000 题生成和 1024-token SFT 推荐 40GB/48GB GPU；系统内存建议至少 32GB，数据盘预留 50GB。选择带 PyTorch、CUDA 12.x、Python 3.10/3.11 的镜像。

在 AutoDL 数据盘操作：

```bash
cd /root/autodl-tmp
git clone --branch star-gsm8k-single-round --single-branch \
  https://github.com/isabellabcc/cognitive-selfevolving.git
cd cognitive-selfevolving

python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

nvidia-smi
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30)
PY
python -m pytest -q
python -m pipeline.prepare_dataset --config configs/gsm8k_gptj6b_single_round.yaml --validate-only
```

缓存放到数据盘：

```bash
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf-cache/datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
```

如果服务器只能通过镜像访问 Hugging Face，可在 shell 中临时设置 `HF_ENDPOINT`；不要把镜像地址或 token 提交到仓库。

复制一份运行配置到仓库外，避免以后 `git pull` 与服务器本地路径发生冲突：

```bash
mkdir -p /root/autodl-tmp/star/configs
cp configs/gsm8k_gptj6b_single_round.yaml /root/autodl-tmp/star/configs/gsm8k_gptj6b_r0.yaml
CONFIG=/root/autodl-tmp/star/configs/gsm8k_gptj6b_r0.yaml
```

编辑 `$CONFIG`，把两个路径改成绝对数据盘路径，并为本次实验保留唯一 run：

```yaml
paths:
  root: /root/autodl-tmp/star/data/gsm8k_gptj6b_r0
  run_root: /root/autodl-tmp/star/runs/gsm8k_gptj6b_r0
```

## 3. 数据准备和 32 题 pilot

以下命令才会下载 GSM8K；模型在第一次真实生成/评测时下载。

```bash
python -m pipeline.prepare_dataset --config "$CONFIG"

python -m pipeline.generate --config "$CONFIG" --subset pilot --resume
python -m pipeline.verify --config "$CONFIG" --subset pilot
```

检查：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('/root/autodl-tmp/star/data/gsm8k_gptj6b_r0/verification/pilot_seed_train_verified.jsonl')
rows = [json.loads(x) for x in p.read_text().splitlines()]
from collections import Counter
print(Counter(r['verification_status'] for r in rows))
PY
```

若全部 `unverifiable`，先检查 completion 是否出现 `#### number`；若大量 `length`，再检查 stopping criteria 和 `max_new_tokens`，不要直接进入正式生成。

## 4. 基模验证集基线

先只打印计划，再显式运行：

```bash
python -m pipeline.evaluate_star --config "$CONFIG" --checkpoint base --split validation
python -m pipeline.evaluate_star --config "$CONFIG" --checkpoint base --split validation --run
```

官方 test 暂时不要运行，避免根据测试集反复调参。

## 5. 普通生成、验证与固定池

```bash
python -m pipeline.generate --config "$CONFIG" --subset seed_train --resume
python -m pipeline.verify --config "$CONFIG" --subset seed_train
python -m pipeline.star_pool build-pool --config "$CONFIG" --branch ordinary
```

记录下面文件的 `pool_hash` 和正确轨迹数：

```text
/root/autodl-tmp/star/data/gsm8k_gptj6b_r0/star/pools/ordinary_manifest.json
```

## 6. 创建一个选择并训练

全量示例：

```bash
python -m pipeline.star_pool select \
  --config "$CONFIG" --branch ordinary --method all --seed 42 \
  | tee /tmp/star_selection.json

SEL=$(python -c "import json; print(json.load(open('/tmp/star_selection.json'))['selection_id'])")
echo "$SEL"

python -m pipeline.sft --config "$CONFIG" --selection-id "$SEL" --seed 42
python -m pipeline.sft --config "$CONFIG" --selection-id "$SEL" --seed 42 --run

ADAPTER=/root/autodl-tmp/star/runs/gsm8k_gptj6b_r0/star/${SEL}-train-s42/adapter
python -m pipeline.evaluate_star \
  --config "$CONFIG" --checkpoint "$ADAPTER" --split validation --run
```

每个条件都会从配置中的原始 GPT-J 基座独立初始化；不要把某个子集 adapter 填回 `model.adapter_path` 后再训练另一个子集。

## 7. 子样本实验矩阵

普通池至少运行：

| condition | selector | fraction | selection/train seeds |
| --- | --- | ---: | --- |
| `star_full` | all | 1.00 | 42 |
| `star_rand25` | random | 0.25 | 42, 43, 44 |
| `star_rand50` | random | 0.50 | 42, 43, 44 |
| `star_rand75` | random | 0.75 | 42, 43, 44 |
| `star_exact_dedup` | exact_dedup | 1.00 | 42 |

随机 50% 示例：

```bash
for SEED in 42 43 44; do
  OUT=/tmp/star-rand50-${SEED}.json
  python -m pipeline.star_pool select \
    --config "$CONFIG" --branch ordinary --method random --fraction 0.50 --seed "$SEED" \
    | tee "$OUT"
  SEL=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['selection_id'])" "$OUT")
  python -m pipeline.sft --config "$CONFIG" --selection-id "$SEL" --seed "$SEED" --run
  ADAPTER=/root/autodl-tmp/star/runs/gsm8k_gptj6b_r0/star/${SEL}-train-s${SEED}/adapter
  python -m pipeline.evaluate_star \
    --config "$CONFIG" --checkpoint "$ADAPTER" --split validation --run
done
```

将 `0.50` 分别替换为 `0.25` 和 `0.75`。精确去重使用 `--method exact_dedup` 且不传 fraction。所有同池比较必须确认 `selection_manifest.json` 中的 `pool_hash` 相同。

## 8. Rationalization 分支

```bash
python -m pipeline.rationalize generate --config "$CONFIG" --resume
python -m pipeline.rationalize verify --config "$CONFIG"
python -m pipeline.star_pool build-pool \
  --config "$CONFIG" --branch with_rationalization
```

然后至少在新池上运行 `all` 和 `random --fraction 0.50`。该池的 `pool_hash` 必然不同；其结果用于比较 STaR 候选池生成机制，不能与 ordinary 同池筛选器混为一谈。

## 9. 汇总与最终测试

```bash
python -m pipeline.star_report --config "$CONFIG"
```

结果位于：

```text
data/gsm8k_gptj6b_r0/star/reports/results.csv
data/gsm8k_gptj6b_r0/star/reports/results.md
```

根据 validation 预先确定最终条件后，每个最终条件只运行一次官方 test：

```bash
python -m pipeline.evaluate_star \
  --config "$CONFIG" --checkpoint "$ADAPTER" --split test --run
python -m pipeline.star_report --config "$CONFIG"
```

## 10. 续跑和故障处理

- 生成和 rationalization 支持 `--resume`，会跳过完整 trajectory ID。
- 修改模型、prompt、采样参数或数据拆分后，必须更换 `run_id` 和两个输出目录。
- SFT run 目录已存在时不会覆盖；确认是失败残留后手工移动目录留档，再重跑。
- OOM 时先把 `max_seq_length` 降到 768 或将 GPU 换到 40/48GB；所有实验条件必须使用相同长度规则。
- test 不参与 prompt、超参数、样本选择或停止条件调整。

更严格的方法映射和公平性约束见 `docs/star_method_mapping.md` 与 `docs/star_experiment_protocol.md`。
