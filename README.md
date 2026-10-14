# Cognitive-guided trajectory sampling · Experiment 1

本仓库实现 `trajectory_pool_plan.md` 定义的第一期流水线：只负责 MATH Train 数据准备、Qwen3 自生成轨迹、最终答案验证、同题文本去重和候选池冻结，不包含认知评分、MIRT、排序、SFT 或多轮进化。

`verification_status=correct` 仅表示最终答案通过数学等价校验，不代表推理过程正确。

## 模块与数据边界

| 命令模块 | 读取 | 写入 |
| --- | --- | --- |
| `pipeline.prepare_dataset` | Hugging Face MATH train 的 7 个配置 | `splits/`、`private/gold_answers.jsonl`、数据审计与报告 |
| `pipeline.generate` | 仅 `splits/*.jsonl` 中的公开题干 | `trajectories/*_raw.jsonl`、run manifest、事件日志 |
| `pipeline.verify` | 原始轨迹与私有 gold | `verification/*_verified.jsonl`，不调用生成模型 |
| `pipeline.build_pool` | 仅 seed_train raw/verified 与题目元数据 | `pool/` 下候选、去重审计及冻结 manifest |
| `pipeline.report` | 前述阶段文件 | `reports/pilot_report.json` 或综合报告 |

生成模块会主动拒绝包含 `solution`、`gold_answer` 或 `source_solution` 的输入记录。原始 completion 不会被验证或建池模块改写。

## 目标命令

依赖版本应在正式运行环境中安装并锁定；本仓库不包含模型权重或下载产物。

```bash
python -m pipeline.prepare_dataset --config configs/experiment1.yaml --validate-only
python -m pipeline.prepare_dataset --config configs/experiment1.yaml

python -m pipeline.generate --config configs/experiment1.yaml --subset pilot --resume
python -m pipeline.verify --config configs/experiment1.yaml --subset pilot
python -m pipeline.report --config configs/experiment1.yaml --stage pilot

python -m pipeline.generate --config configs/experiment1.yaml --subset seed_train --resume
python -m pipeline.generate --config configs/experiment1.yaml --subset anchor --resume
python -m pipeline.verify --config configs/experiment1.yaml --subset seed_train
python -m pipeline.verify --config configs/experiment1.yaml --subset anchor

python -m pipeline.build_pool --config configs/experiment1.yaml --freeze
python -m pipeline.report --config configs/experiment1.yaml --stage all
pytest -q
```

正式运行前在目标环境记录依赖与硬件信息：

```bash
pip freeze > runs/math_qwen3_17b_v1/requirements.lock.txt
```

代码会把 Python、平台以及可读取的核心包版本写入 `run_manifest.json`；模型加载后还会记录解析得到的 revision、设备和 dtype。

## 无 GPU 的 mock 流程

`--backend mock` 不加载模型。可通过 `--mock-fixture` 传入每行包含 `problem_id`、`sample_index`、`raw_completion` 和可选 `finish_reason` 的 JSONL。请复制配置并为 mock 使用独立的 `run_id`、`paths.root` 和 `paths.run_root`，不要污染正式 run：

```bash
python -m pipeline.generate \
  --config configs/experiment1.yaml \
  --subset pilot --backend mock --mock-fixture tests/fixtures/mock_completions.jsonl --resume
```

默认 mock 后端也能生成确定性占位输出，但它仅用于验证 I/O 和续跑，不用于评估数学正确率。

## 续跑、调试与冻结

- 每条轨迹写入后立即 flush/fsync。`--resume` 只跳过 schema 完整且状态为 success 的稳定 ID。
- 不完整的 JSONL 末行会被移除并写入 `generation_events.jsonl`；其它位置损坏会直接报错。
- 同一 run 的配置哈希、生成参数哈希或模型 revision 改变时会拒绝混写，应改用新的 `run_id` 和路径。
- `--limit N` 只用于调试；run manifest 会记录 limited，正式 `--freeze` 会拒绝该 seed_train 运行。
- 若有效题不足目标值，建池会先写真实缺口报告，再以非零状态退出；不会用单条正确轨迹题目补足。
- 冻结 manifest 记录所有输入与输出 SHA256。输入变化后不能复用已冻结池。

## 常见错误恢复

- OOM：错误会进入事件日志。不要在原 run 内临时降低 `max_new_tokens`；修改配置并启用新 `run_id`。
- gold 缺失或 raw/verified ID 不一一对应：修复上游数据后重新执行对应阶段，不要直接编辑候选池。
- `unverifiable` 比例异常：使用 `--manual-review-export [PATH]` 导出全部 unverifiable 及 correct/incorrect 分层样本，修复验证规则后再冻结。
- 候选池不足 600 题：保留缺口报告，按规格建立新的扩容 run 或新的较小实验配置。

真实数据下载、Qwen3 推理和正式冻结需要目标机器上的依赖、网络与模型硬件；仓库测试使用 mock 数据，不声称已经完成真实实验。
