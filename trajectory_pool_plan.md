# 实验 1 · 数据生成与推理轨迹池构建：脚本开发计划

> 项目：认知测量指导的自进化样本轨迹筛选  
> 文档用途：供代码生成工具（Codex / Claude Code 等）逐模块实现、运行和测试。  
> 版本：v1.0（2026-10-12）  
> 本期边界：**只完成数据准备 → 自生成轨迹 → 答案验证 → 冻结候选轨迹池**；不实现认知评分、MIRT、样本排序、SFT 或多轮进化。

## 0. 本期目标与不可变约束

### 0.1 核心目标

从 MATH 训练集构建三个题目互斥的子集；由**同一个固定的 Qwen3-1.7B checkpoint**对每道题独立采样多条推理轨迹；自动判断最终答案；为后续 Random / Rubric / MIRT 组提供**同一份**、可溯源、可复现、不可被事后悄悄改写的候选池。

| 项目 | v1 固定配置 |
| --- | --- |
| 基础模型 | `Qwen/Qwen3-1.7B`；Thinking 模式 |
| 原始题库 | `EleutherAI/hendrycks_math` 的 **train**，读取全部 7 个配置 |
| 题目划分 | `seed_train=2500`，`anchor=300`，`dev=200`；按题目 ID 互斥 |
| 抽题方式 | 数学 Level（目标 20/30/30/15/5%）× 主题的分层随机抽样；记录真实实现比例 |
| 正式轨迹数 | 每道 `seed_train` 题采样 `n=6`，总计计划 **15,000** 条 |
| 锚点轨迹 | 300 题 × 6 = **1,800** 条，供**后续**测量模型校准；本期仅生成/验证 |
| 预实验 | 从 `dev` 中固定抽取 100 题 × 6 = 600 条；不得混入正式训练池 |
| 有效题定义 | 同一道题至少 2 条**正确且非近重复**的有效轨迹 |
| 目标主实验规模 | 在满足有效题定义的 `seed_train` 题中，固定抽取目标 600 道题 |
| 答案判定 | `math-verify` (`parse` + `verify`)；解析失败标为 `unverifiable` |
| 存储方式 | UTF-8、每行一条 JSON 的 `.jsonl`；所有中间状态保留 |

### 0.2 强约束（必须实现）

1. **禁止参考答案泄漏**：生成器只接收 `problem` 和固定 prompt 模板；`solution`/`gold_answer` 仅供离线验证模块使用。生成脚本甚至不应读取含答案的文件。
2. **题目划分严格独立**：同一道题及其近乎相同的规范化题干不得同时进入 `seed_train` / `anchor` / `dev`。
3. **不按评分筛题**：有效性仅采用最终答案正确、格式可用和文本非重复等客观规则；不采用认知分数、LLM Judge 或轨迹长短作为候选优先级。
4. **保留完整轨迹**：不能仅保存 `\boxed{}` 中的答案；必须保留全部模型 completion，包括 `<think>` 内容、最终回答和解析失败的样本。
5. **区分三态验证结果**：`correct`、`incorrect`、`unverifiable`。解析或验证器异常不可以静默计入错误答案。
6. **可复现且可续跑**：给每个 `(run_id, problem_id, sample_index)` 稳定 ID 和随机种子；中断续跑跳过已成功写盘记录；严禁不同配置混入同一 run。
7. **防数据泄漏**：只使用 MATH `train` 构建上述三个子集；MATH-500/GSM8K test 不进入本期生成或校准池；额外执行训练题与评测题的近重复检查（若评测集可用）。
8. **本期不丢弃原始错误样本**：错误和无法验证的轨迹不进入“正确候选轨迹池”，但继续保存在完整轨迹表和验证表中。

---

## 1. 技术栈、环境与可复现约束

- Python 3.11（首选，避免使用未验证的 Python 3.13 组合）。
- PyTorch + Hugging Face `transformers`（支持 Qwen3 的版本，建议 `>=4.51.0`；运行前对当前发布版本做一次 smoke test）。
- `datasets`、`accelerate`、`safetensors`、`pyyaml`、`tqdm`、`pandas`、`pytest`。
- `math-verify`、`sympy`，以及其要求的 ANTLR runtime（依项目官方 README 选择兼容版本）。
- 第一版用 **Transformers 单进程/单 GPU** 实现；批量与 vLLM 是后续优化，避免第一版混用不同采样后端。
- 尽可能使用 NVIDIA GPU；允许 `--device cpu` 或 `--backend mock` 做不加载模型的端到端测试。
- 安装完成后运行 `pip freeze > runs/<run_id>/requirements.lock.txt`，保存 CUDA、GPU、Torch、Transformers、数据集 revision、模型 revision 和 Git commit。

运行参数不能只存在代码常量中，应集中在 YAML；每次 run 计算配置哈希并写入 manifest。

### 1.1 配置文件 `configs/experiment1.yaml`

```yaml
project: cognitive_self_evolution
run_id: math_qwen3_17b_v1
master_seed: 20261012

dataset:
  repo_id: EleutherAI/hendrycks_math
  split: train
  configs:
    - algebra
    - counting_and_probability
    - geometry
    - intermediate_algebra
    - number_theory
    - prealgebra
    - precalculus
  subset_sizes:
    seed_train: 2500
    anchor: 300
    dev: 200
  target_level_ratios: {1: 0.20, 2: 0.30, 3: 0.30, 4: 0.15, 5: 0.05}
  exclude_asymptote: true
  split_seed: 42

model:
  id: Qwen/Qwen3-1.7B
  revision: main  # 运行时解析为不可变 commit SHA 并记录
  enable_thinking: true
  device: auto
  torch_dtype: auto

sampling:
  system_prompt: null
  prompt_version: math_cot_v1
  n_per_problem: 6
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  do_sample: true
  max_new_tokens: 2048
  batch_size: 1

pilot:
  source_subset: dev
  n_problems: 100

pool:
  min_distinct_correct: 2
  target_n_problems: 600
  near_duplicate_jaccard_threshold: 0.90
  dedup_token_ngram: 5
  pool_seed: 20261012

paths:
  root: data/experiment1
  run_root: runs/math_qwen3_17b_v1
```

**配置规则：**第一次成功启动后冻结配置和模型 revision；任何改变 prompt、采样参数、模型 revision、数据划分、题目过滤规则的行为，都必须启动新 `run_id`，不得继续写入旧实验文件。

---

## 2. 仓库结构与数据文件契约

```text
cognitive-self-evolution/
├── configs/
│   └── experiment1.yaml
├── pipeline/
│   ├── __init__.py
│   ├── common.py                # 加载配置、路径、哈希、JSONL、安全写盘
│   ├── prepare_dataset.py       # 下载、清洗、分层划分
│   ├── generate.py              # Qwen3 Thinking 多次采样
│   ├── parse_output.py          # 拆分 think/final 与格式检查
│   ├── verify.py                # 提取答案、判定正确性
│   ├── build_pool.py            # 去重、有效题判定、冻结候选池
│   └── report.py                # 全流程统计报告
├── tests/
│   ├── test_prepare_dataset.py
│   ├── test_parse_output.py
│   ├── test_verify.py
│   ├── test_dedup.py
│   ├── test_resume.py
│   └── fixtures/
├── data/experiment1/
│   ├── raw/                    # 原始 MATH 记录，仅数据准备使用
│   ├── splits/                 # 只含公开题干/元数据；不含参考解答
│   │   ├── seed_train.jsonl
│   │   ├── anchor.jsonl
│   │   └── dev.jsonl
│   ├── private/
│   │   └── gold_answers.jsonl  # 答案与来源，只有 verify 读取
│   ├── trajectories/
│   │   ├── seed_train_raw.jsonl
│   │   ├── anchor_raw.jsonl
│   │   └── pilot_dev_raw.jsonl
│   ├── verification/
│   │   ├── seed_train_verified.jsonl
│   │   ├── anchor_verified.jsonl
│   │   └── pilot_dev_verified.jsonl
│   ├── pool/
│   │   ├── eligible_questions.jsonl
│   │   ├── selected_questions.jsonl
│   │   ├── candidates_correct.jsonl
│   │   ├── candidates_all.jsonl
│   │   ├── dedup_audit.jsonl
│   │   └── pool_manifest.json
│   └── reports/
│       ├── dataset_report.json
│       ├── pilot_report.json
│       ├── verification_report.json
│       └── pool_report.json
├── runs/math_qwen3_17b_v1/
│   ├── config_frozen.yaml
│   ├── run_manifest.json
│   ├── generation_events.jsonl
│   └── requirements.lock.txt
└── README.md
```

**文件设计原则：**原始轨迹、验证结果和候选选择结果为不同的不可变阶段数据；使用 `trajectory_id` / `problem_id` 关联，避免在不同脚本中重复生成或覆盖字段。下游脚本从前一阶段的完整文件读取，不应从摘要 report 反向推导记录。

### 2.1 规范题目结构：`splits/*.jsonl`

每行一个对象，字段**必须一致**：

```json
{
  "problem_id": "math_algebra_train_000123_ab12cd34",
  "source_repo": "EleutherAI/hendrycks_math",
  "source_config": "algebra",
  "source_split": "train",
  "source_row_index": 123,
  "level": 2,
  "topic": "Algebra",
  "problem": "Solve ...",
  "problem_sha256": "<64-character-hex>",
  "subset": "seed_train"
}
```

- 以 `source_config + train + source_row_index + 规范化题干 SHA256 前8位` 形成稳定的 `problem_id`。
- `problem_sha256` 使用规范化题干（统一 Unicode、空白符、行尾），用于跨配置完全重复检测。
- `level` 保留整数 1–5；未识别难度写异常审计，不允许设为 0 混入分层。
- `topic` 使用 MATH 原数据 `type` 字段；保留原始 topic 映射关系。
- 题目文件**绝不包含** `solution`、`gold_answer`、原始参考推理。

`private/gold_answers.jsonl` 行结构：

```json
{
  "problem_id": "math_algebra_train_000123_ab12cd34",
  "gold_answer": "\\frac{1}{2}",
  "gold_extraction_status": "ok",
  "source_solution": "... original solution ...",
  "gold_extraction_method": "last_boxed_balanced"
}
```

`gold_extraction_status` 使用 `ok | ambiguous | missing | parse_error`；保留原始解答以供审核。由于 MATH 提供的是含推理过程的 `solution`，要实现**正确处理嵌套花括号**的 `\\boxed{...}` 提取，不能仅用简单的非贪婪正则；多 `boxed`、无 `boxed` 或格式异常应标记待复核。

### 2.2 规范原始轨迹结构：`trajectories/*_raw.jsonl`

```json
{
  "trajectory_id": "math_qwen3_17b_v1:math_algebra_train_000123_ab12cd34:s00",
  "run_id": "math_qwen3_17b_v1",
  "problem_id": "math_algebra_train_000123_ab12cd34",
  "subset": "seed_train",
  "sample_index": 0,
  "model_id": "Qwen/Qwen3-1.7B",
  "model_revision": "<resolved-commit-SHA>",
  "prompt_version": "math_cot_v1",
  "sampling_seed": 123456789,
  "generation_config_sha256": "<64-character-hex>",
  "raw_completion": "<think>... </think> Final answer: \\boxed{...}",
  "reasoning_text": "...",
  "final_text": "Final answer: \\boxed{...}",
  "format_status": "ok",
  "finish_reason": "eos",
  "generation_status": "success",
  "prompt_token_count": 83,
  "completion_token_count": 451,
  "completion_sha256": "<64-character-hex>",
  "created_at": "2026-10-12T12:00:00Z"
}
```

枚举：

- `format_status`: `ok | missing_think_end | missing_final | empty_output | malformed`。
- `finish_reason`: `eos | length | unknown`。若生成达到 `max_new_tokens` 且无明确 EOS，标记 `length`；**不得推断已经完成**。
- `generation_status`: `success | failed`（失败可以只记录在 event log，续跑时单独重试）。

核心规则：**`raw_completion` 永远不能被覆盖**。`reasoning_text` 与 `final_text` 仅是从原始 completion 解析出的便捷字段。若无法识别 think 结构，将原始信息保留并设置 `format_status`，不要强制伪造推理部分。对截断的输出标记并从有效池排除，但原始样本仍保存。

### 2.3 验证结果结构：`verification/*_verified.jsonl`

每条原始轨迹恰好对应一条验证结果：

```json
{
  "trajectory_id": "math_qwen3_17b_v1:math_algebra_train_000123_ab12cd34:s00",
  "problem_id": "math_algebra_train_000123_ab12cd34",
  "subset": "seed_train",
  "predicted_answer_text": "\\frac{1}{2}",
  "answer_extraction_status": "ok",
  "gold_parse_status": "ok",
  "pred_parse_status": "ok",
  "verification_status": "correct",
  "is_correct": true,
  "verification_method": "math_verify",
  "verification_error": null,
  "verifier_config_sha256": "<64-character-hex>"
}
```

- `verification_status`: **`correct | incorrect | unverifiable`**。
- `is_correct`: `true | false | null`，分别与上述三态一一对应。
- `answer_extraction_status`: `ok | missing | ambiguous | malformed`。
- 解析报错/超时/黄金答案异常统一标 `unverifiable` 且记录原因，不作为错误答案。
- 输出答案只从 `final_text` 中提取，绝不从 `<think>` 中找候选答案。

### 2.4 候选池结构：`pool/*`

`eligible_questions.jsonl` 每行：

```json
{
  "problem_id": "math_algebra_train_000123_ab12cd34",
  "level": 2,
  "topic": "Algebra",
  "n_generated": 6,
  "n_correct": 4,
  "n_distinct_correct": 3,
  "n_incorrect": 1,
  "n_unverifiable": 1,
  "eligible": true,
  "distinct_correct_trajectory_ids": ["...:s00", "...:s02", "...:s05"]
}
```

`selected_questions.jsonl` 为冻结后的主实验题目清单，字段至少包括 `problem_id, level, topic, selection_seed, selection_rule`；`candidates_correct.jsonl` 为所选题目上的全部**正确且已去重**轨迹及其原始/验证字段（或指向两表的 ID）；`candidates_all.jsonl` 为所选题目上的所有生成成功轨迹，包含错误、未验证及被判近重复的输出。`pool_manifest.json` 存储来源文件 SHA256、题目数、候选数、规则、配置版本、生成时间。

---

## 3. 模块 A：下载、清洗和划分题目

**实现：** `pipeline/prepare_dataset.py`。

### 3.1 读取与标准化

1. 用 `datasets.load_dataset(repo_id, name=<config>, split="train")` **逐个读取上述 7 个 config**，拼接成原始题表，并保存原始索引和来源信息；不得默认单个 config 代表整个 MATH 数据集。
2. 校验 `problem / solution / level / type` 都存在，空题干、空 solution 进入排除审计。
3. 解析 `level`：支持 `Level 1` ～ `Level 5` 字符串；保留来源主题。
4. 按规范化题干哈希去重；同一题出现于多个主题时，选择稳定的一条来源并写入 dedup 审计。
5. 若 `exclude_asymptote: true`，以 `\\begin{asy}`、`[asy]` 等明确图形脚本标记排除视觉依赖题；**不能把全部 Geometry 题剔除**。
6. 从 `solution` 提取最后一个结构完备的 `\\boxed{...}` 答案。若多答案歧义/缺失，仍记录审计，但默认不纳入可自动验证的本期样本集合。
7. 统计各 `(level, topic)` 的可用题量，以**受题库支持的配额**实施分层抽样；对无法满足的配额，记录短缺并按固定规则按同级其它主题、相邻难度重新分配，绝不复制题目充数。
8. 按 `split_seed=42` 划分 `seed_train=2500`、`anchor=300`、`dev=200`；完整保存划分 manifest 和各层实际数量。

建议先划分全部 3,000 题（分层抽样），再由确定的第二个 seeded shuffle 分配到三个子集，最后检查每个子集难度与主题比例。也可以分别按配额抽样，但实现必须固定且记录策略。

**补充**：若某类答案提取失败较多，统计其主题/难度分布；这样做会引入样本偏差，报告中不能忽略。

### 3.2 验收条件

- 三个子集分别恰好 `2500 / 300 / 200`；如实际有效题不足则报错并提供统计，不允许静默缩小。
- `problem_id` 唯一；`problem_sha256` 在三子集之间无交集。
- split 文件中不存在 `solution`、`gold_answer` 等字段。
- `private/gold_answers.jsonl` 与三个 split 中题目以 `problem_id` 一一匹配。
- 重复运行同一配置应得到相同题目列表、相同顺序及相同哈希。
- 输出数据量、层级分布、过滤原因与数量报告。

---

## 4. 模块 B：Qwen3 Thinking 模式生成轨迹

**实现：** `pipeline/generate.py` + `pipeline/parse_output.py`。

### 4.1 固定 Prompt

版本 `math_cot_v1`，只包含题干，不包含官方答案：

```text
Solve the following math problem step by step.
Show the necessary mathematical reasoning.
Put your final answer in \boxed{}.

Problem: {problem}
```

消息格式采用一个 user message（不额外加入系统提示），通过 Qwen tokenizer 的 `apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)` 构造输入。不要手工拼接 Qwen3 内部控制 token。

采用模型官方 Thinking 采样建议：`temperature=0.6`、`top_p=0.95`、`top_k=20`、`do_sample=True`、`max_new_tokens=2048`。先做预实验确认长度截断比例，再决定是否启动**新配置 run**增加长度；不能中途改变正式 run 配置。

### 4.2 推理和随机性

1. 按 `problem_id` 固定排序遍历，`sample_index=0..5`。
2. 每条轨迹的 `sampling_seed` 由 `SHA256(master_seed | run_id | problem_id | sample_index)` 稳定映射为整数；**不要使用 Python 内建 `hash()`**（它可能随进程变化）。
3. 第一版 `batch_size=1`，每次生成前设置 Python/NumPy/PyTorch/CUDA 所需随机种子；记录实际推理设备、dtype 和 backend。相同软件及硬件条件下争取可重现，但不承诺跨 GPU/版本 bitwise 一致。
4. 只解码新增 completion token，不把输入 prompt 重复存为 `raw_completion`；统计真实新增 token 数。
5. 不使用 `skip_special_tokens=True` 丢掉控制符后再识别 think/final；先保留原始完整解码，再进行解析。对非内容性的 EOS 做标记处理。
6. Qwen3 thinking 的输出可能包含 `<think>...</think>`。严格依据标记分离 `reasoning_text` 与 `final_text`；若缺少结束标记或 final 为空，设置对应 `format_status`，不得将整段错误当作最终回答。
7. 采样故障写入 `generation_events.jsonl`，记录时间、ID、异常类别及重试次数；OOM 时给出清晰错误。不要在脚本内部不留记录地动态减小 token 长度。
8. 每成功完成一条采样即追加并 flush 到 JSONL；支持 `--resume`：扫描已有文件的 `trajectory_id`，仅跳过 schema 完整且 `generation_status=success` 的记录；失败记录按显式 retry 策略重试。
9. 写 JSONL 时采取单进程写入或队列 writer，避免多个 worker 同时破坏文件。实现不完整末行检测和恢复日志；不要直接将已生成结果覆盖。

### 4.3 三种运行模式

- `pilot`：从 `dev` 固定选取 100 道题，每题 6 条；可用于检查 Prompt、截断率、输出结构和验证能力；**不得用于正式训练池**。
- `seed_train`：对全部 2,500 道正式种子题各采样 6 条；生成完成后冻结原始文件。
- `anchor`：对 300 道锚点题各采样 6 条；本期只生成和验证，不做认知标注。

### 4.4 预实验通过标准（建议门槛，而非统计结论）

- 100 道 × 6 = 600 条生成任务有明确终态（成功或带原因失败），正常成功率争取达到 99%。
- **有效 `final_text`** 占比目标 >95%；严重不达标时优先修复模板/解析逻辑。
- 统计 `completion_token_count` 的 P50/P90/P95、`finish_reason=length` 比例。
- 手工检查至少 20 条不同格式输出，确认 thinking 内容、最终答案和输出结束条件正确解析。
- 报告每题正确轨迹数 0/1/2/.../6 的分布，以评估主实验有效题是否可能达到 600 道。暂不依据某一种筛选算法的预期收益挑选题目。

---

## 5. 模块 C：数学答案提取与验证

**实现：** `pipeline/verify.py`。

### 5.1 判定顺序

1. 只读 `raw_completion/final_text`、`gold_answers.jsonl` 和配置；不得调用任何生成模型。
2. 如果 `generation_status != success`、`finish_reason=length`（可能截断）、`final_text` 缺失或 `format_status` 非 `ok`：默认为 `unverifiable`，详细标明原因。
3. 优先从最终回答最后的 `\boxed{...}` 提取预测答案（支持嵌套大括号）。若无 boxed，按明确的 final-answer 格式退化提取；不要扫描 reasoning 文本。
4. 使用 `math_verify.parse(...)` 分别解析 `gold`、预测答案，然后 `math_verify.verify(gold, prediction)`；参数/提取策略固定在 verifier 配置中。**`verify(gold, prediction)` 的参数顺序不可反转。**
5. Gold 或 prediction 解析失败、数学校验报错/超时、多个答案存在歧义，一律记录为 `unverifiable`，而不是 `incorrect`。
6. 两侧成功解析，明确等价为 `correct`，不等价为 `incorrect`。
7. 额外支持 `--manual-review-export`：导出 `unverifiable` 和分层抽样的 `correct/incorrect`，供人工复核；人工修订写入单独审计文件，不改写原始生成数据。

### 5.2 必需测试样例

必须覆盖至少下列情况：

- `\boxed{1/2}` 与 `\boxed{\frac12}` 等价。
- 复数、多项式、集合/区间等 parser 支持的不同数学表达方式。
- 原始 `solution` 包含嵌套花括号。
- 多个 `\boxed{}`，只选择最终回答中合适的末次答案，并标记有歧义时的处理方式。
- 模型只在 `<think>` 中出现正确答案，最终回答错误：判为错误（不能从 think 取答案）。
- 无法解析的 `gold`、截断输出、缺失 `</think>`、空回答、数学 parser 异常：`unverifiable`。
- 重跑验证不重新生成模型输出，结果只依赖相同输入与 verifier 版本。

### 5.3 人工抽检

从 `correct`、`incorrect` 和 `unverifiable` 三类中分层抽取至少 100 条，共享审计表记录人工判断（不是将人工判断混入模型输入）；同时记录主要解析错误类型和不同数学主题的错误分布。若 `correct` 有明显误判，先修复验证器，再冻结候选池。

---

## 6. 模块 D：去重、有效题判定与冻结轨迹池

**实现：** `pipeline/build_pool.py`。只处理 `seed_train` 的结果；`anchor` 和 `dev` 不得纳入主训练候选池。

### 6.1 候选资格

一条轨迹成为“可选正确轨迹”的必要条件：

```text
generation_status == "success"
AND format_status == "ok"
AND finish_reason == "eos"
AND verification_status == "correct"
AND reasoning_text 非空
AND final_text 非空
AND completion_token_count > 0
```

注意：这只能保证输出结构完整和最终答案正确，**不意味着推理过程正确**。有过程缺陷但答案正确的轨迹必须保留，供后续认知筛选研究。

### 6.2 同题去重规则

1. **仅在同一 `problem_id` 的正确轨迹中去重**，不能跨题删除轨迹。
2. 第一级：对 `reasoning_text + final_text` 进行轻度文本标准化（Unicode NFC、空白规范化），按 SHA256 去除完全相同的输出。
3. 第二级：用 token 的 5-gram Jaccard 相似度识别近重复（初始阈值 `>=0.90`，可配置）；先只用于形成“有效不同轨迹”的判定与审计，**不得将相似但推理策略不同的轨迹不可逆删除**。
4. 同一重复组保留确定性的代表（优先非截断、格式完整；同等条件选字典序最小 `trajectory_id`），同时记录所有成员及相似度。
5. 后续若发现近重复误伤，可依据完整原始轨迹池重建；原始记录永不删除。

`distinct_correct` 是文本多样性的操作性定义，而不是已经通过 Judge 确认的真正逻辑策略多样性。报告必须区分两者。

### 6.3 有效题判定与固定选题

计算每题 `n_correct`、`n_distinct_correct`；有效题定义为 `n_distinct_correct >= 2`。

- 若有效题数量 `>= 600`：按 `level × topic` 分层随机、固定 `pool_seed` 抽取 **600 道**题，并把所有已去重的正确候选轨迹全部纳入候选池。不得按照最高正确率、最短回答或任何未来认知评分排序。
- 若有效题数量 `< 600`：输出缺口、难度分布和建议；**不允许静默补入只有 1 条正确轨迹的题目**。可启动一个新的预先记录的扩容 run，统一增加每题采样数（例如从 6 到 8）或补充从未使用的 MATH Train 题目；扩容需保留每个样本来源和生成版本并冻结新的 pool manifest。
- 如果预算有限，可显式改为 400 题的**新实验配置**；所有后续方法仍需在这同一批题上比较。

需要生成：

- `eligible_questions.jsonl`：全部符合条件的题目，以及每题数量分布。
- `selected_questions.jsonl`：确定的 600 题主实验清单。
- `candidates_correct.jsonl`：所选题目中全部、去重后的正确轨迹（下游筛选的实际输入）。
- `candidates_all.jsonl`：所选题目中所有完整原始轨迹及验证状态，包括错误、无法验证和重复样本，供分析使用。
- `dedup_audit.jsonl`：重复组、保留轨迹 ID、删除候选的原因；**原始数据不删除**。
- `pool_manifest.json`：配置、模型版本、输入与输出 SHA256、候选统计与 pool 冻结时间。

### 6.4 冻结规则

最终 `build_pool` 执行 `--freeze` 后写只读/带哈希 manifest，后续 Random / Rubric / MIRT 不得改变候选题目、候选轨迹集合或重跑生成。任何变更均须给 pool 新版本和新的 `pool_id`。

---

## 7. 命令行接口（代码生成工具必须按此实现）

下面命令为**目标接口定义**，不是已存在的脚本：

```bash
# 0. 安装环境后校验配置（仅检查，不触发下载或推理）
python -m pipeline.prepare_dataset --config configs/experiment1.yaml --validate-only

# 1. 下载、清洗、划分，生成 seed_train / anchor / dev
python -m pipeline.prepare_dataset --config configs/experiment1.yaml

# 2. 开发预实验：dev 中固定 100 题，每题 6 次
python -m pipeline.generate --config configs/experiment1.yaml --subset pilot --resume
python -m pipeline.verify   --config configs/experiment1.yaml --subset pilot
python -m pipeline.report   --config configs/experiment1.yaml --stage pilot

# 3. 正式生成：训练种子集和独立 anchor 集
python -m pipeline.generate --config configs/experiment1.yaml --subset seed_train --resume
python -m pipeline.generate --config configs/experiment1.yaml --subset anchor --resume

# 4. 独立验证（不调用生成模型）
python -m pipeline.verify --config configs/experiment1.yaml --subset seed_train
python -m pipeline.verify --config configs/experiment1.yaml --subset anchor

# 5. 构建并冻结轨迹池
python -m pipeline.build_pool --config configs/experiment1.yaml --freeze
python -m pipeline.report --config configs/experiment1.yaml --stage all

# 6. 自动化测试
pytest -q
```

统一 CLI 规范：`--config` 必填；`--subset` 枚举 `pilot | seed_train | anchor`；`--resume` 只用于生成；加 `--limit N` 作为调试开关，但有 `--limit` 的输出不得被正式 `--freeze`；`--backend mock` 可用于 CPU 端到端测试；任何缺失输入、配置冲突或数据损坏返回非 0 exit code 并显示原因。

---

## 8. 报表和质量控制

`report.py` 至少输出以下指标（JSON + 终端可读摘要）：

**数据准备：**
- 原始 MATH 题量、各 config 题量、各过滤原因数、可用题数；
- 3 个 split 的题量、难度/主题分布、重复 ID/重复题干数；
- gold 答案抽取成功率与无法抽取的主要类别。

**生成：**
- 预期与实际生成数，失败数，断点续跑跳过数；
- 平均及 P50/P90/P95 token 数、截断比例、格式异常比例；
- 生成耗时与 token/s；各难度/主题产生的轨迹数。

**验证：**
- `correct/incorrect/unverifiable` 总数和比例；
- 每道题 0/1/2/.../6 条正确轨迹的题目数；
- 每个难度/主题的答案正确率与解析失败率；
- 人工抽检错误类别、数量及纠正记录。

**候选池：**
- 有效题目总数、最终冻结题数、各难度/主题分布；
- 正确轨迹总数、去重后数、平均每题候选轨迹数；
- 完全重复数、近重复数、冻结 manifest 文件哈希。

在报告顶部明确本期 `verification_status=correct` 只表示**最终答案正确**，不是过程正确性标注。

---

## 9. 自动化测试与验收标准

### 9.1 单元测试（全部必须写）

| 测试文件 | 必须覆盖的行为 |
| --- | --- |
| `test_prepare_dataset.py` | 7 config 合并、`Level N` 解析、交叉 split 无重叠、无 gold 泄漏、嵌套 boxed、种子固定 |
| `test_parse_output.py` | 完整 think、缺失 `</think>`、无 final、空字符串、EOS/length、保留 raw completion |
| `test_verify.py` | 数学等价、明显不等价、解析失败三态、gold/pred 顺序、不能扫描 think、不会混入 gold |
| `test_dedup.py` | 同题重复去除、跨题不去重、确定性代表选择、保留去重审计 |
| `test_resume.py` | 中断续跑无重复 ID、失败任务重试、配置变化禁止混入、半行损坏检测 |

### 9.2 Smoke test（不用 GPU 即可跑一部分）

- 提供 3 道 mock 题 × 3 条固定生成结果的数据夹具；其中必须覆盖正确、错误、unverifiable、重复答案和相同题的不同推理轨迹。
- 执行 `mock generate → verify → build_pool → report` 完整流程，验证 JSON schema、文件关联和 manifest。
- 有 GPU 时再运行 `--limit 2` 的真实 Qwen3 输出，确认 reasoning/final 解析无误。

### 9.3 阶段验收门槛

- [ ] `seed_train=2500`，`anchor=300`，`dev=200`，互斥、可复现。
- [ ] 预实验完成 100×6 采样，已有正确率/截断率/有效题率报告。
- [ ] 正式轨迹原始记录齐全，且每条有稳定 `trajectory_id` 和版本信息。
- [ ] 验证结果三态完备；`unverifiable` 不被误记为错误。
- [ ] 所有脚本支持 `--config`，生成模块支持安全 `--resume`。
- [ ] 原始输出不被删除，所有阶段能通过 ID 回溯到题干及模型生成设置。
- [ ] 固定的正确候选池和目标题目清单已 `--freeze`，具有 manifest SHA256。
- [ ] 通过 `pytest` 及完整 mock 流程；报告输出满足第 8 节。

**不达标时严禁伪造 600 道有效题**：如实际达不到，提交真实数量和瓶颈统计，并生成需显式启动的新配置扩容计划。

---

## 10. 给代码生成工具的分步实施任务

不要一次性生成整个系统。按下列顺序开发；前一个阶段通过测试后才能继续下一个。

**Task 1 — 基础工程与数据准备**
- 建立上述目录、YAML 解析、通用日志与 JSONL I/O。
- 实现 `prepare_dataset.py` 和题目/gold schemas。
- 先写并通过数据集分层、数据泄漏、ID 稳定性测试。
- 交付：3 个 split + 私有 gold + `dataset_report.json`。

**Task 2 — 轨迹生成与输出解析**
- 实现 `parse_output.py`（先用 mock fixtures 测试）。
- 实现 `generate.py`（先 mock，再接 Qwen3 HF Transformers），严格保留 `raw_completion`。
- 实现确定性采样种子、事件日志和断点续跑。
- 交付：`pilot_dev_raw.jsonl` + `pilot_report.json`。

**Task 3 — 自动验证**
- 实现 gold/pred 的 LaTeX 提取、`math_verify.parse/verify`、三态结果与异常日志。
- 使用人工构造的数学校验 fixtures 执行测试，再对 pilot 数据全量验证。
- 交付：`pilot_dev_verified.jsonl` + `verification_report.json`。

**Task 4 — 正式轨迹生成**
- 确认 pilot 的输出格式、截断率和正确轨迹分布满足实验可行性；冻结 prompt/config。
- 分别生成 `seed_train` 和 `anchor`，允许续跑，不交叉修改。
- 独立生成验证表；保留原始错误/无法解析样本。
- 交付：4 份正式 raw/verified JSONL 与运行记录。

**Task 5 — 候选池构建与冻结**
- 实现正确轨迹资格、近重复审计、题目有效性与分层固定抽样。
- 输出正确候选池、全量候选关联、池报告及 pool manifest。
- 交付：正式可供轨迹评价/筛选的冻结候选数据集。

**Task 6 — 回归测试与 README**
- 补齐 pytest、mock end-to-end、命令说明、环境安装和常见错误恢复指南。
- `README.md` 列出每个命令的输入/输出路径与预期完成条件；不要把实验运行命令与数据文件契约写成两套不一致的版本。

### 给代码生成工具的总约束

> 请按照本计划逐模块实现可执行 Python 项目。严格遵守配置文件、CLI、JSONL schema 和输出目录契约。先写 tests，再写对应脚本；不得擅自增加认知打分、轨迹排序或 SFT。不要声称脚本执行成功，除非实际运行并通过测试。对于无法运行的真实模型/数据下载环节，用 mock fixtures 验证接口并明确标记尚未真实验证的部分。

---

## 11. 外部规范与参考地址

- Qwen3-1.7B 模型卡（Thinking 模式、官方推荐采样参数）：https://huggingface.co/Qwen/Qwen3-1.7B
- MATH 数据集及 7 个配置说明：https://huggingface.co/datasets/EleutherAI/hendrycks_math
- Math-Verify 使用与答案等价验证：https://github.com/huggingface/Math-Verify
- Hugging Face Datasets 文档：https://huggingface.co/docs/datasets/
- Transformers 文档：https://huggingface.co/docs/transformers/

> 文档界限提醒：本计划是**可执行脚本的开发规格**，不是已经运行的实验结果。所有样本数量中，15,000 和 1,800 表示配置下的计划生成任务数；600 道有效题是目标而不是对实际生成成功率的保证。
