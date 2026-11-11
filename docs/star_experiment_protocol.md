# Single-round experiment protocol

## Splits

- 4 个 GSM8K train 示例（source index 0、2、3、5，对应论文附录示例）只用于 few-shot prompt。
- 300 个 train 示例作为 validation。
- 4,000 个 train 示例用于本轮普通生成和 rationalization。
- 其余 train 示例为 reserve。
- 官方 1,319 test 仅在 validation 已确定最终条件后使用。

所有拆分由 seed 42 固定并保存 ID、内容哈希与 manifest。公开 split 不含 answer；few-shot rationale 独立存放；private gold 只供验证和 rationalization 使用。

## Fair comparison

普通池的 full、random 25/50/75% 和 exact dedup 必须拥有相同 `pool_hash`。模型、prompt、LoRA 配置、训练 seed 和评测设置保持一致。随机子集至少三个 seed；先报告每个 seed，再报告均值与离散程度。

`with_rationalization` 是另一个候选池，包含额外 gold-hint 生成开销。它与 ordinary 的比较属于端到端 STaR 生成机制比较，不是同池筛选质量比较。

## Reporting

同时报告候选轨迹数、正确率、选择数、唯一问题数、completion tokens、训练 tokens、步数、耗时、显存峰值、validation/test Pass@1 和解析失败率。最终答案正确不得表述为推理过程已验证。
