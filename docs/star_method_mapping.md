# STaR method mapping

| 论文方法 | 本仓库 | 说明 |
| --- | --- | --- |
| GPT-J 6B | `EleutherAI/gpt-j-6b` | 同一模型家族与规模，Transformers/PyTorch 加载 |
| GSM8K | `openai/gsm8k`, `main` | 官方 train/test；train 再固定拆分 few-shot/validation/generation/reserve |
| Few-shot rationale prompt | `pipeline.prompts` | 4 个固定 train 示例，纯文本 `Q:/A:` |
| Greedy rationale generation | `pipeline.generate` | K=1、`do_sample=false`，在完整 `#### answer` 行停止 |
| Answer filtering | `pipeline.verify` | 最终答案数学等价；不验证过程正确性 |
| Rationalization | `pipeline.rationalize` | 普通生成失败题输入 gold hint；SFT 输入移除 hint |
| Fine-tuning | `pipeline.sft` | 单卡 LoRA、completion-only loss；不是论文 TPU 全参数训练 |
| Subset comparison | `pipeline.star_pool` | full/random/exact dedup 读取同一完整内容哈希池 |
| Evaluation | `pipeline.evaluate_star` | validation/test greedy Pass@1，解析失败按错误计入 |

论文每个外循环使用上一轮模型生成数据，但用原始预训练模型重新训练当前轮模型。本项目的单轮所有选择条件同样从原始 GPT-J 独立初始化。`model.adapter_path` 只留给以后明确开展第二轮时使用，不能用于同轮条件之间串行续训。
