from pipeline.prompts import build_prompt
from pipeline.sft import training_prompt


EXAMPLES = [{"question": "Demo?", "answer": "Demo work.\n#### 1"}]


def test_plain_gptj_prompt_and_rationalization_hint():
    ordinary = build_prompt("Target?", EXAMPLES)
    hinted = build_prompt("Target?", EXAMPLES, gold_hint="7")
    assert ordinary.endswith("Q: Target?\nA:")
    assert "correct answer is" not in ordinary
    assert "The correct answer is 7" in hinted


def test_sft_prompt_removes_rationalization_hint():
    prompt = training_prompt("Target?", "gsm8k_star_v1", EXAMPLES)
    assert prompt.endswith("Q: Target?\nA:")
    assert "correct answer is" not in prompt
