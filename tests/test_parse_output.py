from pipeline.parse_output import finish_reason, parse_completion


def test_complete_thinking_output_preserves_raw():
    raw = "<think>work</think> Final answer: \\boxed{2}"
    parsed = parse_completion(raw)
    assert parsed.raw_completion == raw
    assert parsed.reasoning_text == "work"
    assert parsed.final_text == "Final answer: \\boxed{2}"
    assert parsed.format_status == "ok"


def test_qwen_completion_with_opening_marker_in_prompt():
    raw = "reasoning emitted as new tokens</think> Final answer: \\boxed{2}"
    parsed = parse_completion(raw)
    assert parsed.reasoning_text == "reasoning emitted as new tokens"
    assert parsed.final_text == "Final answer: \\boxed{2}"
    assert parsed.format_status == "ok"


def test_missing_think_end_and_missing_final_and_empty():
    assert parse_completion("<think>unfinished").format_status == "missing_think_end"
    assert parse_completion("<think>work</think>  ").format_status == "missing_final"
    assert parse_completion("").format_status == "empty_output"
    assert parse_completion("answer only").format_status == "malformed"


def test_finish_reason_is_not_inferred_as_eos():
    assert finish_reason(generated_token_count=10, max_new_tokens=10, eos_reached=False) == "length"
    assert finish_reason(generated_token_count=3, max_new_tokens=10, eos_reached=True) == "eos"
    assert finish_reason(generated_token_count=3, max_new_tokens=10, eos_reached=False) == "unknown"
