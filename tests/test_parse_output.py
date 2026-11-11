from pipeline.parse_output import finish_reason, parse_completion


def test_gsm8k_completion_is_split_at_explicit_marker():
    raw = "First compute 12 / 3 = 4.\n#### 4"
    parsed = parse_completion(raw)
    assert parsed.raw_completion == raw
    assert parsed.reasoning_text == "First compute 12 / 3 = 4."
    assert parsed.final_text == "#### 4"
    assert parsed.format_status == "ok"


def test_missing_empty_and_conflicting_finals_are_rejected():
    assert parse_completion("").format_status == "empty_output"
    assert parse_completion("reasoning only").format_status == "missing_final"
    assert parse_completion("work\n#### 1\nmore\n#### 2").format_status == "ambiguous_final"


def test_finish_reason_is_explicit():
    assert finish_reason(generated_token_count=10, max_new_tokens=10, eos_reached=False) == "length"
    assert finish_reason(generated_token_count=3, max_new_tokens=10, eos_reached=True) == "eos"
    assert finish_reason(generated_token_count=3, max_new_tokens=10, eos_reached=False) == "unknown"
