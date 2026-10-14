from fractions import Fraction

from pipeline.verify import extract_prediction, verify_record


def parse_fraction(value):
    normalized = value.replace(r"\frac12", "1/2").replace(r"\frac{1}{2}", "1/2")
    return Fraction(normalized)


def raw(final=r"\boxed{1/2}", **updates):
    value = {
        "trajectory_id": "r:p:s00", "problem_id": "p", "subset": "seed_train",
        "generation_status": "success", "finish_reason": "eos", "format_status": "ok",
        "final_text": final, "reasoning_text": "reasoning says 99",
    }
    value.update(updates)
    return value


def gold(answer="1/2", status="ok"):
    return {"gold_answer": answer, "gold_extraction_status": status}


def test_math_equivalence_and_obvious_inequality():
    equivalent = verify_record(raw(r"\boxed{\frac12}"), gold(), parse_fraction, lambda g, p: g == p, "h")
    wrong = verify_record(raw(r"\boxed{2}"), gold(), parse_fraction, lambda g, p: g == p, "h")
    assert equivalent["verification_status"] == "correct"
    assert wrong["verification_status"] == "incorrect"


def test_gold_is_first_verify_argument():
    observed = []
    result = verify_record(raw(r"\boxed{2}"), gold("1"), int, lambda g, p: observed.append((g, p)) or False, "h")
    assert result["verification_status"] == "incorrect"
    assert observed == [(1, 2)]


def test_never_scans_thinking_and_parse_failures_are_unverifiable():
    result = verify_record(raw(r"\boxed{2}", reasoning_text=r"correct is \boxed{1/2}"), gold(), parse_fraction, lambda g, p: g == p, "h")
    assert result["verification_status"] == "incorrect"
    failed = verify_record(raw(r"\boxed{not-math}"), gold(), parse_fraction, lambda g, p: g == p, "h")
    assert failed["verification_status"] == "unverifiable"
    assert failed["is_correct"] is None


def test_bad_format_and_truncation_are_unverifiable():
    assert verify_record(raw(format_status="missing_think_end"), gold(), int, lambda g, p: True, "h")["verification_status"] == "unverifiable"
    assert verify_record(raw(finish_reason="length"), gold(), int, lambda g, p: True, "h")["verification_status"] == "unverifiable"


def test_multiple_distinct_boxes_are_ambiguous():
    assert extract_prediction(r"first \boxed{1}, finally \boxed{2}") == ("2", "ambiguous")
