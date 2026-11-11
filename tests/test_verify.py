from fractions import Fraction

from pipeline.verify import extract_prediction, verify_record


def raw(final="#### 1/2", **updates):
    value = {
        "trajectory_id": "r:p:s00", "problem_id": "p", "subset": "seed_train",
        "generation_status": "success", "finish_reason": "eos", "format_status": "ok",
        "final_text": final, "reasoning_text": "reasoning says 99",
    }
    value.update(updates)
    return value


def gold(answer="1/2"):
    return {"gold_answer": answer, "gold_extraction_status": "ok"}


def test_math_equivalence_and_wrong_answer():
    parse = lambda value: Fraction(value.replace(",", ""))
    equivalent = verify_record(raw(), gold(), parse, lambda g, p: g == p, "h")
    wrong = verify_record(raw("#### 2"), gold(), parse, lambda g, p: g == p, "h")
    assert equivalent["verification_status"] == "correct"
    assert wrong["verification_status"] == "incorrect"


def test_only_explicit_hash_marker_is_read():
    assert extract_prediction("#### 42") == ("42", "ok")
    assert extract_prediction("reasoning mentions 42") == (None, "missing")
    assert extract_prediction("#### 1\n#### 2") == ("2", "ambiguous")


def test_bad_format_and_truncation_are_unverifiable():
    assert verify_record(raw(format_status="missing_final"), gold(), int, lambda g, p: True, "h")["verification_status"] == "unverifiable"
    assert verify_record(raw(finish_reason="length"), gold(), int, lambda g, p: True, "h")["verification_status"] == "unverifiable"
