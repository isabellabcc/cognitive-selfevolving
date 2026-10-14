"""Extract final answers and verify them with math-verify."""

from __future__ import annotations

import argparse
import collections
import random
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import (
    PipelineError,
    canonical_json,
    experiment_root,
    load_config,
    read_jsonl,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)
from .prepare_dataset import extract_balanced_boxed


VERIFIER_CONFIG = {
    "version": 1,
    "prediction_source": "final_text_only",
    "boxed_policy": "one_distinct_last_balanced",
    "fallback_patterns": ["final_answer_label", "answer_label"],
    "argument_order": "verify(gold, prediction)",
    "truncated_is_unverifiable": True,
}


def extract_prediction(final_text: str) -> tuple[str | None, str]:
    boxes = extract_balanced_boxed(final_text)
    if boxes:
        distinct = list(dict.fromkeys(boxes))
        if len(distinct) > 1:
            return boxes[-1], "ambiguous"
        if not boxes[-1]:
            return None, "malformed"
        return boxes[-1], "ok"
    patterns = (
        r"(?is)(?:final\s+answer)\s*(?:is|:|=)\s*(.+?)\s*$",
        r"(?is)(?:answer)\s*(?:is|:|=)\s*(.+?)\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, final_text)
        if match and match.group(1).strip():
            return match.group(1).strip(), "ok"
    return None, "missing"


def load_math_verifier() -> tuple[Callable[[str], Any], Callable[[Any, Any], bool], str]:
    try:
        import math_verify
        from math_verify import parse, verify
    except ImportError as exc:
        raise PipelineError("math-verify is required for answer verification") from exc
    return parse, verify, getattr(math_verify, "__version__", "unknown")


def _unverifiable(raw: Mapping[str, Any], error: str, verifier_hash: str, **extra: Any) -> dict[str, Any]:
    return {
        "trajectory_id": raw["trajectory_id"],
        "problem_id": raw["problem_id"],
        "subset": raw["subset"],
        "predicted_answer_text": extra.get("predicted_answer_text"),
        "answer_extraction_status": extra.get("answer_extraction_status", "missing"),
        "gold_parse_status": extra.get("gold_parse_status", "not_attempted"),
        "pred_parse_status": extra.get("pred_parse_status", "not_attempted"),
        "verification_status": "unverifiable",
        "is_correct": None,
        "verification_method": "math_verify",
        "verification_error": error,
        "verifier_config_sha256": verifier_hash,
    }


def verify_record(
    raw: Mapping[str, Any], gold: Mapping[str, Any], parse_fn: Callable[[str], Any],
    verify_fn: Callable[[Any, Any], bool], verifier_hash: str,
) -> dict[str, Any]:
    if raw.get("generation_status") != "success":
        return _unverifiable(raw, "generation_not_successful", verifier_hash)
    if raw.get("finish_reason") == "length":
        return _unverifiable(raw, "truncated_at_max_new_tokens", verifier_hash)
    if raw.get("format_status") != "ok":
        return _unverifiable(raw, f"invalid_format:{raw.get('format_status')}", verifier_hash)
    final_text = str(raw.get("final_text") or "")
    if not final_text.strip():
        return _unverifiable(raw, "missing_final_text", verifier_hash)
    predicted, extraction_status = extract_prediction(final_text)
    if extraction_status != "ok" or predicted is None:
        return _unverifiable(
            raw, f"answer_extraction:{extraction_status}", verifier_hash,
            predicted_answer_text=predicted, answer_extraction_status=extraction_status,
        )
    if gold.get("gold_extraction_status") != "ok" or not gold.get("gold_answer"):
        return _unverifiable(
            raw, f"gold_extraction:{gold.get('gold_extraction_status')}", verifier_hash,
            predicted_answer_text=predicted, answer_extraction_status="ok",
        )
    try:
        parsed_gold = parse_fn(str(gold["gold_answer"]))
        if parsed_gold is None or parsed_gold == []:
            raise ValueError("gold parser returned no expression")
    except Exception as exc:
        return _unverifiable(
            raw, f"gold_parse:{type(exc).__name__}:{exc}", verifier_hash,
            predicted_answer_text=predicted, answer_extraction_status="ok", gold_parse_status="parse_error",
        )
    try:
        parsed_prediction = parse_fn(predicted)
        if parsed_prediction is None or parsed_prediction == []:
            raise ValueError("prediction parser returned no expression")
    except Exception as exc:
        return _unverifiable(
            raw, f"prediction_parse:{type(exc).__name__}:{exc}", verifier_hash,
            predicted_answer_text=predicted, answer_extraction_status="ok",
            gold_parse_status="ok", pred_parse_status="parse_error",
        )
    try:
        equivalent = bool(verify_fn(parsed_gold, parsed_prediction))
    except Exception as exc:
        return _unverifiable(
            raw, f"verification:{type(exc).__name__}:{exc}", verifier_hash,
            predicted_answer_text=predicted, answer_extraction_status="ok",
            gold_parse_status="ok", pred_parse_status="ok",
        )
    return {
        "trajectory_id": raw["trajectory_id"],
        "problem_id": raw["problem_id"],
        "subset": raw["subset"],
        "predicted_answer_text": predicted,
        "answer_extraction_status": "ok",
        "gold_parse_status": "ok",
        "pred_parse_status": "ok",
        "verification_status": "correct" if equivalent else "incorrect",
        "is_correct": equivalent,
        "verification_method": "math_verify",
        "verification_error": None,
        "verifier_config_sha256": verifier_hash,
    }


def paths_for_subset(config: Mapping[str, Any], subset: str) -> tuple[Path, Path]:
    root = experiment_root(config)
    prefix = "pilot_dev" if subset == "pilot" else subset
    return root / "trajectories" / f"{prefix}_raw.jsonl", root / "verification" / f"{prefix}_verified.jsonl"


def export_manual_review(
    results: list[dict[str, Any]], raw_by_id: Mapping[str, Mapping[str, Any]], output: Path, seed: int
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for result in results:
        grouped[result["verification_status"]].append(result)
    selected = list(grouped["unverifiable"])
    rng = random.Random(seed)
    for status in ("correct", "incorrect"):
        rows = sorted(grouped[status], key=lambda row: row["trajectory_id"])
        rng.shuffle(rows)
        selected.extend(rows[:50])
    exported = []
    for result in sorted(selected, key=lambda row: row["trajectory_id"]):
        raw = raw_by_id[result["trajectory_id"]]
        exported.append({
            **result,
            "raw_completion": raw.get("raw_completion"),
            "reasoning_text": raw.get("reasoning_text"),
            "final_text": raw.get("final_text"),
            "manual_judgment": None,
            "manual_error_category": None,
            "reviewer_notes": None,
        })
    write_jsonl_atomic(output, exported)


def run_verification(
    config: Mapping[str, Any], subset: str, manual_review_export: str | None = None
) -> dict[str, Any]:
    raw_path, output_path = paths_for_subset(config, subset)
    raw_rows = list(read_jsonl(raw_path))
    ids = [row.get("trajectory_id") for row in raw_rows]
    if len(ids) != len(set(ids)):
        raise PipelineError(f"Duplicate trajectory_id in {raw_path}")
    gold_rows = list(read_jsonl(experiment_root(config) / "private" / "gold_answers.jsonl"))
    gold_by_id = {row["problem_id"]: row for row in gold_rows}
    missing = sorted({row["problem_id"] for row in raw_rows} - set(gold_by_id))
    if missing:
        raise PipelineError(f"Gold records missing for {len(missing)} problems; first: {missing[0]}")
    parse_fn, verify_fn, version = load_math_verifier()
    verifier_config = {**VERIFIER_CONFIG, "math_verify_version": version}
    verifier_hash = sha256_text(canonical_json(verifier_config))
    results = [
        verify_record(row, gold_by_id[row["problem_id"]], parse_fn, verify_fn, verifier_hash)
        for row in raw_rows
    ]
    write_jsonl_atomic(output_path, results)
    if manual_review_export:
        export_manual_review(
            results, {row["trajectory_id"]: row for row in raw_rows}, Path(manual_review_export),
            int(config["master_seed"]),
        )
    counts = collections.Counter(row["verification_status"] for row in results)
    report = {
        "subset": subset,
        "total": len(results),
        "counts": dict(sorted(counts.items())),
        "verifier_config": verifier_config,
        "verifier_config_sha256": verifier_hash,
    }
    write_json_atomic(experiment_root(config) / "reports" / "verification_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subset", required=True, choices=("pilot", "seed_train", "anchor"))
    parser.add_argument("--manual-review-export", nargs="?", const="manual_review.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_verification(load_config(args.config), args.subset, args.manual_review_export)
    print(canonical_json(report))


if __name__ == "__main__":
    main()
