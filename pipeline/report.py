"""Build JSON and terminal summaries for pipeline quality control."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import PipelineError, canonical_json, experiment_root, load_config, read_jsonl, sha256_file, utc_now, write_json_atomic


NOTICE = (
    "verification_status=correct means only that the final answer was verified as equivalent; "
    "it is not a process-correctness annotation."
)


def percentile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_generation(rows: list[Mapping[str, Any]], questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    tokens = [int(row.get("completion_token_count") or 0) for row in rows]
    formats = collections.Counter(str(row.get("format_status")) for row in rows)
    finishes = collections.Counter(str(row.get("finish_reason")) for row in rows)
    strata = collections.Counter(
        f"L{questions[row['problem_id']]['level']}|{questions[row['problem_id']]['topic']}"
        for row in rows if row["problem_id"] in questions
    )
    return {
        "actual_success_records": len(rows),
        "completion_tokens": {
            "mean": sum(tokens) / len(tokens) if tokens else None,
            "p50": percentile(tokens, 0.50), "p90": percentile(tokens, 0.90), "p95": percentile(tokens, 0.95),
        },
        "finish_reason_counts": dict(sorted(finishes.items())),
        "length_truncation_rate": finishes["length"] / len(rows) if rows else None,
        "format_status_counts": dict(sorted(formats.items())),
        "format_abnormal_rate": sum(count for key, count in formats.items() if key != "ok") / len(rows) if rows else None,
        "level_topic_counts": dict(sorted(strata.items())),
    }


def summarize_verification(
    rows: list[Mapping[str, Any]], questions: Mapping[str, Mapping[str, Any]], n_per_problem: int
) -> dict[str, Any]:
    statuses = collections.Counter(str(row["verification_status"]) for row in rows)
    correct_by_problem = collections.Counter(
        row["problem_id"] for row in rows if row["verification_status"] == "correct"
    )
    distribution = collections.Counter(
        correct_by_problem.get(problem_id, 0) for problem_id in {row["problem_id"] for row in rows}
    )
    strata: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in rows:
        question = questions.get(row["problem_id"])
        if question:
            key = f"L{question['level']}|{question['topic']}"
            strata[key][row["verification_status"]] += 1
    stratum_summary = {}
    for key, value in sorted(strata.items()):
        total = sum(value.values())
        stratum_summary[key] = {
            "counts": dict(sorted(value.items())),
            "correct_rate": value["correct"] / total if total else None,
            "unverifiable_rate": value["unverifiable"] / total if total else None,
        }
    return {
        "total": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "status_rates": {key: count / len(rows) for key, count in sorted(statuses.items())} if rows else {},
        "correct_trajectories_per_problem": {
            str(index): distribution.get(index, 0) for index in range(n_per_problem + 1)
        },
        "by_level_topic": stratum_summary,
    }


def _optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path)) if path.exists() else []


def generate_report(config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    root = experiment_root(config)
    report: dict[str, Any] = {"notice": NOTICE, "stage": stage, "created_at": utc_now()}
    dataset_report = _optional_json(root / "reports" / "dataset_report.json")
    if dataset_report:
        report["dataset"] = dataset_report

    subsets = ["pilot"] if stage == "pilot" else ["pilot", "seed_train", "anchor"]
    generation: dict[str, Any] = {}
    verification: dict[str, Any] = {}
    events = _optional_jsonl(Path(config["paths"]["run_root"]) / "generation_events.jsonl")
    for subset in subsets:
        split_name = "dev" if subset == "pilot" else subset
        prefix = "pilot_dev" if subset == "pilot" else subset
        question_rows = _optional_jsonl(root / "splits" / f"{split_name}.jsonl")
        questions = {row["problem_id"]: row for row in question_rows}
        raw_path = root / "trajectories" / f"{prefix}_raw.jsonl"
        verified_path = root / "verification" / f"{prefix}_verified.jsonl"
        raw_rows = _optional_jsonl(raw_path)
        verified_rows = _optional_jsonl(verified_path)
        if raw_rows:
            generation[subset] = summarize_generation(raw_rows, questions)
            generation[subset]["source_sha256"] = sha256_file(raw_path)
            expected_questions = min(len(question_rows), int(config["pilot"]["n_problems"])) if subset == "pilot" else len(question_rows)
            generation[subset]["expected_tasks"] = expected_questions * int(config["sampling"]["n_per_problem"])
            generation[subset]["missing_success_records"] = generation[subset]["expected_tasks"] - len(raw_rows)
            summaries = [event for event in events if event.get("event") == "generation_run_summary" and event.get("subset") == subset]
            if summaries:
                generation[subset]["last_run_summary"] = summaries[-1]
            success_ids = {row["trajectory_id"] for row in raw_rows}
            terminal_failures = {
                event["trajectory_id"] for event in events
                if event.get("event") == "generation_terminal_failure"
                and event.get("trajectory_id") not in success_ids
                and any(f":{problem_id}:" in event.get("trajectory_id", "") for problem_id in questions)
            }
            generation[subset]["unrecovered_terminal_failure_count"] = len(terminal_failures)
            subset_success_events = [
                event for event in events
                if event.get("event") == "generation_success"
                and event.get("trajectory_id") in success_ids
            ]
            elapsed = sum(float(event.get("elapsed_seconds") or 0) for event in subset_success_events)
            event_tokens = sum(int(event.get("completion_token_count") or 0) for event in subset_success_events)
            generation[subset]["recorded_generation_seconds"] = elapsed
            generation[subset]["recorded_completion_tokens_per_second"] = event_tokens / elapsed if elapsed else None
        if verified_rows:
            verification[subset] = summarize_verification(
                verified_rows, questions, int(config["sampling"]["n_per_problem"])
            )
            verification[subset]["source_sha256"] = sha256_file(verified_path)
    report["generation"] = generation
    report["verification"] = verification

    if stage == "all":
        pool_report = _optional_json(root / "reports" / "pool_report.json")
        pool_manifest = _optional_json(root / "pool" / "pool_manifest.json")
        report["pool"] = pool_report
        report["pool_manifest"] = pool_manifest
        destination = root / "reports" / "verification_report.json"
    else:
        destination = root / "reports" / "pilot_report.json"
    write_json_atomic(destination, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("pilot", "all"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = generate_report(load_config(args.config), args.stage)
    print(NOTICE)
    print(canonical_json({
        "stage": report["stage"],
        "generation_subsets": sorted(report["generation"]),
        "verification_subsets": sorted(report["verification"]),
        "pool_available": bool(report.get("pool")),
    }))


if __name__ == "__main__":
    main()
