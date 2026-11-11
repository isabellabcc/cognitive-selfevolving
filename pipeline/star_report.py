"""Aggregate completed STaR selections, training runs, and evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .common import canonical_json, experiment_root, load_config, run_root, write_json_atomic


FIELDS = (
    "condition_id", "branch", "selector", "fraction", "seed", "pool_hash",
    "n_selected", "n_unique_problems", "selected_completion_tokens", "train_id",
    "train_tokens_seen_estimate", "global_steps", "train_seconds", "gpu_peak_gb",
    "validation_pass_at_1", "test_pass_at_1", "status",
)


def _json_files(root: Path, name: str) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in root.rglob(name)]


def collect_results(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    selections = _json_files(experiment_root(config) / "star" / "selections", "selection_manifest.json")
    trainings = _json_files(run_root(config) / "star", "train_manifest.json")
    evaluations = _json_files(run_root(config) / "star" / "evaluations", "metrics.json")
    training_by_selection: dict[str, list[dict[str, Any]]] = {}
    for training in trainings:
        training_by_selection.setdefault(training["selection_id"], []).append(training)
    rows = []
    for evaluation in sorted(evaluations, key=lambda item: (str(item.get("split")), str(item.get("eval_id")))):
        if evaluation.get("checkpoint") != "base":
            continue
        rows.append({
            "condition_id": "base",
            "branch": "none", "selector": "none", "fraction": 0,
            "seed": evaluation.get("eval_seed"), "pool_hash": None,
            "n_selected": 0, "n_unique_problems": 0,
            "selected_completion_tokens": 0, "train_id": None,
            "train_tokens_seen_estimate": 0, "global_steps": 0,
            "train_seconds": 0, "gpu_peak_gb": None,
            "validation_pass_at_1": evaluation.get("pass_at_1") if evaluation.get("split") == "validation" else None,
            "test_pass_at_1": evaluation.get("pass_at_1") if evaluation.get("split") == "test" else None,
            "status": "evaluated",
        })
    for selection in sorted(selections, key=lambda item: item["selection_id"]):
        runs = training_by_selection.get(selection["selection_id"], []) or [None]
        for training in runs:
            matched_evals = []
            if training:
                marker = f"/{training['train_id']}/adapter".replace("/", "").lower()
                matched_evals = [
                    item for item in evaluations
                    if marker in str(item.get("checkpoint", "")).replace("\\", "").replace("/", "").lower()
                ]
            def score(split: str) -> float | None:
                candidates = [item for item in matched_evals if item.get("split") == split]
                return candidates[-1].get("pass_at_1") if candidates else None
            row = {
                "condition_id": selection["selection_id"],
                "branch": selection["branch"],
                "selector": selection["method"],
                "fraction": selection["fraction"],
                "seed": training.get("train_id", "").rsplit("-s", 1)[-1] if training else selection["seed"],
                "pool_hash": selection["pool_hash"],
                "n_selected": selection["n_selected"],
                "n_unique_problems": selection["n_unique_problems"],
                "selected_completion_tokens": selection["selected_completion_tokens"],
                "train_id": training.get("train_id") if training else None,
                "train_tokens_seen_estimate": training.get("train_tokens_seen_estimate") if training else None,
                "global_steps": training.get("global_steps") if training else None,
                "train_seconds": training.get("train_seconds") if training else None,
                "gpu_peak_gb": training.get("gpu_peak_gb") if training else None,
                "validation_pass_at_1": score("validation"),
                "test_pass_at_1": score("test"),
                "status": "evaluated" if matched_evals else "trained" if training else "selected_only",
            }
            rows.append(row)
    return rows


def write_report(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = collect_results(config)
    output = experiment_root(config) / "star" / "reports"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# STaR experiment results",
        "",
        "`correct` denotes final-answer equivalence only, not verified process correctness.",
        "",
        "| condition | branch | selector | fraction | selected | status | validation Pass@1 | test Pass@1 |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition_id']} | {row['branch']} | {row['selector']} | "
            f"{row['fraction']} | {row['n_selected']} | {row['status']} | "
            f"{row['validation_pass_at_1'] if row['validation_pass_at_1'] is not None else ''} | "
            f"{row['test_pass_at_1'] if row['test_pass_at_1'] is not None else ''} |"
        )
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "n_conditions": len(rows),
        "n_trained": sum(row["status"] in {"trained", "evaluated"} for row in rows),
        "n_evaluated": sum(row["status"] == "evaluated" for row in rows),
        "comparison_warning": "Only compare selectors with the same pool_hash and branch.",
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(canonical_json(write_report(load_config(args.config))))


if __name__ == "__main__":
    main()
