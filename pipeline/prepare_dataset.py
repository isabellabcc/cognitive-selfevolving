"""Download, normalize, stratify, and split the MATH train dataset."""

from __future__ import annotations

import argparse
import collections
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import (
    PipelineError,
    ensure_directories,
    experiment_root,
    load_config,
    normalize_problem,
    sha256_text,
    sha256_file,
    utc_now,
    write_json_atomic,
    write_jsonl_atomic,
)


PUBLIC_FIELDS = {
    "problem_id", "source_repo", "source_config", "source_split",
    "source_row_index", "level", "topic", "problem", "problem_sha256", "subset",
}
ASYMPTOTE_MARKERS = (r"\begin{asy}", "[asy]", "import graph", "import olympiad")


def parse_level(value: Any) -> int:
    if isinstance(value, int) and 1 <= value <= 5:
        return value
    match = re.fullmatch(r"\s*Level\s+([1-5])\s*", str(value), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized MATH level: {value!r}")
    return int(match.group(1))


def extract_balanced_boxed(text: str) -> list[str]:
    """Return all structurally complete ``\\boxed{...}`` contents."""
    answers: list[str] = []
    cursor = 0
    marker = r"\boxed"
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            break
        brace = start + len(marker)
        while brace < len(text) and text[brace].isspace():
            brace += 1
        if brace >= len(text) or text[brace] != "{":
            cursor = brace
            continue
        depth = 0
        escaped = False
        end = None
        for index in range(brace, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end is None:
            cursor = brace + 1
            continue
        answers.append(text[brace + 1 : end].strip())
        cursor = end + 1
    return answers


def extract_gold(solution: str) -> tuple[str | None, str, str]:
    boxes = extract_balanced_boxed(solution)
    if not boxes:
        return None, "missing", "last_boxed_balanced"
    distinct = list(dict.fromkeys(boxes))
    if len(distinct) > 1:
        return boxes[-1], "ambiguous", "last_boxed_balanced"
    if not boxes[-1]:
        return None, "parse_error", "last_boxed_balanced"
    return boxes[-1], "ok", "last_boxed_balanced"


def _largest_remainder(total: int, weights: Mapping[Any, float]) -> dict[Any, int]:
    if not weights or sum(weights.values()) <= 0:
        return {}
    denominator = float(sum(weights.values()))
    raw = {key: total * float(weight) / denominator for key, weight in weights.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(weights, key=lambda key: (-(raw[key] - result[key]), str(key)))
    for key in order[:remaining]:
        result[key] += 1
    return result


def stratified_sample(
    records: list[dict[str, Any]], total: int, level_ratios: Mapping[int, float], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select exact total with level targets and proportional topic strata.

    Level shortages are reassigned deterministically to the closest level with
    capacity. Within each level, topic shares follow the available population.
    """
    by_level: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        by_level[int(record["level"])].append(record)
    if len(records) < total:
        raise PipelineError(f"Only {len(records)} usable questions are available; {total} required")

    requested = _largest_remainder(total, level_ratios)
    allocated = {level: min(requested.get(level, 0), len(by_level[level])) for level in range(1, 6)}
    shortage = total - sum(allocated.values())
    while shortage:
        deficient = [level for level in range(1, 6) if requested.get(level, 0) > allocated[level]]
        origin = deficient[0] if deficient else 3
        choices = [
            level for level in range(1, 6)
            if allocated[level] < len(by_level[level])
        ]
        if not choices:
            raise PipelineError("Unable to redistribute stratified-sampling shortage")
        target = min(choices, key=lambda level: (abs(level - origin), level))
        allocated[target] += 1
        shortage -= 1

    selected: list[dict[str, Any]] = []
    topic_allocations: dict[str, dict[str, int]] = {}
    for level in range(1, 6):
        groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for record in by_level[level]:
            groups[str(record["topic"])].append(record)
        topic_target = _largest_remainder(allocated[level], {k: len(v) for k, v in groups.items()})
        # Proportional quotas cannot exceed their source stratum. The loop also
        # covers any rounding/capacity corner case deterministically.
        topic_target = {k: min(v, len(groups[k])) for k, v in topic_target.items()}
        remain = allocated[level] - sum(topic_target.values())
        for topic in sorted(groups):
            take = min(remain, len(groups[topic]) - topic_target.get(topic, 0))
            topic_target[topic] = topic_target.get(topic, 0) + take
            remain -= take
            if not remain:
                break
        topic_allocations[str(level)] = topic_target
        for topic in sorted(groups):
            candidates = sorted(groups[topic], key=lambda row: row["problem_id"])
            random.Random(seed + level * 1009 + int(sha256_text(topic)[:8], 16)).shuffle(candidates)
            selected.extend(candidates[: topic_target.get(topic, 0)])

    selected.sort(key=lambda row: row["problem_id"])
    return selected, {
        "requested_level_counts": requested,
        "actual_level_counts": allocated,
        "topic_allocations": topic_allocations,
        "redistributed": requested != allocated,
    }


def split_selected(
    selected: list[dict[str, Any]], subset_sizes: Mapping[str, int], seed: int
) -> dict[str, list[dict[str, Any]]]:
    shuffled = sorted(selected, key=lambda row: row["problem_id"])
    random.Random(seed + 1).shuffle(shuffled)
    result: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for subset in ("seed_train", "anchor", "dev"):
        size = int(subset_sizes[subset])
        rows = []
        for record in shuffled[offset : offset + size]:
            public = {key: record[key] for key in PUBLIC_FIELDS if key != "subset"}
            public["subset"] = subset
            rows.append(public)
        result[subset] = rows
        offset += size
    return result


def assert_split_contract(splits: Mapping[str, list[dict[str, Any]]], sizes: Mapping[str, int]) -> None:
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for name in ("seed_train", "anchor", "dev"):
        rows = splits[name]
        if len(rows) != int(sizes[name]):
            raise PipelineError(f"{name} has {len(rows)} records; expected {sizes[name]}")
        for row in rows:
            if set(row) != PUBLIC_FIELDS:
                raise PipelineError(f"Public split schema mismatch for {row.get('problem_id')}")
            if row["problem_id"] in seen_ids or row["problem_sha256"] in seen_hashes:
                raise PipelineError(f"Cross-split duplicate: {row['problem_id']}")
            seen_ids.add(row["problem_id"])
            seen_hashes.add(row["problem_sha256"])


def prepare(config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise PipelineError("datasets is required for data preparation") from exc

    ensure_directories(config)
    dataset_cfg = config["dataset"]
    audit: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    config_counts: dict[str, int] = {}
    source_metadata: dict[str, Any] = {}
    for source_config in dataset_cfg["configs"]:
        dataset = load_dataset(
            dataset_cfg["repo_id"], name=source_config, split=dataset_cfg["split"]
        )
        config_counts[source_config] = len(dataset)
        source_metadata[source_config] = {
            "fingerprint": getattr(dataset, "_fingerprint", None),
            "version": str(getattr(getattr(dataset, "info", None), "version", "unknown")),
        }
        for row_index, row in enumerate(dataset):
            raw_records.append({
                "source_repo": dataset_cfg["repo_id"],
                "source_config": source_config,
                "source_split": dataset_cfg["split"],
                "source_row_index": row_index,
                "problem": row.get("problem"),
                "solution": row.get("solution"),
                "level_raw": row.get("level"),
                "topic": row.get("type"),
            })

    usable: list[dict[str, Any]] = []
    gold_by_id: dict[str, dict[str, Any]] = {}
    seen_hashes: dict[str, str] = {}
    seen_near_keys: dict[str, str] = {}
    filter_counts: collections.Counter[str] = collections.Counter()
    for raw in raw_records:
        reason = None
        if not str(raw.get("problem") or "").strip():
            reason = "empty_problem"
        elif not str(raw.get("solution") or "").strip():
            reason = "empty_solution"
        elif not str(raw.get("topic") or "").strip():
            reason = "empty_topic"
        if reason:
            filter_counts[reason] += 1
            audit.append({"reason": reason, **raw})
            continue
        try:
            level = parse_level(raw["level_raw"])
        except ValueError:
            filter_counts["invalid_level"] += 1
            audit.append({"reason": "invalid_level", **raw})
            continue
        problem = normalize_problem(str(raw["problem"]))
        problem_hash = sha256_text(problem)
        if problem_hash in seen_hashes:
            filter_counts["duplicate_problem"] += 1
            audit.append({"reason": "duplicate_problem", "kept_problem_id": seen_hashes[problem_hash], **raw})
            continue
        # Preserve mathematical operators: dropping '<', '>' or '-' can turn
        # different questions into the same key. This catches presentation-only
        # differences in case, whitespace, and prose punctuation.
        near_key = unicodedata.normalize("NFKC", problem).casefold()
        near_key = re.sub(r"[\s.,;:!?\"'`]+", "", near_key, flags=re.UNICODE)
        if near_key in seen_near_keys:
            filter_counts["near_duplicate_problem"] += 1
            audit.append({"reason": "near_duplicate_problem", "kept_problem_id": seen_near_keys[near_key], **raw})
            continue
        if dataset_cfg.get("exclude_asymptote") and any(marker in problem for marker in ASYMPTOTE_MARKERS):
            filter_counts["asymptote"] += 1
            audit.append({"reason": "asymptote", **raw})
            continue
        gold, gold_status, method = extract_gold(str(raw["solution"]))
        if gold_status != "ok":
            filter_counts[f"gold_{gold_status}"] += 1
            audit.append({"reason": f"gold_{gold_status}", **raw})
            continue
        problem_id = (
            f"math_{raw['source_config']}_{raw['source_split']}_"
            f"{int(raw['source_row_index']):06d}_{problem_hash[:8]}"
        )
        record = {
            "problem_id": problem_id,
            "source_repo": raw["source_repo"],
            "source_config": raw["source_config"],
            "source_split": raw["source_split"],
            "source_row_index": raw["source_row_index"],
            "level": level,
            "topic": str(raw["topic"]),
            "problem": problem,
            "problem_sha256": problem_hash,
        }
        usable.append(record)
        seen_hashes[problem_hash] = problem_id
        seen_near_keys[near_key] = problem_id
        gold_by_id[problem_id] = {
            "problem_id": problem_id,
            "gold_answer": gold,
            "gold_extraction_status": gold_status,
            "source_solution": raw["solution"],
            "gold_extraction_method": method,
        }

    sizes = dataset_cfg["subset_sizes"]
    total = sum(int(value) for value in sizes.values())
    ratios = {int(key): float(value) for key, value in dataset_cfg["target_level_ratios"].items()}
    selected, allocation = stratified_sample(usable, total, ratios, int(dataset_cfg["split_seed"]))
    splits = split_selected(selected, sizes, int(dataset_cfg["split_seed"]))
    assert_split_contract(splits, sizes)
    selected_ids = {row["problem_id"] for rows in splits.values() for row in rows}

    root = experiment_root(config)
    write_jsonl_atomic(root / "raw" / "math_train.jsonl", raw_records)
    for name, rows in splits.items():
        write_jsonl_atomic(root / "splits" / f"{name}.jsonl", rows)
    write_jsonl_atomic(root / "private" / "gold_answers.jsonl", (gold_by_id[key] for key in sorted(selected_ids)))
    write_jsonl_atomic(root / "audits" / "dataset_exclusions.jsonl", audit)
    split_manifest = {
        "created_at": utc_now(),
        "strategy": "level_target_then_proportional_topic_then_seeded_subset_shuffle",
        "split_seed": int(dataset_cfg["split_seed"]),
        "source_repo": dataset_cfg["repo_id"],
        "source_split": dataset_cfg["split"],
        "allocation": allocation,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "sha256": {
            name: sha256_file(root / "splits" / f"{name}.jsonl") for name in splits
        },
        "gold_sha256": sha256_file(root / "private" / "gold_answers.jsonl"),
    }
    write_json_atomic(root / "splits" / "split_manifest.json", split_manifest)
    report = {
        "created_at": utc_now(),
        "source_repo": dataset_cfg["repo_id"],
        "source_split": dataset_cfg["split"],
        "raw_total": len(raw_records),
        "config_counts": config_counts,
        "source_metadata": source_metadata,
        "filter_counts": dict(sorted(filter_counts.items())),
        "usable_total": len(usable),
        "gold_extraction_ok_count": len(usable),
        "gold_extraction_success_rate": len(usable) / (len(usable) + sum(
            count for reason, count in filter_counts.items() if reason.startswith("gold_")
        )) if usable or any(reason.startswith("gold_") for reason in filter_counts) else None,
        "selected_total": len(selected_ids),
        "allocation": allocation,
        "splits": {name: summarize_rows(rows) for name, rows in splits.items()},
        "gold_extraction_note": "Non-ok gold records are excluded and retained in the audit.",
        "evaluation_leakage_check": "not_run_no_evaluation_dataset_configured",
    }
    write_json_atomic(root / "reports" / "dataset_report.json", report)
    return report


def summarize_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "count": len(rows),
        "levels": dict(sorted(collections.Counter(str(row["level"]) for row in rows).items())),
        "topics": dict(sorted(collections.Counter(str(row["topic"]) for row in rows).items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.validate_only:
        print(f"Configuration is valid: {config['_meta']['config_sha256']}")
        return
    report = prepare(config)
    print(f"Prepared {report['selected_total']} mutually exclusive MATH train questions")


if __name__ == "__main__":
    main()
