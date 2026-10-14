"""Deduplicate correct trajectories, select eligible questions, and freeze a pool."""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import (
    PipelineError,
    canonical_json,
    experiment_root,
    load_config,
    normalize_light,
    read_jsonl,
    sha256_file,
    sha256_text,
    utc_now,
    write_json_atomic,
    write_jsonl_atomic,
)


def token_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = re.findall(r"\\[A-Za-z]+|[A-Za-z0-9_]+|[^\s]", normalize_light(text).lower())
    if not tokens:
        return set()
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def is_candidate(raw: Mapping[str, Any], verified: Mapping[str, Any]) -> bool:
    return (
        raw.get("generation_status") == "success"
        and raw.get("format_status") == "ok"
        and raw.get("finish_reason") == "eos"
        and verified.get("verification_status") == "correct"
        and bool(str(raw.get("reasoning_text") or "").strip())
        and bool(str(raw.get("final_text") or "").strip())
        and int(raw.get("completion_token_count") or 0) > 0
    )


def deduplicate_problem(
    rows: Iterable[Mapping[str, Any]], *, ngram_size: int, threshold: float
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return deterministic representatives and a reversible duplicate audit."""
    ordered = sorted(rows, key=lambda row: str(row["trajectory_id"]))
    if not ordered:
        return [], []
    text_by_id = {
        str(row["trajectory_id"]): normalize_light(
            f"{row.get('reasoning_text', '')}\n{row.get('final_text', '')}"
        )
        for row in ordered
    }
    exact_groups: dict[str, list[str]] = collections.defaultdict(list)
    for trajectory_id, text in text_by_id.items():
        exact_groups[sha256_text(text)].append(trajectory_id)
    exact_representatives: list[str] = []
    audit: list[dict[str, Any]] = []
    for digest in sorted(exact_groups):
        members = sorted(exact_groups[digest])
        representative = members[0]
        exact_representatives.append(representative)
        if len(members) > 1:
            audit.append({
                "problem_id": ordered[0]["problem_id"],
                "duplicate_type": "exact",
                "representative_trajectory_id": representative,
                "member_trajectory_ids": members,
                "removed_candidate_ids": members[1:],
                "normalized_text_sha256": digest,
                "similarities": {member: 1.0 for member in members},
            })

    exact_representatives.sort()
    grams = {item: token_ngrams(text_by_id[item], ngram_size) for item in exact_representatives}
    parent = {item: item for item in exact_representatives}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            keep, merge = sorted((left_root, right_root))
            parent[merge] = keep

    pair_scores: dict[tuple[str, str], float] = {}
    for index, left in enumerate(exact_representatives):
        for right in exact_representatives[index + 1 :]:
            score = jaccard(grams[left], grams[right])
            if score >= threshold:
                pair_scores[(left, right)] = score
                union(left, right)
    components: dict[str, list[str]] = collections.defaultdict(list)
    for item in exact_representatives:
        components[find(item)].append(item)
    representatives: list[str] = []
    for members in sorted((sorted(value) for value in components.values()), key=lambda value: value[0]):
        representative = members[0]
        representatives.append(representative)
        if len(members) > 1:
            similarities = {
                f"{left}|{right}": round(score, 8)
                for (left, right), score in sorted(pair_scores.items())
                if left in members and right in members
            }
            audit.append({
                "problem_id": ordered[0]["problem_id"],
                "duplicate_type": "near",
                "representative_trajectory_id": representative,
                "member_trajectory_ids": members,
                "removed_candidate_ids": members[1:],
                "threshold": threshold,
                "ngram_size": ngram_size,
                "pair_similarities": similarities,
                "note": "Textual near-duplicate grouping is not a judgment of reasoning-strategy identity.",
            })
    return representatives, audit


def stratified_select(rows: list[dict[str, Any]], total: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[(int(row["level"]), str(row["topic"]))].append(row)
    population = len(rows)
    raw_quota = {key: total * len(value) / population for key, value in groups.items()}
    quota = {key: min(len(groups[key]), int(value)) for key, value in raw_quota.items()}
    remaining = total - sum(quota.values())
    order = sorted(groups, key=lambda key: (-(raw_quota[key] - int(raw_quota[key])), key))
    while remaining:
        progressed = False
        for key in order:
            if quota[key] < len(groups[key]):
                quota[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise PipelineError("Unable to fill the requested stratified pool size")
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda row: row["problem_id"])
        stratum_seed = seed + int(sha256_text(f"{key[0]}|{key[1]}")[:8], 16)
        random.Random(stratum_seed).shuffle(candidates)
        selected.extend(candidates[: quota[key]])
    return sorted(selected, key=lambda row: row["problem_id"])


def _merge(raw: Mapping[str, Any], verified: Mapping[str, Any]) -> dict[str, Any]:
    overlap = set(raw) & set(verified) - {"trajectory_id", "problem_id", "subset"}
    if overlap:
        raise PipelineError(f"Unexpected raw/verification field collision: {sorted(overlap)}")
    return {**raw, **verified}


def build_pool(config: Mapping[str, Any], *, freeze: bool) -> dict[str, Any]:
    root = experiment_root(config)
    raw_path = root / "trajectories" / "seed_train_raw.jsonl"
    verified_path = root / "verification" / "seed_train_verified.jsonl"
    split_path = root / "splits" / "seed_train.jsonl"
    for path in (raw_path, verified_path, split_path):
        if not path.is_file():
            raise PipelineError(f"Required pool input does not exist: {path}")
    run_manifest_path = Path(config["paths"]["run_root"]) / "run_manifest.json"
    if freeze and not run_manifest_path.exists():
        raise PipelineError("Cannot freeze without run_manifest.json from the generation stage")
    if freeze and run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("config_sha256") != config["_meta"]["config_sha256"]:
            raise PipelineError("Run manifest config hash does not match the active config")
        if run_manifest.get("subsets", {}).get("seed_train", {}).get("limited"):
            raise PipelineError("A --limit generation run cannot be frozen as a formal pool")

    manifest_path = root / "pool" / "pool_manifest.json"
    input_hashes = {
        "seed_train_split": sha256_file(split_path),
        "seed_train_raw": sha256_file(raw_path),
        "seed_train_verified": sha256_file(verified_path),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("frozen"):
            if existing.get("input_sha256") == input_hashes:
                return existing
            raise PipelineError("Frozen pool inputs changed; create a new pool/run version")

    questions = {row["problem_id"]: row for row in read_jsonl(split_path)}
    raw_rows = list(read_jsonl(raw_path))
    verified_rows = list(read_jsonl(verified_path))
    raw_by_id = {row["trajectory_id"]: row for row in raw_rows}
    verified_by_id = {row["trajectory_id"]: row for row in verified_rows}
    if len(raw_by_id) != len(raw_rows) or len(verified_by_id) != len(verified_rows):
        raise PipelineError("Duplicate trajectory_id in pool inputs")
    if set(raw_by_id) != set(verified_by_id):
        missing_verification = set(raw_by_id) - set(verified_by_id)
        orphan_verification = set(verified_by_id) - set(raw_by_id)
        raise PipelineError(
            f"Raw/verification ID mismatch: {len(missing_verification)} missing, "
            f"{len(orphan_verification)} orphan"
        )
    if freeze:
        expected_ids = {
            f"{config['run_id']}:{problem_id}:s{sample_index:02d}"
            for problem_id in questions
            for sample_index in range(int(config["sampling"]["n_per_problem"]))
        }
        raw_ids = set(raw_by_id)
        extra_ids = raw_ids - expected_ids
        events_path = Path(config["paths"]["run_root"]) / "generation_events.jsonl"
        terminal_failure_ids = set()
        if events_path.exists():
            terminal_failure_ids = {
                row["trajectory_id"] for row in read_jsonl(events_path)
                if row.get("event") == "generation_terminal_failure" and row.get("trajectory_id") in expected_ids
            }
        missing_terminal_state = expected_ids - raw_ids - terminal_failure_ids
        if extra_ids or missing_terminal_state:
            raise PipelineError(
                f"Generation task coverage is incomplete: {len(missing_terminal_state)} without a terminal state, "
                f"{len(extra_ids)} unexpected trajectory IDs"
            )
    unknown_problems = {row["problem_id"] for row in raw_rows} - set(questions)
    if unknown_problems:
        raise PipelineError(f"Raw trajectories include non-seed_train problem: {sorted(unknown_problems)[0]}")

    by_problem: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for raw in raw_rows:
        by_problem[raw["problem_id"]].append(raw)
    pool_cfg = config["pool"]
    min_distinct = int(pool_cfg["min_distinct_correct"])
    representatives_by_problem: dict[str, list[str]] = {}
    audit: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    all_question_summaries: list[dict[str, Any]] = []
    for problem_id in sorted(questions):
        problem_raw = by_problem.get(problem_id, [])
        candidate_rows = [
            row for row in problem_raw if is_candidate(row, verified_by_id[row["trajectory_id"]])
        ]
        representatives, problem_audit = deduplicate_problem(
            candidate_rows,
            ngram_size=int(pool_cfg["dedup_token_ngram"]),
            threshold=float(pool_cfg["near_duplicate_jaccard_threshold"]),
        )
        representatives_by_problem[problem_id] = representatives
        audit.extend(problem_audit)
        status_counts = collections.Counter(
            verified_by_id[row["trajectory_id"]]["verification_status"] for row in problem_raw
        )
        question = questions[problem_id]
        summary = {
            "problem_id": problem_id,
            "level": question["level"],
            "topic": question["topic"],
            "n_generated": len(problem_raw),
            "n_correct": status_counts["correct"],
            "n_distinct_correct": len(representatives),
            "n_incorrect": status_counts["incorrect"],
            "n_unverifiable": status_counts["unverifiable"],
            "eligible": len(representatives) >= min_distinct,
            "distinct_correct_trajectory_ids": representatives,
        }
        all_question_summaries.append(summary)
        if summary["eligible"]:
            summaries.append(summary)

    pool_dir = root / "pool"
    write_jsonl_atomic(pool_dir / "eligible_questions.jsonl", summaries)
    write_jsonl_atomic(pool_dir / "dedup_audit.jsonl", audit)
    target = int(pool_cfg["target_n_problems"])
    base_report = {
        "final_answer_correctness_only": True,
        "process_correctness_is_not_assessed": True,
        "eligible_problem_count": len(summaries),
        "target_problem_count": target,
        "eligibility_gap": max(0, target - len(summaries)),
        "all_seed_problem_count": len(questions),
        "eligibility_counts": dict(collections.Counter(str(row["eligible"]).lower() for row in all_question_summaries)),
        "eligible_by_level": dict(sorted(collections.Counter(str(row["level"]) for row in summaries).items())),
        "eligible_by_topic": dict(sorted(collections.Counter(row["topic"] for row in summaries).items())),
    }
    if len(summaries) < target:
        write_json_atomic(root / "reports" / "pool_report.json", base_report)
        raise PipelineError(
            f"Only {len(summaries)} eligible questions; target is {target}. "
            "No ineligible questions were substituted."
        )
    selected = stratified_select(summaries, target, int(pool_cfg["pool_seed"]))
    selection_rows = [
        {
            "problem_id": row["problem_id"], "level": row["level"], "topic": row["topic"],
            "selection_seed": int(pool_cfg["pool_seed"]),
            "selection_rule": "proportional_level_topic_stratified_random",
        }
        for row in selected
    ]
    selected_ids = {row["problem_id"] for row in selected}
    correct_ids = {
        trajectory_id
        for problem_id in selected_ids
        for trajectory_id in representatives_by_problem[problem_id]
    }
    candidates_correct = [
        _merge(raw_by_id[trajectory_id], verified_by_id[trajectory_id])
        for trajectory_id in sorted(correct_ids)
    ]
    representative_for: dict[str, str] = {}
    for item in audit:
        representative = item["representative_trajectory_id"]
        for member in item["member_trajectory_ids"]:
            representative_for[member] = representative
    for member in list(representative_for):
        target = representative_for[member]
        seen_targets = {member}
        while target in representative_for and representative_for[target] != target:
            if target in seen_targets:
                raise PipelineError(f"Dedup representative cycle involving {member}")
            seen_targets.add(target)
            target = representative_for[target]
        representative_for[member] = target
    candidates_all = []
    for row in sorted(raw_rows, key=lambda item: item["trajectory_id"]):
        if row["problem_id"] not in selected_ids or row.get("generation_status") != "success":
            continue
        trajectory_id = row["trajectory_id"]
        merged = _merge(row, verified_by_id[trajectory_id])
        if trajectory_id in correct_ids:
            merged["pool_candidate_status"] = "distinct_correct_representative"
            merged["dedup_representative_trajectory_id"] = trajectory_id
        elif trajectory_id in representative_for:
            merged["pool_candidate_status"] = "correct_duplicate"
            merged["dedup_representative_trajectory_id"] = representative_for[trajectory_id]
        elif verified_by_id[trajectory_id]["verification_status"] == "correct":
            merged["pool_candidate_status"] = "correct_but_ineligible_format"
            merged["dedup_representative_trajectory_id"] = None
        else:
            merged["pool_candidate_status"] = verified_by_id[trajectory_id]["verification_status"]
            merged["dedup_representative_trajectory_id"] = None
        candidates_all.append(merged)
    write_jsonl_atomic(pool_dir / "selected_questions.jsonl", selection_rows)
    write_jsonl_atomic(pool_dir / "candidates_correct.jsonl", candidates_correct)
    write_jsonl_atomic(pool_dir / "candidates_all.jsonl", candidates_all)
    report = {
        **base_report,
        "selected_problem_count": len(selected),
        "correct_before_dedup": sum(row["n_correct"] for row in selected),
        "correct_after_dedup": len(candidates_correct),
        "mean_candidates_per_problem": len(candidates_correct) / len(selected),
        "exact_duplicate_groups": sum(row["duplicate_type"] == "exact" for row in audit),
        "near_duplicate_groups": sum(row["duplicate_type"] == "near" for row in audit),
    }
    write_json_atomic(root / "reports" / "pool_report.json", report)
    output_paths = {
        name: pool_dir / name
        for name in (
            "eligible_questions.jsonl", "selected_questions.jsonl", "candidates_correct.jsonl",
            "candidates_all.jsonl", "dedup_audit.jsonl",
        )
    }
    manifest = {
        "pool_id": f"{config['run_id']}:pool_v1",
        "run_id": config["run_id"],
        "created_at": utc_now(),
        "frozen": bool(freeze),
        "config_sha256": config["_meta"]["config_sha256"],
        "input_sha256": input_hashes,
        "output_sha256": {name: sha256_file(path) for name, path in output_paths.items()},
        "rules": {
            "min_distinct_correct": min_distinct,
            "target_n_problems": target,
            "near_duplicate_jaccard_threshold": float(pool_cfg["near_duplicate_jaccard_threshold"]),
            "dedup_token_ngram": int(pool_cfg["dedup_token_ngram"]),
            "pool_seed": int(pool_cfg["pool_seed"]),
        },
        "statistics": report,
    }
    write_json_atomic(manifest_path, manifest)
    if freeze:
        for path in (*output_paths.values(), manifest_path):
            try:
                path.chmod(0o444)
            except OSError:
                pass
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--freeze", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_pool(load_config(args.config), freeze=args.freeze)
    print(canonical_json(result["statistics"]))


if __name__ == "__main__":
    main()
