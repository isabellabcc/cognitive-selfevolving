"""Build fixed STaR pools and reproducible same-pool selections."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Mapping

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


def _merge_verified(
    raw_path: Path, verified_path: Path, *, origin: str, used_gold_hint: bool
) -> list[dict[str, Any]]:
    raw_rows = list(read_jsonl(raw_path))
    verified_rows = list(read_jsonl(verified_path))
    raw_by_id = {str(row["trajectory_id"]): row for row in raw_rows}
    verified_by_id = {str(row["trajectory_id"]): row for row in verified_rows}
    if len(raw_by_id) != len(raw_rows) or len(verified_by_id) != len(verified_rows):
        raise PipelineError("Duplicate trajectory IDs in a STaR pool input")
    if set(raw_by_id) != set(verified_by_id):
        raise PipelineError("Raw and verification trajectory IDs are not one-to-one")
    pool = []
    for trajectory_id in sorted(raw_by_id):
        verification = verified_by_id[trajectory_id]
        if verification.get("verification_status") != "correct":
            continue
        raw = raw_by_id[trajectory_id]
        if not str(raw.get("reasoning_text") or "").strip():
            continue
        pool.append({
            **raw,
            **verification,
            "origin": origin,
            "used_gold_hint": used_gold_hint,
        })
    return pool


def pool_content_hash(rows: list[Mapping[str, Any]]) -> str:
    """Hash full, ID-sorted pool content so all selectors share an auditable parent."""
    ordered = sorted(rows, key=lambda row: str(row["trajectory_id"]))
    return sha256_text(canonical_json(ordered))


def build_star_pool(config: Mapping[str, Any], branch: str) -> dict[str, Any]:
    root = experiment_root(config)
    star_dir = root / "star" / "pools"
    ordinary_raw = root / "trajectories" / "seed_train_raw.jsonl"
    ordinary_verified = root / "verification" / "seed_train_verified.jsonl"
    ordinary = _merge_verified(
        ordinary_raw, ordinary_verified, origin="ordinary", used_gold_hint=False
    )
    rows = list(ordinary)
    inputs = [ordinary_raw, ordinary_verified]
    if branch == "with_rationalization":
        rat_raw = root / "trajectories" / "seed_train_rationalized_raw.jsonl"
        rat_verified = root / "verification" / "seed_train_rationalized_verified.jsonl"
        rows.extend(_merge_verified(
            rat_raw, rat_verified, origin="rationalization", used_gold_hint=True
        ))
        inputs.extend([rat_raw, rat_verified])
    elif branch != "ordinary":
        raise PipelineError(f"Unsupported STaR pool branch: {branch}")

    rows.sort(key=lambda row: str(row["trajectory_id"]))
    output = star_dir / f"{branch}_correct.jsonl"
    write_jsonl_atomic(output, rows)
    pool_hash = pool_content_hash(rows)
    manifest = {
        "branch": branch,
        "created_at": utc_now(),
        "pool_hash": pool_hash,
        "n_trajectories": len(rows),
        "n_unique_problems": len({row["problem_id"] for row in rows}),
        "origin_counts": {
            origin: sum(row["origin"] == origin for row in rows)
            for origin in ("ordinary", "rationalization")
            if any(row["origin"] == origin for row in rows)
        },
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "output_sha256": sha256_file(output),
        "correctness_scope": "final_answer_only",
    }
    write_json_atomic(star_dir / f"{branch}_manifest.json", manifest)
    return manifest


def _exact_representatives(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    representatives: dict[str, dict[str, Any]] = {}
    for row in sorted(pool, key=lambda item: str(item["trajectory_id"])):
        key = sha256_text(
            normalize_light(str(row.get("reasoning_text") or "")).casefold()
            + "\n"
            + normalize_light(str(row.get("final_text") or "")).casefold()
        )
        representatives.setdefault(key, row)
    return list(representatives.values())


def select_rows(
    pool: list[dict[str, Any]], method: str, fraction: float, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0 < fraction <= 1:
        raise PipelineError("fraction must be in (0, 1]")
    ordered = sorted(pool, key=lambda row: str(row["trajectory_id"]))
    requested_budget = math.ceil(fraction * len(ordered))
    if method == "all":
        if fraction != 1:
            raise PipelineError("all selection requires fraction=1")
        selected = ordered
    elif method == "random":
        selected = random.Random(seed).sample(ordered, requested_budget)
        selected.sort(key=lambda row: str(row["trajectory_id"]))
    elif method == "exact_dedup":
        representatives = _exact_representatives(ordered)
        selected = representatives[:requested_budget]
    else:
        raise PipelineError(f"Unsupported selector: {method}")
    detail = {
        "requested_budget": requested_budget,
        "actual_budget": len(selected),
        "shortfall": requested_budget - len(selected),
        "dedup_removed": len(ordered) - len(_exact_representatives(ordered)),
    }
    return selected, detail


def run_selection(
    config: Mapping[str, Any], branch: str, method: str, fraction: float, seed: int
) -> dict[str, Any]:
    root = experiment_root(config)
    pool_path = root / "star" / "pools" / f"{branch}_correct.jsonl"
    pool = list(read_jsonl(pool_path))
    pool_hash = pool_content_hash(pool)
    selected, detail = select_rows(pool, method, fraction, seed)
    selection_id = f"{branch}-{method}-f{fraction:.4f}-s{seed}-{pool_hash[:10]}"
    destination = root / "star" / "selections" / selection_id
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(destination / "selected.jsonl", selected)
    write_json_atomic(destination / "selected_ids.json", [row["trajectory_id"] for row in selected])
    lengths = [int(row.get("completion_token_count") or 0) for row in selected]
    manifest = {
        "selection_id": selection_id,
        "created_at": utc_now(),
        "branch": branch,
        "method": method,
        "fraction": fraction,
        "seed": seed,
        "pool_hash": pool_hash,
        "pool_size": len(pool),
        "n_selected": len(selected),
        "n_unique_problems": len({row["problem_id"] for row in selected}),
        "selected_completion_tokens": sum(lengths),
        "selection_sha256": sha256_file(destination / "selected.jsonl"),
        **detail,
    }
    write_json_atomic(destination / "selection_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-pool")
    build.add_argument("--config", required=True)
    build.add_argument("--branch", choices=("ordinary", "with_rationalization"), default="ordinary")
    select = sub.add_parser("select")
    select.add_argument("--config", required=True)
    select.add_argument("--branch", choices=("ordinary", "with_rationalization"), default="ordinary")
    select.add_argument("--method", choices=("all", "random", "exact_dedup"), required=True)
    select.add_argument("--fraction", type=float, default=1.0)
    select.add_argument("--seed", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "build-pool":
        result = build_star_pool(config, args.branch)
    else:
        result = run_selection(config, args.branch, args.method, args.fraction, args.seed)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
