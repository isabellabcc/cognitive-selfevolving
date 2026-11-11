"""STaR gold-answer rationalization for problems ordinary sampling did not solve."""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path
from typing import Any, Mapping

from .common import (
    PipelineError,
    append_jsonl,
    canonical_json,
    ensure_directories,
    experiment_root,
    load_config,
    read_jsonl,
    sha256_text,
    stable_seed,
    utc_now,
    write_json_atomic,
    write_jsonl_atomic,
)
from .generate import MockBackend, TransformersBackend, _repair_incomplete_last_line
from .parse_output import parse_completion
from .prompts import PROMPT_VERSION, build_prompt, load_few_shot
from .verify import VERIFIER_CONFIG, load_math_verifier, verify_record


RATIONALIZATION_PROMPT_VERSION = "star_gold_answer_v1"
def failed_problem_ids(
    raw_rows: list[Mapping[str, Any]], verified_rows: list[Mapping[str, Any]]
) -> set[str]:
    raw_by_id = {str(row["trajectory_id"]): row for row in raw_rows}
    if len(raw_by_id) != len(raw_rows):
        raise PipelineError("Duplicate ordinary trajectory IDs")
    verified_by_id = {str(row["trajectory_id"]): row for row in verified_rows}
    if set(raw_by_id) != set(verified_by_id):
        raise PipelineError("Ordinary raw and verification IDs are not one-to-one")
    attempted = {str(row["problem_id"]) for row in raw_rows}
    solved = {
        str(row["problem_id"])
        for row in verified_rows
        if row.get("verification_status") == "correct"
    }
    return attempted - solved


def _existing_ids(path: Path, expected_hash: str) -> set[str]:
    if not path.exists():
        return set()
    rows = list(read_jsonl(path))
    ids = [str(row.get("trajectory_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise PipelineError(f"Duplicate rationalization trajectory IDs in {path}")
    if any(row.get("generation_config_sha256") != expected_hash for row in rows):
        raise PipelineError("Existing rationalizations use different generation settings")
    return set(ids)


def run_rationalization(
    config: Mapping[str, Any], *, resume: bool, backend_name: str,
    mock_fixture: str | None = None, limit: int | None = None,
) -> dict[str, Any]:
    ensure_directories(config)
    root = experiment_root(config)
    questions = {row["problem_id"]: row for row in read_jsonl(root / "splits" / "seed_train.jsonl")}
    few_shot = load_few_shot(root / "prompts" / "few_shot.jsonl")
    gold = {row["problem_id"]: row for row in read_jsonl(root / "private" / "gold_answers.jsonl")}
    ordinary_raw = list(read_jsonl(root / "trajectories" / "seed_train_raw.jsonl"))
    ordinary_verified = list(read_jsonl(root / "verification" / "seed_train_verified.jsonl"))
    failed = sorted(failed_problem_ids(ordinary_raw, ordinary_verified))
    if limit is not None:
        if limit <= 0:
            raise PipelineError("--limit must be positive")
        failed = failed[:limit]
    missing = set(failed) - (set(questions) & set(gold))
    if missing:
        raise PipelineError(f"Missing question/gold for rationalization: {sorted(missing)[0]}")

    backend = MockBackend(mock_fixture) if backend_name == "mock" else TransformersBackend(config)
    rat_cfg = config["star"]["rationalization"]
    generation_settings = {
        "prompt_version": RATIONALIZATION_PROMPT_VERSION,
        "model_id": config["model"]["id"],
        "model_revision": backend.model_revision,
        "sampling": dict(config["sampling"]),
        "n_per_problem": int(rat_cfg["n_per_problem"]),
        "max_new_tokens": int(rat_cfg["max_new_tokens"]),
        "round": int(config["star"].get("round", 0)),
    }
    settings_hash = sha256_text(canonical_json(generation_settings))
    output = root / "trajectories" / "seed_train_rationalized_raw.jsonl"
    repair_events: list[dict[str, Any]] = []
    _repair_incomplete_last_line(output, repair_events.append)
    existing = _existing_ids(output, settings_hash)
    if existing and not resume:
        raise PipelineError(f"Rationalization output exists; pass --resume: {output}")
    events_path = Path(config["paths"]["run_root"]) / "rationalization_events.jsonl"
    summary = {"failed_problems": len(failed), "expected": 0, "success": 0, "skipped": 0, "failed": 0}
    n_per_problem = int(rat_cfg["n_per_problem"])
    summary["expected"] = len(failed) * n_per_problem
    with append_jsonl(output) as append_output, append_jsonl(events_path) as append_event:
        for repair_event in repair_events:
            append_event(repair_event)
        for problem_id in failed:
            for sample_index in range(n_per_problem):
                round_index = int(config["star"].get("round", 0))
                trajectory_id = f"{config['run_id']}:{problem_id}:r{round_index}:rat{sample_index:02d}"
                if trajectory_id in existing:
                    summary["skipped"] += 1
                    continue
                answer = str(gold[problem_id]["gold_answer"])
                prompt = build_prompt(
                    str(questions[problem_id]["question"]), few_shot, gold_hint=answer
                )
                seed = stable_seed(config["master_seed"], "rationalize", round_index, problem_id, sample_index)
                if isinstance(backend, MockBackend):
                    backend.set_task(problem_id, sample_index)
                started = time.perf_counter()
                try:
                    generated = backend.generate(prompt, seed, int(rat_cfg["max_new_tokens"]))
                    parsed = parse_completion(str(generated["raw_completion"]))
                    row = {
                        "trajectory_id": trajectory_id,
                        "run_id": config["run_id"],
                        "problem_id": problem_id,
                        "subset": "seed_train_rationalized",
                        "sample_index": sample_index,
                        "round": round_index,
                        "model_id": config["model"]["id"],
                        "model_revision": backend.model_revision,
                        "prompt_version": PROMPT_VERSION,
                        "generation_prompt_version": RATIONALIZATION_PROMPT_VERSION,
                        "rationalization_prompt_sha256": sha256_text(prompt),
                        "sampling_seed": seed,
                        "generation_config_sha256": settings_hash,
                        **parsed.as_dict(),
                        "finish_reason": generated["finish_reason"],
                        "generation_status": "success",
                        "prompt_token_count": generated.get("prompt_token_count"),
                        "completion_token_count": generated.get("completion_token_count"),
                        "completion_sha256": sha256_text(parsed.raw_completion),
                        "origin": "rationalization",
                        "used_gold_hint": True,
                        "created_at": utc_now(),
                    }
                    append_output(row)
                    append_event({
                        "event": "rationalization_success", "trajectory_id": trajectory_id,
                        "elapsed_seconds": time.perf_counter() - started, "created_at": utc_now(),
                    })
                    summary["success"] += 1
                except Exception as exc:
                    append_event({
                        "event": "rationalization_failure", "trajectory_id": trajectory_id,
                        "error": f"{type(exc).__name__}: {exc}", "created_at": utc_now(),
                    })
                    summary["failed"] += 1
    write_json_atomic(root / "reports" / "rationalization_generation_report.json", {
        **summary, "generation_config_sha256": settings_hash,
        "gold_hint_warning": "Gold answers were present in generation prompts; compare as a separate pool.",
    })
    return summary


def verify_rationalizations(config: Mapping[str, Any]) -> dict[str, Any]:
    root = experiment_root(config)
    raw_rows = list(read_jsonl(root / "trajectories" / "seed_train_rationalized_raw.jsonl"))
    gold = {row["problem_id"]: row for row in read_jsonl(root / "private" / "gold_answers.jsonl")}
    parse_fn, verify_fn, version = load_math_verifier()
    verifier_hash = sha256_text(canonical_json({**VERIFIER_CONFIG, "math_verify_version": version}))
    results = []
    for raw in raw_rows:
        if not str(raw.get("reasoning_text") or "").strip():
            result = {
                "trajectory_id": raw["trajectory_id"], "problem_id": raw["problem_id"],
                "subset": raw["subset"], "verification_status": "unverifiable",
                "is_correct": None, "verification_error": "empty_rationalization_reasoning",
                "verifier_config_sha256": verifier_hash,
            }
        else:
            result = verify_record(raw, gold[raw["problem_id"]], parse_fn, verify_fn, verifier_hash)
        results.append(result)
    output = root / "verification" / "seed_train_rationalized_verified.jsonl"
    write_jsonl_atomic(output, results)
    counts = collections.Counter(row["verification_status"] for row in results)
    report = {"total": len(results), "counts": dict(sorted(counts.items()))}
    write_json_atomic(root / "reports" / "rationalization_verification_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--backend", choices=("transformers", "mock"), default="transformers")
    generate.add_argument("--mock-fixture")
    generate.add_argument("--limit", type=int)
    verify = sub.add_parser("verify")
    verify.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "generate":
        result = run_rationalization(
            config, resume=args.resume, backend_name=args.backend,
            mock_fixture=args.mock_fixture, limit=args.limit,
        )
    else:
        result = verify_rationalizations(config)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
