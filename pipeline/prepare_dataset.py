"""Prepare deterministic, leakage-audited GSM8K splits for one-round STaR."""

from __future__ import annotations

import argparse
import random
import re
from typing import Any, Mapping

from .common import (
    PipelineError, canonical_json, ensure_directories, experiment_root, git_commit, load_config,
    normalize_problem, sha256_file, sha256_text, utc_now, write_json_atomic,
    write_jsonl_atomic,
)


GOLD_PATTERN = re.compile(r"####\s*([^\n]+)\s*$")
PUBLIC_FIELDS = {
    "problem_id", "source_repo", "source_config", "source_split",
    "source_row_index", "question", "question_sha256",
}


def extract_gold(answer: str) -> str:
    match = GOLD_PATTERN.search(answer)
    if not match:
        raise PipelineError("GSM8K answer is missing its final #### marker")
    value = match.group(1).strip().replace(",", "")
    if not value:
        raise PipelineError("GSM8K final answer is empty")
    return value


def _load_dataset(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str | None]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise PipelineError("datasets is required to prepare GSM8K") from exc
    dataset_cfg = config["dataset"]
    loaded = load_dataset(
        dataset_cfg["repo_id"], dataset_cfg["config_name"],
        revision=dataset_cfg.get("revision", "main"),
    )
    train = [dict(row) for row in loaded["train"]]
    test = [dict(row) for row in loaded["test"]]
    version = getattr(getattr(loaded["train"], "info", None), "version", None)
    fingerprint = getattr(loaded["train"], "_fingerprint", None)
    return train, test, {
        "version": str(version) if version is not None else None,
        "train_fingerprint": str(fingerprint) if fingerprint is not None else None,
    }


def _records(rows: list[dict[str, Any]], split: str, repo_id: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    public = []
    private = {}
    seen_questions: set[str] = set()
    for index, row in enumerate(rows):
        question = normalize_problem(str(row["question"]))
        question_hash = sha256_text(question)
        if question_hash in seen_questions:
            raise PipelineError(f"Duplicate GSM8K question in {split}: row {index}")
        seen_questions.add(question_hash)
        problem_id = f"gsm8k-{split}-{index:05d}-{question_hash[:8]}"
        public_row = {
            "problem_id": problem_id, "source_repo": repo_id,
            "source_config": "main", "source_split": split,
            "source_row_index": index, "question": question,
            "question_sha256": question_hash,
        }
        public.append(public_row)
        private[problem_id] = {
            "problem_id": problem_id, "gold_answer": extract_gold(str(row["answer"])),
            "gold_extraction_status": "ok", "source_answer": str(row["answer"]),
        }
    return public, private


def prepare(config: Mapping[str, Any]) -> dict[str, Any]:
    ensure_directories(config)
    dataset_cfg = config["dataset"]
    train_raw, test_raw, resolved_dataset = _load_dataset(config)
    train, train_gold = _records(train_raw, "train", dataset_cfg["repo_id"])
    test, test_gold = _records(test_raw, "test", dataset_cfg["repo_id"])
    overlap = {row["question_sha256"] for row in train} & {row["question_sha256"] for row in test}
    if overlap:
        raise PipelineError(f"GSM8K train/test text overlap detected: {len(overlap)}")

    few_indices = {int(index) for index in dataset_cfg["few_shot_indices"]}
    few_shot = sorted(
        (row for row in train if int(row["source_row_index"]) in few_indices),
        key=lambda row: int(row["source_row_index"]),
    )
    if len(few_shot) != len(few_indices):
        raise PipelineError("A configured GSM8K few-shot source index is unavailable")
    ordered = sorted(
        (row for row in train if int(row["source_row_index"]) not in few_indices),
        key=lambda row: row["problem_id"],
    )
    random.Random(int(dataset_cfg["split_seed"])).shuffle(ordered)
    val_count = int(dataset_cfg["validation_size"])
    generation_count = int(dataset_cfg["generation_size"])
    validation = ordered[:val_count]
    seed_train = ordered[val_count:val_count + generation_count]
    reserve = ordered[val_count + generation_count:]

    root = experiment_root(config)
    splits = {"seed_train": seed_train, "validation": validation, "reserve": reserve, "test": test}
    for name, rows in splits.items():
        write_jsonl_atomic(root / "splits" / f"{name}.jsonl", rows)
    prompt_examples = [{
        "problem_id": row["problem_id"], "question": row["question"],
        "answer": train_gold[row["problem_id"]]["source_answer"],
    } for row in few_shot]
    write_jsonl_atomic(root / "prompts" / "few_shot.jsonl", prompt_examples)
    selected_ids = {row["problem_id"] for rows in splits.values() for row in rows}
    all_gold = {**train_gold, **test_gold}
    write_jsonl_atomic(
        root / "private" / "gold_answers.jsonl",
        (all_gold[problem_id] for problem_id in sorted(selected_ids)),
    )
    split_ids = {name: {row["problem_id"] for row in rows} for name, rows in splits.items()}
    for left_index, left in enumerate(splits):
        for right in list(splits)[left_index + 1:]:
            if split_ids[left] & split_ids[right]:
                raise PipelineError(f"Prepared splits overlap: {left}, {right}")
    manifest = {
        "created_at": utc_now(), "source_repo": dataset_cfg["repo_id"],
        "requested_revision": dataset_cfg.get("revision", "main"),
        "resolved_dataset": resolved_dataset,
        "split_seed": int(dataset_cfg["split_seed"]),
        "few_shot_source_indices": sorted(few_indices),
        "counts": {**{name: len(rows) for name, rows in splits.items()}, "few_shot": len(few_shot)},
        "sha256": {name: sha256_file(root / "splits" / f"{name}.jsonl") for name in splits},
        "few_shot_sha256": sha256_file(root / "prompts" / "few_shot.jsonl"),
        "gold_sha256": sha256_file(root / "private" / "gold_answers.jsonl"),
        "train_test_question_overlap": 0,
        "git_commit": git_commit(),
    }
    write_json_atomic(root / "splits" / "split_manifest.json", manifest)
    report = {"source_train": len(train), "source_test": len(test), "counts": manifest["counts"], "parse_failures": 0}
    write_json_atomic(root / "reports" / "dataset_report.json", report)
    return report


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
    print(canonical_json(prepare(config)))


if __name__ == "__main__":
    main()
