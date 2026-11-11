"""Deterministic Pass@1 evaluation for a base model or STaR LoRA adapter."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping

from .common import (
    PipelineError,
    canonical_json,
    experiment_root,
    git_commit,
    load_config,
    read_jsonl,
    run_root,
    sha256_text,
    stable_seed,
    utc_now,
    write_json_atomic,
    write_jsonl_atomic,
)
from .parse_output import finish_reason, parse_completion
from .prompts import PROMPT_VERSION, build_prompt, load_few_shot
from .generate import gsm8k_stopping_criteria
from .verify import VERIFIER_CONFIG, load_math_verifier, verify_record


def bootstrap_accuracy(
    correctness: list[bool], *, seed: int, samples: int = 2000
) -> tuple[float | None, float | None]:
    if not correctness:
        return None, None
    rng = random.Random(seed)
    size = len(correctness)
    estimates = sorted(
        sum(correctness[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    return estimates[int(samples * 0.025)], estimates[min(samples - 1, int(samples * 0.975))]


def summarize_predictions(rows: list[Mapping[str, Any]], seed: int) -> dict[str, Any]:
    correctness = [row.get("verification_status") == "correct" for row in rows]
    lower, upper = bootstrap_accuracy(correctness, seed=seed)
    parse_failed = sum(row.get("verification_status") == "unverifiable" for row in rows)
    return {
        "correct": sum(correctness),
        "total": len(rows),
        "pass_at_1": sum(correctness) / len(rows) if rows else None,
        "bootstrap_95_ci": [lower, upper],
        "unverifiable_count": parse_failed,
        "unverifiable_rate": parse_failed / len(rows) if rows else None,
    }


def describe_evaluation(
    config: Mapping[str, Any], checkpoint: str, split: str, limit: int | None
) -> dict[str, Any]:
    split_path = experiment_root(config) / "splits" / f"{split}.jsonl"
    count = sum(1 for _ in read_jsonl(split_path))
    return {
        "mode": "plan_only",
        "checkpoint": checkpoint,
        "split": split,
        "n_problems": min(count, limit) if limit is not None else count,
        "greedy": True,
        "requires_explicit_run": True,
    }


def evaluate(
    config: Mapping[str, Any], checkpoint: str, split: str, *,
    limit: int | None = None, local_files_only: bool = False,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise PipelineError("Evaluation requires torch and transformers") from exc
    if split not in {"validation", "test"}:
        raise PipelineError("STaR evaluation is restricted to validation or test")
    root = experiment_root(config)
    questions = list(read_jsonl(root / "splits" / f"{split}.jsonl"))
    questions.sort(key=lambda row: str(row["problem_id"]))
    few_shot = load_few_shot(root / "prompts" / "few_shot.jsonl")
    if limit is not None:
        if limit <= 0:
            raise PipelineError("--limit must be positive")
        questions = questions[:limit]
    gold = {row["problem_id"]: row for row in read_jsonl(root / "private" / "gold_answers.jsonl")}
    model_cfg = config["model"]
    local_files_only = local_files_only or bool(model_cfg.get("local_files_only", False))
    tokenizer_source = model_cfg["id"] if checkpoint == "base" else checkpoint
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=model_cfg["revision"] if checkpoint == "base" else None,
        local_files_only=local_files_only,
    )
    dtype_name = model_cfg.get("torch_dtype", "auto")
    dtype_value = {
        "float16": torch.float16, "bfloat16": torch.bfloat16,
        "float32": torch.float32, "auto": "auto",
    }.get(dtype_name)
    if dtype_value is None:
        raise PipelineError(f"Unsupported model.torch_dtype: {dtype_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["id"], revision=model_cfg["revision"],
        torch_dtype=dtype_value, local_files_only=local_files_only, low_cpu_mem_usage=True,
    )
    if checkpoint != "base":
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise PipelineError("peft is required to evaluate a LoRA adapter") from exc
        model = PeftModel.from_pretrained(model, checkpoint, local_files_only=local_files_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    parse_fn, verify_fn, verifier_version = load_math_verifier()
    verifier_hash = sha256_text(canonical_json({
        **VERIFIER_CONFIG, "math_verify_version": verifier_version,
    }))
    max_new_tokens = int(config["star"]["evaluation"]["max_new_tokens"])
    prompt_version = str(config["sampling"]["prompt_version"])
    if prompt_version != PROMPT_VERSION:
        raise PipelineError(f"Unknown prompt version: {prompt_version}")
    predictions = []
    for question in questions:
        prompt = build_prompt(str(question["question"]), few_shot)
        encoded_batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = encoded_batch["input_ids"].to(device)
        attention_mask = encoded_batch["attention_mask"].to(device)
        with torch.inference_mode():
            generated = model.generate(
                encoded, attention_mask=attention_mask, do_sample=False,
                max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=gsm8k_stopping_criteria(tokenizer, int(encoded.shape[-1])),
            )
        new_tokens = generated[0, encoded.shape[-1]:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed = parse_completion(raw_text)
        eos = model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        raw = {
            "trajectory_id": f"eval:{split}:{question['problem_id']}",
            "problem_id": question["problem_id"],
            "subset": split,
            "generation_status": "success",
            "finish_reason": finish_reason(
                generated_token_count=int(new_tokens.shape[-1]),
                max_new_tokens=max_new_tokens,
                eos_reached=bool(len(new_tokens) and int(new_tokens[-1]) in eos_ids),
            ),
            **parsed.as_dict(),
            "completion_token_count": int(new_tokens.shape[-1]),
        }
        verification = verify_record(raw, gold[question["problem_id"]], parse_fn, verify_fn, verifier_hash)
        predictions.append({**raw, **verification})
    eval_seed = stable_seed(config["master_seed"], "evaluation", checkpoint, split)
    metrics = summarize_predictions(predictions, eval_seed)
    eval_id = sha256_text(canonical_json({
        "checkpoint": str(checkpoint), "split": split, "limit": limit,
        "config": config["_meta"]["config_sha256"],
    }))[:16]
    output = run_root(config) / "star" / "evaluations" / eval_id
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl_atomic(output / "predictions.jsonl", predictions)
    report = {
        **metrics,
        "eval_id": eval_id,
        "created_at": utc_now(),
        "checkpoint": checkpoint,
        "base_model": model_cfg["id"],
        "split": split,
        "eval_seed": eval_seed,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "git_commit": git_commit(),
    }
    write_json_atomic(output / "metrics.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Adapter directory, or 'base'")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run", action="store_true", help="Actually load weights and generate")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.run:
        result = evaluate(
            config, args.checkpoint, args.split, limit=args.limit,
            local_files_only=args.local_files_only,
        )
    else:
        result = describe_evaluation(config, args.checkpoint, args.split, args.limit)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
