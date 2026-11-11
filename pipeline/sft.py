"""Completion-only LoRA SFT for a frozen STaR selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import (
    PipelineError,
    canonical_json,
    experiment_root,
    git_commit,
    load_config,
    read_jsonl,
    run_root,
    runtime_versions,
    sha256_file,
    sha256_text,
    utc_now,
    write_json_atomic,
)
from .prompts import PROMPT_VERSION, build_prompt, load_few_shot


def training_prompt(
    question: str, prompt_version: str, examples: Sequence[Mapping[str, Any]]
) -> str:
    if prompt_version != PROMPT_VERSION:
        raise PipelineError(f"Unknown training prompt version: {prompt_version}")
    return build_prompt(question, examples)


def _prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def tokenize_prompt_completion(
    tokenizer: Any, prompt: str, completion: str, max_seq_length: int,
) -> dict[str, list[int]]:
    """Tokenize the exact generation prefix and mask every non-completion token."""
    prompt_ids = _prompt_ids(tokenizer, prompt)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if hasattr(completion_ids, "tolist"):
        completion_ids = completion_ids.tolist()
    completion_ids = [int(token) for token in completion_ids]
    if not completion_ids:
        raise PipelineError("Refusing an empty SFT completion")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and completion_ids[-1] != int(eos_id):
        completion_ids.append(int(eos_id))
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_seq_length:
        raise PipelineError(
            f"SFT example has {len(input_ids)} tokens, exceeding max_seq_length={max_seq_length}; "
            "the final answer will not be silently truncated"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def pad_features(
    features: Sequence[Mapping[str, list[int]]], pad_token_id: int
) -> dict[str, list[list[int]]]:
    """Pad features while keeping prompt and padding labels masked."""
    max_length = max(len(feature["input_ids"]) for feature in features)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        padding = max_length - len(feature["input_ids"])
        batch["input_ids"].append(feature["input_ids"] + [int(pad_token_id)] * padding)
        batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
        batch["labels"].append(feature["labels"] + [-100] * padding)
    return batch


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: Sequence[Mapping[str, list[int]]]) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise PipelineError("torch is required to collate SFT batches") from exc
        batch = pad_features(features, self.pad_token_id)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def _selection_paths(config: Mapping[str, Any], selection_id: str) -> tuple[Path, Path]:
    directory = experiment_root(config) / "star" / "selections" / selection_id
    return directory / "selected.jsonl", directory / "selection_manifest.json"


def describe_training(config: Mapping[str, Any], selection_id: str, seed: int) -> dict[str, Any]:
    selected_path, manifest_path = _selection_paths(config, selection_id)
    if not selected_path.exists() or not manifest_path.exists():
        raise PipelineError(f"Unknown/incomplete selection: {selection_id}")
    rows = list(read_jsonl(selected_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "mode": "plan_only",
        "selection_id": selection_id,
        "selection_pool_hash": manifest["pool_hash"],
        "n_examples": len(rows),
        "seed": seed,
        "base_model": config["model"]["id"],
        "requires_explicit_run": True,
    }


def train_adapter(
    config: Mapping[str, Any], selection_id: str, seed: int, *, local_files_only: bool = False
) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise PipelineError("STaR training requires torch, transformers, accelerate, and peft") from exc

    selected_path, selection_manifest_path = _selection_paths(config, selection_id)
    selected = list(read_jsonl(selected_path))
    if not selected:
        raise PipelineError("Cannot train on an empty selection")
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    questions = {
        row["problem_id"]: row
        for row in read_jsonl(experiment_root(config) / "splits" / "seed_train.jsonl")
    }
    few_shot = load_few_shot(experiment_root(config) / "prompts" / "few_shot.jsonl")
    missing = {row["problem_id"] for row in selected} - set(questions)
    if missing:
        raise PipelineError(f"Selection refers to an unknown problem: {sorted(missing)[0]}")

    model_cfg = config["model"]
    local_files_only = local_files_only or bool(model_cfg.get("local_files_only", False))
    training_cfg = config["star"]["training"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["id"], revision=model_cfg["revision"], local_files_only=local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    features = []
    excluded = []
    for row in selected:
        prompt = training_prompt(
            str(questions[row["problem_id"]]["question"]), str(row["prompt_version"]), few_shot
        )
        try:
            features.append(tokenize_prompt_completion(
                tokenizer, prompt, str(row["raw_completion"]),
                int(training_cfg["max_seq_length"]),
            ))
        except PipelineError as exc:
            excluded.append({"trajectory_id": row["trajectory_id"], "reason": str(exc)})
    if excluded:
        raise PipelineError(
            f"{len(excluded)} selected examples exceed/violate the SFT contract; "
            f"first: {excluded[0]}"
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
    targets = [str(value) for value in training_cfg["target_modules"]]
    present_suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    absent = sorted(set(targets) - present_suffixes)
    if absent:
        raise PipelineError(f"LoRA target modules are absent from the base model: {absent}")
    initial_adapter = model_cfg.get("adapter_path")
    if initial_adapter:
        adapter_path = Path(str(initial_adapter))
        if not adapter_path.is_dir():
            raise PipelineError(f"Configured initial adapter does not exist: {adapter_path}")
        model = PeftModel.from_pretrained(
            model, str(adapter_path), is_trainable=True, local_files_only=local_files_only
        )
    else:
        lora = LoraConfig(
            r=int(training_cfg["lora_r"]),
            lora_alpha=int(training_cfg["lora_alpha"]),
            lora_dropout=float(training_cfg["lora_dropout"]),
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    if bool(training_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False

    train_id = f"{selection_id}-train-s{seed}"
    output_dir = run_root(config) / "star" / train_id
    output_dir.mkdir(parents=True, exist_ok=False)
    use_bf16 = bool(
        dtype_name == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        data_seed=seed,
        learning_rate=float(training_cfg["learning_rate"]),
        num_train_epochs=float(training_cfg["epochs"]),
        per_device_train_batch_size=int(training_cfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        warmup_ratio=float(training_cfg["warmup_ratio"]),
        weight_decay=float(training_cfg["weight_decay"]),
        bf16=use_bf16,
        fp16=bool(dtype_name == "float16" and torch.cuda.is_available()),
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
    )
    supervised_tokens = sum(sum(label != -100 for label in item["labels"]) for item in features)
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=features,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
    )
    result = trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    elapsed = time.perf_counter() - started
    manifest = {
        "train_id": train_id,
        "created_at": utc_now(),
        "selection_id": selection_id,
        "selection_sha256": sha256_file(selected_path),
        "pool_hash": selection_manifest["pool_hash"],
        "base_model": model_cfg["id"],
        "requested_model_revision": model_cfg["revision"],
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "initial_adapter": str(initial_adapter) if initial_adapter else None,
        "training_config_sha256": sha256_text(canonical_json(dict(training_cfg))),
        "n_examples": len(features),
        "supervised_tokens_per_epoch": supervised_tokens,
        "train_tokens_seen_estimate": supervised_tokens * float(training_cfg["epochs"]),
        "global_steps": int(result.global_step),
        "train_loss": float(result.training_loss),
        "train_seconds": elapsed,
        "gpu_peak_gb": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        ),
        "git_commit": git_commit(),
        "runtime": runtime_versions(),
    }
    write_json_atomic(output_dir / "train_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run", action="store_true", help="Actually load weights and start training")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.run:
        result = train_adapter(
            config, args.selection_id, args.seed, local_files_only=args.local_files_only
        )
    else:
        result = describe_training(config, args.selection_id, args.seed)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
