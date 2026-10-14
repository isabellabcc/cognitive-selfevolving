"""Generate independent Qwen3 thinking trajectories with safe resume support."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Protocol

from .common import (
    PipelineError,
    append_jsonl,
    canonical_json,
    ensure_directories,
    experiment_root,
    freeze_config,
    load_config,
    read_jsonl,
    run_root,
    runtime_versions,
    sha256_text,
    stable_seed,
    utc_now,
    write_json_atomic,
)
from .parse_output import finish_reason, parse_completion


PROMPTS = {
    "math_cot_v1": (
        "Solve the following math problem step by step.\n"
        "Show the necessary mathematical reasoning.\n"
        "Put your final answer in \\boxed{{}}.\n\n"
        "Problem: {problem}"
    )
}

RAW_REQUIRED_FIELDS = {
    "trajectory_id", "run_id", "problem_id", "subset", "sample_index",
    "model_id", "model_revision", "prompt_version", "sampling_seed",
    "generation_config_sha256", "raw_completion", "reasoning_text", "final_text",
    "format_status", "finish_reason", "generation_status", "prompt_token_count",
    "completion_token_count", "completion_sha256", "created_at",
}


class Backend(Protocol):
    model_revision: str
    device: str
    dtype: str
    name: str

    def generate(self, prompt: str, seed: int, max_new_tokens: int) -> dict[str, Any]: ...


class MockBackend:
    name = "mock"
    model_revision = "mock-v1"
    device = "cpu"
    dtype = "none"

    def __init__(self, fixture_path: str | None = None):
        self._items: dict[tuple[str, int], dict[str, Any]] = {}
        if fixture_path:
            for row in read_jsonl(fixture_path):
                self._items[(str(row["problem_id"]), int(row["sample_index"]))] = row

    def set_task(self, problem_id: str, sample_index: int) -> None:
        self._task = (problem_id, sample_index)

    def generate(self, prompt: str, seed: int, max_new_tokens: int) -> dict[str, Any]:
        row = self._items.get(getattr(self, "_task", ("", -1)))
        if row is None:
            answer = seed % 17
            raw = f"<think>Deterministic mock reasoning for seed {seed}.</think>\n\\boxed{{{answer}}}"
            reason = "eos"
        else:
            raw = str(row.get("raw_completion", ""))
            reason = str(row.get("finish_reason", "eos"))
        return {
            "raw_completion": raw,
            "prompt_token_count": len(prompt.split()),
            "completion_token_count": len(raw.split()),
            "finish_reason": reason,
        }


class TransformersBackend:
    name = "transformers"

    def __init__(self, config: Mapping[str, Any]):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise PipelineError("torch and transformers are required for the transformers backend") from exc
        self.torch = torch
        model_cfg = config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["id"], revision=model_cfg["revision"])
        dtype_value = model_cfg.get("torch_dtype", "auto")
        kwargs: dict[str, Any] = {"revision": model_cfg["revision"], "torch_dtype": dtype_value}
        device_setting = model_cfg.get("device", "auto")
        if device_setting == "auto":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_cfg["id"], **kwargs)
        if device_setting not in ("auto", None):
            self.model.to(device_setting)
        self.model.eval()
        self.model_revision = str(
            getattr(self.model.config, "_commit_hash", None)
            or getattr(self.tokenizer, "_commit_hash", None)
            or model_cfg["revision"]
        )
        self.device = str(self.model.device)
        self.dtype = str(self.model.dtype).replace("torch.", "")
        self.config = config

    def generate(self, prompt: str, seed: int, max_new_tokens: int) -> dict[str, Any]:
        import numpy as np

        torch = self.torch
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        messages = [{"role": "user", "content": prompt}]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=bool(self.config["model"].get("enable_thinking", True)),
            return_tensors="pt",
        )
        encoded = encoded.to(self.model.device)
        attention_mask = torch.ones_like(encoded, dtype=torch.long)
        sampling = self.config["sampling"]
        with torch.inference_mode():
            output = self.model.generate(
                encoded,
                attention_mask=attention_mask,
                do_sample=bool(sampling["do_sample"]),
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                top_k=int(sampling["top_k"]),
                max_new_tokens=max_new_tokens,
            )
        new_tokens = output[0, encoded.shape[-1] :]
        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_set: set[int] = set()
        elif isinstance(eos_ids, int):
            eos_set = {eos_ids}
        else:
            eos_set = set(eos_ids)
        eos_reached = bool(len(new_tokens) and int(new_tokens[-1]) in eos_set)
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return {
            "raw_completion": raw,
            "prompt_token_count": int(encoded.shape[-1]),
            "completion_token_count": int(new_tokens.shape[-1]),
            "finish_reason": finish_reason(
                generated_token_count=int(new_tokens.shape[-1]),
                max_new_tokens=max_new_tokens,
                eos_reached=eos_reached,
            ),
        }


def generation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": config["run_id"],
        "master_seed": config["master_seed"],
        "model": dict(config["model"]),
        "sampling": dict(config["sampling"]),
    }


def paths_for_subset(config: Mapping[str, Any], subset: str) -> tuple[Path, Path]:
    root = experiment_root(config)
    if subset == "pilot":
        return root / "splits" / "dev.jsonl", root / "trajectories" / "pilot_dev_raw.jsonl"
    return root / "splits" / f"{subset}.jsonl", root / "trajectories" / f"{subset}_raw.jsonl"


def _repair_incomplete_last_line(path: Path, event_writer) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    if not lines or lines[-1].endswith((b"\n", b"\r")):
        return
    try:
        json.loads(lines[-1].decode("utf-8"))
        # Valid JSON without a newline is completed by adding one.
        path.write_bytes(data + b"\n")
        return
    except (UnicodeDecodeError, json.JSONDecodeError):
        corrupt = lines[-1]
    repaired = b"".join(lines[:-1])
    temporary = path.with_suffix(path.suffix + ".repair.tmp")
    temporary.write_bytes(repaired)
    os.replace(temporary, path)
    event_writer({
        "event": "incomplete_last_line_removed",
        "path": str(path),
        "removed_byte_count": len(corrupt),
        "created_at": utc_now(),
    })


def _load_completed(path: Path, expected_hash: str, run_id: str) -> set[str]:
    completed: set[str] = set()
    seen: set[str] = set()
    if not path.exists():
        return completed
    for row in read_jsonl(path):
        if row.get("run_id") != run_id or row.get("generation_config_sha256") != expected_hash:
            raise PipelineError(f"Existing trajectories in {path} belong to a different configuration")
        trajectory_id = str(row.get("trajectory_id", ""))
        if not trajectory_id:
            raise PipelineError(f"Trajectory without trajectory_id in {path}")
        if trajectory_id in seen:
            raise PipelineError(f"Duplicate trajectory_id in {path}: {trajectory_id}")
        seen.add(trajectory_id)
        if row.get("generation_status") == "success" and RAW_REQUIRED_FIELDS <= set(row):
            completed.add(trajectory_id)
        else:
            raise PipelineError(
                f"Incomplete/non-success raw record in {path}: {trajectory_id}. "
                "Failed attempts must remain in generation_events.jsonl."
            )
    return completed


def _update_run_manifest(
    config: Mapping[str, Any], backend: Backend, subset: str, limited: bool, generation_hash: str
) -> None:
    path = run_root(config) / "run_manifest.json"
    manifest: dict[str, Any] = {}
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != config["_meta"]["config_sha256"]:
            raise PipelineError("run_manifest config hash differs; use a new run_id")
        if manifest.get("generation_config_sha256") != generation_hash:
            raise PipelineError("Generation settings differ from run_manifest; use a new run_id")
        if manifest.get("model_revision") != backend.model_revision:
            raise PipelineError("Resolved model revision differs from run_manifest; use a new run_id")
    manifest.update({
        "run_id": config["run_id"],
        "config_sha256": config["_meta"]["config_sha256"],
        "generation_config_sha256": generation_hash,
        "model_id": config["model"]["id"],
        "model_revision": backend.model_revision,
        "backend": backend.name,
        "device": backend.device,
        "dtype": backend.dtype,
        "runtime": runtime_versions(),
        "updated_at": utc_now(),
    })
    subsets = manifest.setdefault("subsets", {})
    subsets[subset] = {"limited": limited, "updated_at": utc_now()}
    write_json_atomic(path, manifest)


def run_generation(
    config: Mapping[str, Any], subset: str, *, resume: bool, backend_name: str,
    limit: int | None = None, mock_fixture: str | None = None, max_retries: int = 2,
) -> dict[str, int]:
    if subset not in {"pilot", "seed_train", "anchor"}:
        raise PipelineError(f"Unsupported generation subset: {subset}")
    ensure_directories(config)
    input_path, output_path = paths_for_subset(config, subset)
    # Split records are public by contract. Reject accidental leakage before a
    # backend sees any prompt.
    questions = list(read_jsonl(input_path))
    forbidden = {"solution", "gold_answer", "source_solution"}
    for question in questions:
        leaked = forbidden & set(question)
        if leaked:
            raise PipelineError(f"Answer leakage fields in generation input: {sorted(leaked)}")
    questions.sort(key=lambda row: row["problem_id"])
    if subset == "pilot":
        questions = questions[: int(config["pilot"]["n_problems"])]
    if limit is not None:
        if limit <= 0:
            raise PipelineError("--limit must be positive")
        questions = questions[:limit]

    backend: Backend
    if backend_name == "mock":
        backend = MockBackend(mock_fixture)
    else:
        backend = TransformersBackend(config)
    gen_hash = sha256_text(canonical_json(generation_config(config)))
    freeze_config(config["_meta"]["config_path"], config)
    _update_run_manifest(config, backend, subset, limit is not None, gen_hash)

    events_path = run_root(config) / "generation_events.jsonl"
    summary = {"expected": len(questions) * int(config["sampling"]["n_per_problem"]), "success": 0, "failed": 0, "skipped": 0}
    with append_jsonl(events_path) as event, append_jsonl(output_path) as append_output:
        _repair_incomplete_last_line(output_path, event)
        existing = _load_completed(output_path, gen_hash, str(config["run_id"]))
        if existing and not resume:
            raise PipelineError(f"Output already contains trajectories; pass --resume: {output_path}")
        for question in questions:
            for sample_index in range(int(config["sampling"]["n_per_problem"])):
                trajectory_id = f"{config['run_id']}:{question['problem_id']}:s{sample_index:02d}"
                if trajectory_id in existing:
                    summary["skipped"] += 1
                    continue
                if isinstance(backend, MockBackend):
                    backend.set_task(str(question["problem_id"]), sample_index)
                seed = stable_seed(config["master_seed"], config["run_id"], question["problem_id"], sample_index)
                prompt_version = str(config["sampling"]["prompt_version"])
                try:
                    prompt = PROMPTS[prompt_version].format(problem=question["problem"])
                except KeyError as exc:
                    raise PipelineError(f"Unknown prompt_version: {prompt_version}") from exc
                started = time.perf_counter()
                last_error: Exception | None = None
                generated: dict[str, Any] | None = None
                for attempt in range(1, max_retries + 2):
                    try:
                        generated = backend.generate(prompt, seed, int(config["sampling"]["max_new_tokens"]))
                        break
                    except Exception as exc:  # logged with type; task is retried explicitly
                        last_error = exc
                        event({
                            "event": "generation_failed", "trajectory_id": trajectory_id,
                            "attempt": attempt, "error_type": type(exc).__name__,
                            "error": str(exc), "created_at": utc_now(),
                        })
                if generated is None:
                    summary["failed"] += 1
                    event({
                        "event": "generation_terminal_failure", "trajectory_id": trajectory_id,
                        "attempts": max_retries + 1, "error_type": type(last_error).__name__ if last_error else None,
                        "created_at": utc_now(),
                    })
                    continue
                parsed = parse_completion(str(generated["raw_completion"]))
                record = {
                    "trajectory_id": trajectory_id,
                    "run_id": config["run_id"],
                    "problem_id": question["problem_id"],
                    "subset": subset,
                    "sample_index": sample_index,
                    "model_id": config["model"]["id"],
                    "model_revision": backend.model_revision,
                    "prompt_version": prompt_version,
                    "sampling_seed": seed,
                    "generation_config_sha256": gen_hash,
                    **parsed.as_dict(),
                    "finish_reason": generated["finish_reason"],
                    "generation_status": "success",
                    "prompt_token_count": int(generated["prompt_token_count"]),
                    "completion_token_count": int(generated["completion_token_count"]),
                    "completion_sha256": sha256_text(parsed.raw_completion),
                    "created_at": utc_now(),
                }
                append_output(record)
                summary["success"] += 1
                event({
                    "event": "generation_success", "trajectory_id": trajectory_id,
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                    "completion_token_count": record["completion_token_count"], "created_at": utc_now(),
                })
        event({
            "event": "generation_run_summary", "subset": subset, **summary,
            "limited": limit is not None, "created_at": utc_now(),
        })
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subset", required=True, choices=("pilot", "seed_train", "anchor"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", choices=("transformers", "mock"), default="transformers")
    parser.add_argument("--mock-fixture")
    parser.add_argument("--max-retries", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_retries < 0:
        raise PipelineError("--max-retries must be non-negative")
    summary = run_generation(
        load_config(args.config), args.subset, resume=args.resume, backend_name=args.backend,
        limit=args.limit, mock_fixture=args.mock_fixture, max_retries=args.max_retries,
    )
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
