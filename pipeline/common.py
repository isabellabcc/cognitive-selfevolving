"""Shared configuration, hashing, path, and crash-safe JSONL helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import subprocess
import tempfile
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


class PipelineError(RuntimeError):
    """An actionable pipeline or data-contract error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(config))


def normalize_problem(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def normalize_light(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def stable_seed(*parts: object, bits: int = 63) -> int:
    raw = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:8], "big") & ((1 << bits) - 1)


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise PipelineError("PyYAML is required to read the experiment config") from exc
    config_path = Path(path)
    if not config_path.is_file():
        raise PipelineError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise PipelineError("Config root must be a mapping")
    validate_config(value)
    value["_meta"] = {
        "config_path": str(config_path.resolve()),
        "config_sha256": config_hash(value),
    }
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"project", "run_id", "master_seed", "dataset", "model", "sampling", "pilot", "star", "paths"}
    missing = sorted(required - set(config))
    if missing:
        raise PipelineError(f"Missing config sections: {', '.join(missing)}")
    dataset = config["dataset"]
    if dataset.get("repo_id") != "openai/gsm8k" or dataset.get("config_name") != "main":
        raise PipelineError("This project requires dataset openai/gsm8k, config main")
    for field in ("validation_size", "generation_size"):
        if int(dataset.get(field, 0)) <= 0:
            raise PipelineError(f"dataset.{field} must be positive")
    few_shot_indices = dataset.get("few_shot_indices")
    if not isinstance(few_shot_indices, list) or not few_shot_indices:
        raise PipelineError("dataset.few_shot_indices must be a non-empty list")
    if any(int(index) < 0 for index in few_shot_indices) or len(set(few_shot_indices)) != len(few_shot_indices):
        raise PipelineError("dataset.few_shot_indices must be unique non-negative integers")
    if len(few_shot_indices) + sum(int(dataset[field]) for field in ("validation_size", "generation_size")) >= 7473:
        raise PipelineError("Configured GSM8K train partitions leave no reserve examples")
    model_id = config["model"].get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise PipelineError("model.id must be a non-empty Hugging Face model ID or local path")
    if int(config["sampling"].get("batch_size", 0)) != 1:
        raise PipelineError("Experiment v1 requires sampling.batch_size=1")
    if config["sampling"].get("prompt_version") != "gsm8k_star_v1":
        raise PipelineError("sampling.prompt_version must be gsm8k_star_v1")
    if int(config["sampling"].get("n_per_problem", 0)) != 1:
        raise PipelineError("The registered single-trajectory experiment requires n_per_problem=1")
    if bool(config["sampling"].get("do_sample", True)):
        raise PipelineError("The registered single-trajectory experiment must use greedy decoding")
    pilot = config["pilot"]
    try:
        pilot_count = int(pilot.get("n_problems", 0))
    except (TypeError, ValueError) as exc:
        raise PipelineError("pilot.n_problems must be an integer") from exc
    if pilot_count <= 0:
        raise PipelineError("pilot.n_problems must be positive")
    if pilot_count > int(dataset["generation_size"]):
        raise PipelineError("pilot.n_problems cannot exceed dataset.generation_size")
    try:
        int(pilot.get("selection_seed", config["master_seed"]))
    except (TypeError, ValueError) as exc:
        raise PipelineError("pilot.selection_seed must be an integer") from exc
    star = config.get("star")
    if star is not None:
        if not isinstance(star, Mapping):
            raise PipelineError("star must be a mapping")
        training = star.get("training", {})
        if float(training.get("learning_rate", 0)) <= 0:
            raise PipelineError("star.training.learning_rate must be positive")
        if int(training.get("max_seq_length", 0)) <= 0:
            raise PipelineError("star.training.max_seq_length must be positive")
        if int(training.get("epochs", 0)) <= 0:
            raise PipelineError("star.training.epochs must be positive")
        evaluation = star.get("evaluation", {})
        if int(evaluation.get("max_new_tokens", 0)) <= 0:
            raise PipelineError("star.evaluation.max_new_tokens must be positive")


def experiment_root(config: Mapping[str, Any]) -> Path:
    return Path(config["paths"]["root"])


def run_root(config: Mapping[str, Any]) -> Path:
    return Path(config["paths"]["run_root"])


def ensure_directories(config: Mapping[str, Any]) -> None:
    root = experiment_root(config)
    for relative in ("raw", "splits", "prompts", "private", "trajectories", "verification", "star", "reports", "audits"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    run_root(config).mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path, *, tolerate_incomplete_last_line: bool = False) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise PipelineError(f"Required JSONL file does not exist: {source}")
    with source.open("rb") as handle:
        lines = handle.readlines()
    for index, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            decoded = raw.decode("utf-8")
            item = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_last = index == len(lines)
            has_newline = raw.endswith((b"\n", b"\r"))
            if tolerate_incomplete_last_line and is_last and not has_newline:
                return
            raise PipelineError(f"Invalid JSONL at {source}:{index}: {exc}") from exc
        if not isinstance(item, dict):
            raise PipelineError(f"JSONL item at {source}:{index} is not an object")
        yield item


def write_jsonl_atomic(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


@contextmanager
def append_jsonl(path: str | Path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n", buffering=1) as handle:
        def append(record: Mapping[str, Any]) -> None:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield append


def freeze_config(config_path: str | Path, config: Mapping[str, Any]) -> None:
    frozen = run_root(config) / "config_frozen.yaml"
    source = Path(config_path).read_bytes()
    if frozen.exists():
        if frozen.read_bytes() != source:
            raise PipelineError(f"Run config conflicts with frozen config: {frozen}. Use a new run_id.")
        return
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_bytes(source)


def runtime_versions() -> dict[str, Any]:
    result: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "transformers", "datasets", "peft", "math_verify", "sympy"):
        try:
            module = __import__(package)
            result[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            result[package] = None
    return result


def git_commit() -> str | None:
    """Return the repository commit without making Git a runtime requirement."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def seeded_order(items: Iterable[Any], seed: int, key=lambda item: item) -> list[Any]:
    ordered = sorted(items, key=key)
    random.Random(seed).shuffle(ordered)
    return ordered
