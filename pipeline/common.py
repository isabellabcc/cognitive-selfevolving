"""Shared configuration, hashing, path, and crash-safe JSONL helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
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
    required = {"project", "run_id", "master_seed", "dataset", "model", "sampling", "pilot", "pool", "paths"}
    missing = sorted(required - set(config))
    if missing:
        raise PipelineError(f"Missing config sections: {', '.join(missing)}")
    configs = config["dataset"].get("configs", [])
    if len(configs) != 7 or len(set(configs)) != 7:
        raise PipelineError("dataset.configs must contain seven unique MATH configurations")
    sizes = config["dataset"].get("subset_sizes", {})
    if any(int(sizes.get(name, 0)) <= 0 for name in ("seed_train", "anchor", "dev")):
        raise PipelineError("All dataset subset sizes must be positive")
    ratios = {int(k): float(v) for k, v in config["dataset"].get("target_level_ratios", {}).items()}
    if set(ratios) != {1, 2, 3, 4, 5} or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise PipelineError("target_level_ratios must define levels 1..5 and sum to 1")
    if int(config["sampling"].get("batch_size", 0)) != 1:
        raise PipelineError("Experiment v1 requires sampling.batch_size=1")
    if config["sampling"].get("system_prompt") is not None:
        raise PipelineError("Experiment v1 requires sampling.system_prompt=null")
    threshold = float(config["pool"].get("near_duplicate_jaccard_threshold", -1))
    if not 0 <= threshold <= 1:
        raise PipelineError("pool.near_duplicate_jaccard_threshold must be between 0 and 1")
    if int(config["pool"].get("dedup_token_ngram", 0)) <= 0:
        raise PipelineError("pool.dedup_token_ngram must be positive")


def experiment_root(config: Mapping[str, Any]) -> Path:
    return Path(config["paths"]["root"])


def run_root(config: Mapping[str, Any]) -> Path:
    return Path(config["paths"]["run_root"])


def ensure_directories(config: Mapping[str, Any]) -> None:
    root = experiment_root(config)
    for relative in ("raw", "splits", "private", "trajectories", "verification", "pool", "reports", "audits"):
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
    for package in ("torch", "transformers", "datasets", "math_verify", "sympy"):
        try:
            module = __import__(package)
            result[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            result[package] = None
    return result


def seeded_order(items: Iterable[Any], seed: int, key=lambda item: item) -> list[Any]:
    ordered = sorted(items, key=key)
    random.Random(seed).shuffle(ordered)
    return ordered
