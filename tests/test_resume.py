import json
from pathlib import Path

import pytest

from pipeline.common import PipelineError, append_jsonl
from pipeline.generate import MockBackend, _load_completed, _repair_incomplete_last_line, run_generation


def make_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    return {
        "run_id": "run", "master_seed": 7,
        "model": {"id": "mock", "revision": "mock-v1", "enable_thinking": True, "device": "cpu", "torch_dtype": "auto"},
        "sampling": {"prompt_version": "math_cot_v1", "n_per_problem": 1, "temperature": .6, "top_p": .95, "top_k": 20, "do_sample": True, "max_new_tokens": 32, "batch_size": 1},
        "pilot": {"n_problems": 1}, "pool": {},
        "paths": {"root": str(tmp_path / "data"), "run_root": str(tmp_path / "run")},
        "_meta": {"config_path": str(config_path), "config_sha256": "config-hash"},
    }


def test_incomplete_last_line_is_detected_and_repaired(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"ok": 1}\n{"broken"')
    events = []
    _repair_incomplete_last_line(path, events.append)
    assert path.read_bytes() == b'{"ok": 1}\n'
    assert events[0]["event"] == "incomplete_last_line_removed"


def test_resume_retries_failure_and_never_duplicates(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    split = tmp_path / "data/splits/seed_train.jsonl"
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps({"problem_id": "p", "problem": "1+0?"}) + "\n", encoding="utf-8")
    original = MockBackend.generate
    attempts = {"count": 0}

    def flaky(self, prompt, seed, max_new_tokens):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient")
        return original(self, prompt, seed, max_new_tokens)

    monkeypatch.setattr(MockBackend, "generate", flaky)
    first = run_generation(config, "seed_train", resume=True, backend_name="mock", max_retries=1)
    second = run_generation(config, "seed_train", resume=True, backend_name="mock", max_retries=1)
    assert first["success"] == 1 and second["skipped"] == 1
    output = tmp_path / "data/trajectories/seed_train_raw.jsonl"
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_config_change_is_rejected(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({
        "trajectory_id": "t", "run_id": "run", "generation_config_sha256": "old",
        "generation_status": "success",
    }) + "\n", encoding="utf-8")
    with pytest.raises(PipelineError):
        _load_completed(path, "new", "run")
