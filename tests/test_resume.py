import json

import pytest

from pipeline.common import PipelineError
from pipeline.generate import MockBackend, _load_completed, _repair_incomplete_last_line, run_generation, select_pilot_questions


def make_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    return {
        "run_id": "run", "master_seed": 7,
        "model": {"id": "mock", "revision": "mock-v1", "device": "cpu", "torch_dtype": "auto"},
        "sampling": {"prompt_version": "gsm8k_star_v1", "n_per_problem": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "do_sample": False, "max_new_tokens": 32, "batch_size": 1},
        "pilot": {"n_problems": 1, "selection_seed": 7},
        "paths": {"root": str(tmp_path / "data"), "run_root": str(tmp_path / "run")},
        "_meta": {"config_path": str(config_path), "config_sha256": "config-hash"},
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def add_prompt(config):
    root = config["paths"]["root"]
    from pathlib import Path
    write_jsonl(Path(root) / "prompts/few_shot.jsonl", [{"question": "Demo?", "answer": "work\n#### 1"}])


def test_incomplete_last_line_is_repaired(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"ok": 1}\n{"broken"')
    events = []
    _repair_incomplete_last_line(path, events.append)
    assert path.read_bytes() == b'{"ok": 1}\n'
    assert events[0]["event"] == "incomplete_last_line_removed"


def test_resume_retries_and_never_duplicates(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    add_prompt(config)
    write_jsonl(tmp_path / "data/splits/seed_train.jsonl", [{"problem_id": "p", "question": "1+0?", "question_sha256": "h"}])
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
    assert len((tmp_path / "data/trajectories/seed_train_raw.jsonl").read_text().splitlines()) == 1


def test_config_change_is_rejected(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"trajectory_id": "t", "run_id": "run", "generation_config_sha256": "old", "generation_status": "success"}])
    with pytest.raises(PipelineError):
        _load_completed(path, "new", "run")


def test_pilot_selection_is_seeded_and_non_mutating():
    questions = [{"problem_id": f"p{i}"} for i in range(8)]
    original = [dict(row) for row in questions]
    pilot = {"n_problems": 4, "selection_seed": 123}
    first = select_pilot_questions(questions, pilot, 999)
    second = select_pilot_questions(questions, pilot, 999)
    assert first == second
    assert questions == original
