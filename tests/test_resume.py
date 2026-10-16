import json
from pathlib import Path

import pytest
import yaml

from pipeline.common import PipelineError, append_jsonl
from pipeline.generate import (
    MockBackend,
    _load_completed,
    _repair_incomplete_last_line,
    run_generation,
    select_pilot_questions,
)


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


def test_pilot_hardest_selection_is_reproducible_and_does_not_mutate_source():
    questions = [
        {"problem_id": f"p{index}", "level": level}
        for index, level in enumerate([1, 5, 3, 4, 5, 2, 4, 3])
    ]
    original = [dict(row) for row in questions]
    pilot = {"n_problems": 4, "selection_strategy": "hardest", "selection_seed": 123}

    first = select_pilot_questions(questions, pilot, master_seed=999)
    second = select_pilot_questions(questions, pilot, master_seed=999)

    assert [row["problem_id"] for row in first] == [row["problem_id"] for row in second]
    assert [row["level"] for row in first] == [5, 5, 4, 4]
    assert questions == original


def test_pilot_level_counts_selects_exact_configured_composition():
    questions = [
        {"problem_id": f"l{level}-{index}", "level": level}
        for level in range(1, 6)
        for index in range(4)
    ]
    pilot = {
        "n_problems": 6,
        "selection_strategy": "level_counts",
        "selection_seed": 321,
        "level_counts": {1: 0, 2: 1, 3: 2, 4: 2, 5: 1},
    }

    selected = select_pilot_questions(questions, pilot, master_seed=999)

    assert {level: sum(row["level"] == level for row in selected) for level in range(1, 6)} == {
        1: 0, 2: 1, 3: 2, 4: 2, 5: 1,
    }
    assert [row["problem_id"] for row in selected] == [
        row["problem_id"] for row in select_pilot_questions(questions, pilot, master_seed=999)
    ]


def test_configured_pilot_level_counts_sum_to_n_problems():
    config_path = Path(__file__).parents[1] / "configs" / "experiment1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert sum(config["pilot"]["level_counts"].values()) == config["pilot"]["n_problems"]


def test_pilot_generation_records_numbered_selected_questions(tmp_path):
    config = make_config(tmp_path)
    config["pilot"] = {
        "source_subset": "dev",
        "n_problems": 2,
        "selection_strategy": "level_counts",
        "selection_seed": 17,
        "level_counts": {1: 0, 2: 0, 3: 1, 4: 0, 5: 1},
    }
    split = tmp_path / "data/splits/dev.jsonl"
    split.parent.mkdir(parents=True)
    split.write_text("".join(
        json.dumps({
            "problem_id": problem_id,
            "problem": "Compute 1+0.",
            "level": level,
            "topic": "Algebra",
        }) + "\n"
        for problem_id, level in (("p3", 3), ("p5", 5))
    ), encoding="utf-8")

    summary = run_generation(config, "pilot", resume=True, backend_name="mock")

    raw_path = tmp_path / "data/trajectories/pilot_dev_raw.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    selection_path = tmp_path / "data/reports/pilot_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert [row["question_index"] for row in raw_rows] == [1, 2]
    assert summary["selected_questions"] == selection["selected_questions"]
    assert {row["problem_id"] for row in selection["selected_questions"]} == {"p3", "p5"}
