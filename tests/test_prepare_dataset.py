import json
import sys
import types
from pathlib import Path

from pipeline.prepare_dataset import (
    PUBLIC_FIELDS,
    assert_split_contract,
    extract_balanced_boxed,
    parse_level,
    prepare,
    split_selected,
)


def test_parse_level_and_nested_boxed():
    assert parse_level("Level 4") == 4
    assert parse_level(2) == 2
    assert extract_balanced_boxed(r"work \boxed{\frac{1}{2}} end") == [r"\frac{1}{2}"]


def test_seeded_split_is_disjoint_and_has_no_gold():
    records = []
    for index in range(6):
        record = {
            "problem_id": f"p{index}", "source_repo": "r", "source_config": "c",
            "source_split": "train", "source_row_index": index, "level": 1,
            "topic": "Algebra", "problem": f"q{index}", "problem_sha256": f"h{index}",
        }
        records.append(record)
    first = split_selected(records, {"seed_train": 3, "anchor": 2, "dev": 1}, 42)
    second = split_selected(records, {"seed_train": 3, "anchor": 2, "dev": 1}, 42)
    assert first == second
    assert_split_contract(first, {"seed_train": 3, "anchor": 2, "dev": 1})
    assert all(set(row) == PUBLIC_FIELDS for rows in first.values() for row in rows)


def test_all_seven_configs_are_loaded(tmp_path, monkeypatch):
    config_names = [f"config_{index}" for index in range(7)]
    calls = []

    def fake_load_dataset(repo_id, name, split):
        calls.append(name)
        base = config_names.index(name) * 5
        return [
            {
                "problem": f"Problem {base + index}", "solution": rf"proof \boxed{{{base + index}}}",
                "level": f"Level {(base + index) % 5 + 1}", "type": f"Topic{index % 2}",
            }
            for index in range(5)
        ]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    config = {
        "dataset": {
            "repo_id": "repo", "split": "train", "configs": config_names,
            "subset_sizes": {"seed_train": 2, "anchor": 1, "dev": 1},
            "target_level_ratios": {1: .2, 2: .2, 3: .2, 4: .2, 5: .2},
            "exclude_asymptote": True, "split_seed": 42,
        },
        "paths": {"root": str(tmp_path / "data"), "run_root": str(tmp_path / "run")},
    }
    report = prepare(config)
    assert calls == config_names
    assert report["raw_total"] == 35
    public_rows = [json.loads(line) for line in (tmp_path / "data/splits/seed_train.jsonl").read_text().splitlines()]
    assert all(not ({"solution", "gold_answer"} & set(row)) for row in public_rows)
