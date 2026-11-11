import json
import sys
import types

from pipeline.prepare_dataset import PUBLIC_FIELDS, extract_gold, prepare


def test_extract_gsm8k_gold():
    assert extract_gold("work <<1+1=2>>\n#### 2") == "2"
    assert extract_gold("work\n#### 1,234") == "1234"


def test_prepare_gsm8k_splits_are_disjoint_and_hide_answers(tmp_path, monkeypatch):
    train = [
        {"question": f"Train question {index}?", "answer": f"work {index}\n#### {index}"}
        for index in range(12)
    ]
    test = [
        {"question": f"Test question {index}?", "answer": f"test work {index}\n#### {index}"}
        for index in range(3)
    ]
    calls = []
    def fake_load_dataset(repo_id, config_name, revision):
        calls.append((repo_id, config_name, revision))
        return {"train": train, "test": test}
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    config = {
        "dataset": {
            "repo_id": "openai/gsm8k", "config_name": "main", "revision": "main",
            "split_seed": 42, "few_shot_indices": [0, 2], "validation_size": 2,
            "generation_size": 5,
        },
        "paths": {"root": str(tmp_path / "data"), "run_root": str(tmp_path / "run")},
    }
    report = prepare(config)
    assert calls == [("openai/gsm8k", "main", "main")]
    assert report["counts"] == {
        "seed_train": 5, "validation": 2, "reserve": 3, "test": 3, "few_shot": 2,
    }
    root = tmp_path / "data"
    split_rows = {
        name: [json.loads(line) for line in (root / f"splits/{name}.jsonl").read_text().splitlines()]
        for name in ("seed_train", "validation", "reserve", "test")
    }
    assert all(set(row) == PUBLIC_FIELDS for rows in split_rows.values() for row in rows)
    ids = [set(row["problem_id"] for row in rows) for rows in split_rows.values()]
    assert all(not ids[i] & ids[j] for i in range(len(ids)) for j in range(i + 1, len(ids)))
    prompt_rows = [json.loads(line) for line in (root / "prompts/few_shot.jsonl").read_text().splitlines()]
    assert len(prompt_rows) == 2 and all("####" in row["answer"] for row in prompt_rows)
