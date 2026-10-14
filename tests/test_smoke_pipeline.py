import json
from pathlib import Path

from pipeline.build_pool import build_pool
from pipeline.generate import run_generation
from pipeline.report import generate_report
from pipeline.verify import run_verification


def test_mock_generate_verify_pool_report(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mock: true\n", encoding="utf-8")
    root = tmp_path / "data"
    run_root = tmp_path / "run"
    config = {
        "run_id": "smoke", "master_seed": 11,
        "model": {"id": "mock", "revision": "mock-v1", "enable_thinking": True, "device": "cpu", "torch_dtype": "auto"},
        "sampling": {"prompt_version": "math_cot_v1", "n_per_problem": 3, "temperature": .6, "top_p": .95, "top_k": 20, "do_sample": True, "max_new_tokens": 64, "batch_size": 1},
        "pilot": {"n_problems": 3},
        "pool": {"min_distinct_correct": 2, "target_n_problems": 1, "near_duplicate_jaccard_threshold": .9, "dedup_token_ngram": 5, "pool_seed": 11},
        "paths": {"root": str(root), "run_root": str(run_root)},
        "_meta": {"config_path": str(config_path), "config_sha256": "smoke-config"},
    }
    split_dir = root / "splits"
    private_dir = root / "private"
    split_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    questions = [
        {"problem_id": f"p{i}", "problem": "Compute 1.", "level": i, "topic": "Algebra"}
        for i in range(1, 4)
    ]
    (split_dir / "seed_train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in questions), encoding="utf-8"
    )
    (private_dir / "gold_answers.jsonl").write_text(
        "".join(json.dumps({"problem_id": row["problem_id"], "gold_answer": "1", "gold_extraction_status": "ok"}) + "\n" for row in questions),
        encoding="utf-8",
    )
    fixture = Path(__file__).parent / "fixtures" / "mock_completions.jsonl"
    generated = run_generation(
        config, "seed_train", resume=True, backend_name="mock", mock_fixture=str(fixture)
    )
    assert generated["success"] == 9
    monkeypatch.setattr("pipeline.verify.load_math_verifier", lambda: (lambda value: value, lambda gold, pred: gold == pred, "test"))
    verified = run_verification(config, "seed_train")
    assert verified["counts"] == {"correct": 3, "incorrect": 3, "unverifiable": 3}
    manifest = build_pool(config, freeze=False)
    assert manifest["statistics"]["selected_problem_count"] == 1
    report = generate_report(config, "all")
    assert report["pool"]["selected_problem_count"] == 1
