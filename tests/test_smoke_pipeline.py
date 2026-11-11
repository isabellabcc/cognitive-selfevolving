import json
from pathlib import Path

from pipeline.generate import run_generation
from pipeline.star_pool import build_star_pool, run_selection
from pipeline.verify import run_verification


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_mock_generate_verify_fixed_pool_and_select(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mock: true\n", encoding="utf-8")
    root = tmp_path / "data"
    config = {
        "run_id": "smoke", "master_seed": 42,
        "model": {"id": "mock", "revision": "mock-v1", "device": "cpu", "torch_dtype": "auto"},
        "sampling": {"prompt_version": "gsm8k_star_v1", "n_per_problem": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "do_sample": False, "max_new_tokens": 32, "batch_size": 1},
        "pilot": {"n_problems": 2, "selection_seed": 42},
        "paths": {"root": str(root), "run_root": str(tmp_path / "run")},
        "_meta": {"config_path": str(config_path), "config_sha256": "smoke-config"},
    }
    questions = [
        {"problem_id": f"p{i}", "question": f"Question {i}?", "question_sha256": f"h{i}"}
        for i in range(1, 4)
    ]
    write_jsonl(root / "splits/seed_train.jsonl", questions)
    write_jsonl(root / "prompts/few_shot.jsonl", [{
        "problem_id": "demo", "question": "Demo?", "answer": "work\n#### 1",
    }])
    write_jsonl(root / "private/gold_answers.jsonl", [
        {"problem_id": f"p{i}", "gold_answer": str(i), "gold_extraction_status": "ok"}
        for i in range(1, 4)
    ])
    fixture = Path(__file__).parent / "fixtures" / "mock_completions.jsonl"
    generated = run_generation(
        config, "seed_train", resume=True, backend_name="mock", mock_fixture=str(fixture)
    )
    assert generated["success"] == 3
    monkeypatch.setattr(
        "pipeline.verify.load_math_verifier",
        lambda: (int, lambda gold, prediction: gold == prediction, "test"),
    )
    verified = run_verification(config, "seed_train")
    assert verified["counts"] == {"correct": 1, "incorrect": 1, "unverifiable": 1}
    pool = build_star_pool(config, "ordinary")
    assert pool["n_trajectories"] == 1
    selection = run_selection(config, "ordinary", "all", 1.0, 42)
    assert selection["n_selected"] == 1
    assert selection["pool_hash"] == pool["pool_hash"]
