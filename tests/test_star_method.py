from pipeline.evaluate_star import bootstrap_accuracy, summarize_predictions
import json

from pipeline.rationalize import failed_problem_ids, run_rationalization, verify_rationalizations


def test_rationalization_only_targets_problems_without_any_correct_trajectory():
    raw = [
        {"trajectory_id": "a", "problem_id": "p1"},
        {"trajectory_id": "b", "problem_id": "p1"},
        {"trajectory_id": "c", "problem_id": "p2"},
    ]
    verified = [
        {"trajectory_id": "a", "problem_id": "p1", "verification_status": "incorrect"},
        {"trajectory_id": "b", "problem_id": "p1", "verification_status": "correct"},
        {"trajectory_id": "c", "problem_id": "p2", "verification_status": "unverifiable"},
    ]
    assert failed_problem_ids(raw, verified) == {"p2"}


def test_pass_at_one_counts_parse_failures_as_failures_and_ci_is_reproducible():
    rows = [
        {"verification_status": "correct"},
        {"verification_status": "incorrect"},
        {"verification_status": "unverifiable"},
    ]
    result = summarize_predictions(rows, 7)
    assert result["pass_at_1"] == 1 / 3
    assert result["unverifiable_rate"] == 1 / 3
    assert bootstrap_accuracy([True, False, False], seed=7) == bootstrap_accuracy(
        [True, False, False], seed=7
    )


def test_mock_rationalization_generates_only_for_failed_problem(tmp_path, monkeypatch):
    root = tmp_path / "data"
    for directory in ("splits", "prompts", "private", "trajectories", "verification"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    def write(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write(root / "splits/seed_train.jsonl", [
        {"problem_id": "p1", "question": "one"}, {"problem_id": "p2", "question": "two"},
    ])
    write(root / "prompts/few_shot.jsonl", [
        {"problem_id": "demo", "question": "demo?", "answer": "demo work\n#### 1"},
    ])
    write(root / "private/gold_answers.jsonl", [
        {"problem_id": "p1", "gold_answer": "1", "gold_extraction_status": "ok"},
        {"problem_id": "p2", "gold_answer": "2", "gold_extraction_status": "ok"},
    ])
    write(root / "trajectories/seed_train_raw.jsonl", [
        {"trajectory_id": "a", "problem_id": "p1"}, {"trajectory_id": "b", "problem_id": "p2"},
    ])
    write(root / "verification/seed_train_verified.jsonl", [
        {"trajectory_id": "a", "problem_id": "p1", "verification_status": "correct"},
        {"trajectory_id": "b", "problem_id": "p2", "verification_status": "incorrect"},
    ])
    fixture = tmp_path / "fixture.jsonl"
    write(fixture, [{
        "problem_id": "p2", "sample_index": 0,
        "raw_completion": "derive two\n#### 2", "finish_reason": "eos",
    }])
    config = {
        "run_id": "toy", "master_seed": 7,
        "model": {"id": "mock", "revision": "mock"},
        "sampling": {"do_sample": True},
        "star": {"round": 0, "rationalization": {"n_per_problem": 1, "max_new_tokens": 32}},
        "paths": {"root": str(root), "run_root": str(tmp_path / "run")},
    }
    summary = run_rationalization(
        config, resume=True, backend_name="mock", mock_fixture=str(fixture)
    )
    assert summary["success"] == 1
    generated = [json.loads(line) for line in (root / "trajectories/seed_train_rationalized_raw.jsonl").read_text().splitlines()]
    assert generated[0]["problem_id"] == "p2"
    assert generated[0]["used_gold_hint"] is True
    assert generated[0]["prompt_version"] == "gsm8k_star_v1"
    assert generated[0]["generation_prompt_version"] == "star_gold_answer_v1"
    monkeypatch.setattr(
        "pipeline.rationalize.load_math_verifier",
        lambda: (int, lambda gold, prediction: gold == prediction, "test"),
    )
    report = verify_rationalizations(config)
    assert report["counts"] == {"correct": 1}
