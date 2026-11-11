import json

from pipeline.star_pool import build_star_pool, pool_content_hash, run_selection, select_rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def config(tmp_path):
    return {"paths": {"root": str(tmp_path / "data")}}


def raw(trajectory_id, problem_id, reasoning, final="#### 1"):
    return {
        "trajectory_id": trajectory_id, "problem_id": problem_id,
        "reasoning_text": reasoning, "final_text": final,
        "raw_completion": f"{reasoning}\n{final}",
        "completion_token_count": 3, "prompt_version": "gsm8k_star_v1",
    }


def verified(trajectory_id, problem_id, status="correct"):
    return {
        "trajectory_id": trajectory_id, "problem_id": problem_id,
        "verification_status": status, "is_correct": status == "correct",
    }


def test_fixed_pool_and_random_selection_are_reproducible(tmp_path):
    root = tmp_path / "data"
    raws = [raw("t1", "p1", "same"), raw("t2", "p2", "same"), raw("t3", "p3", "different")]
    checks = [verified("t1", "p1"), verified("t2", "p2"), verified("t3", "p3")]
    write_jsonl(root / "trajectories/seed_train_raw.jsonl", raws)
    write_jsonl(root / "verification/seed_train_verified.jsonl", checks)
    manifest = build_star_pool(config(tmp_path), "ordinary")
    assert manifest["n_trajectories"] == 3
    first = run_selection(config(tmp_path), "ordinary", "random", 2 / 3, 42)
    second = run_selection(config(tmp_path), "ordinary", "random", 2 / 3, 42)
    assert first["selection_sha256"] == second["selection_sha256"]
    assert first["pool_hash"] == manifest["pool_hash"]


def test_exact_dedup_never_silently_backfills():
    rows = [raw("a", "p1", "same"), raw("b", "p2", "same"), raw("c", "p3", "other")]
    selected, detail = select_rows(rows, "exact_dedup", 1.0, 0)
    assert [row["trajectory_id"] for row in selected] == ["a", "c"]
    assert detail["shortfall"] == 1
    assert pool_content_hash(rows) == pool_content_hash(list(reversed(rows)))


def test_with_rationalization_is_a_separate_pool(tmp_path):
    root = tmp_path / "data"
    write_jsonl(root / "trajectories/seed_train_raw.jsonl", [raw("ordinary", "p1", "ordinary")])
    write_jsonl(root / "verification/seed_train_verified.jsonl", [verified("ordinary", "p1")])
    write_jsonl(root / "trajectories/seed_train_rationalized_raw.jsonl", [raw("rat", "p2", "derived")])
    write_jsonl(root / "verification/seed_train_rationalized_verified.jsonl", [verified("rat", "p2")])
    ordinary = build_star_pool(config(tmp_path), "ordinary")
    with_rat = build_star_pool(config(tmp_path), "with_rationalization")
    assert ordinary["n_trajectories"] == 1
    assert with_rat["n_trajectories"] == 2
    assert with_rat["origin_counts"] == {"ordinary": 1, "rationalization": 1}
    assert with_rat["pool_hash"] != ordinary["pool_hash"]
