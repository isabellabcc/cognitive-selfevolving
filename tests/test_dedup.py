from pipeline.build_pool import deduplicate_problem


def row(problem_id, trajectory_id, reasoning, final="boxed 1"):
    return {
        "problem_id": problem_id, "trajectory_id": trajectory_id,
        "reasoning_text": reasoning, "final_text": final,
    }


def test_same_problem_exact_duplicate_uses_deterministic_representative():
    rows = [row("p", "z", "same"), row("p", "a", "same"), row("p", "b", "different method")]
    representatives, audit = deduplicate_problem(rows, ngram_size=2, threshold=.99)
    assert "a" in representatives and "z" not in representatives
    exact = next(item for item in audit if item["duplicate_type"] == "exact")
    assert exact["representative_trajectory_id"] == "a"
    assert exact["removed_candidate_ids"] == ["z"]


def test_near_duplicate_audit_keeps_reversible_member_list():
    rows = [
        row("p", "a", "one two three four five six"),
        row("p", "b", "one two three four five six seven"),
    ]
    representatives, audit = deduplicate_problem(rows, ngram_size=2, threshold=.65)
    assert representatives == ["a"]
    assert audit[-1]["member_trajectory_ids"] == ["a", "b"]


def test_dedup_is_scoped_by_caller_to_one_problem():
    left, _ = deduplicate_problem([row("p1", "p1:a", "same")], ngram_size=2, threshold=.9)
    right, _ = deduplicate_problem([row("p2", "p2:a", "same")], ngram_size=2, threshold=.9)
    assert left == ["p1:a"] and right == ["p2:a"]
