import pytest

from pipeline.common import PipelineError, validate_config


def config(model_id="Qwen/Qwen2.5-3B"):
    return {
        "project": "selection",
        "run_id": "qwen-r0",
        "master_seed": 42,
        "dataset": {
            "repo_id": "openai/gsm8k",
            "config_name": "main",
            "few_shot_indices": [0, 2, 3, 5],
            "validation_size": 300,
            "generation_size": 4000,
        },
        "model": {"id": model_id},
        "sampling": {
            "batch_size": 1,
            "prompt_version": "gsm8k_star_v1",
            "n_per_problem": 1,
            "do_sample": False,
        },
        "pilot": {"n_problems": 32, "selection_seed": 42},
        "star": {
            "training": {"learning_rate": 2e-4, "max_seq_length": 1024, "epochs": 2},
            "evaluation": {"max_new_tokens": 384},
        },
        "paths": {"root": "data/qwen", "run_root": "runs/qwen"},
    }


def test_qwen_base_is_a_valid_single_trajectory_model():
    validate_config(config())


def test_model_id_must_not_be_empty():
    with pytest.raises(PipelineError, match="model.id"):
        validate_config(config(""))


def test_registered_profile_still_rejects_multiple_trajectories():
    value = config()
    value["sampling"]["n_per_problem"] = 2
    with pytest.raises(PipelineError, match="n_per_problem=1"):
        validate_config(value)
