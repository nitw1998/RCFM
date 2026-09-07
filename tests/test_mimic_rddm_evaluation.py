import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.evaluate_mimic_rddm_predictions import (
    EXPECTED_SPLIT_HASH,
    EXPECTED_UPSTREAM_COMMIT,
    _generate_batches,
    _select_examples,
    _validate_checkpoint,
)


def _checkpoint():
    return {
        "schema_version": 1,
        "kind": "independent_rddm_reproduction",
        "epoch": 500,
        "global_step": 33000,
        "rddm_state": {},
        "condition_1_state": {},
        "condition_2_state": {},
        "config": {
            "task": "ppg2ecg",
            "datasets": ["MIMIC-AFib"],
            "dataset_version": "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1",
            "split_hash": EXPECTED_SPLIT_HASH,
            "normalization_id": "rddm_window_minmax_neg1_1_v1",
            "alignment_id": "paired_array_row_rddm_contract_zero_ppg_qc_v1",
            "window_size": 4,
            "expected_train_windows": 8400,
            "expected_test_windows": 1800,
            "nT": 10,
            "attention_heads": 8,
            "seed": 31,
            "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
            "beta_start": 1e-4,
            "beta_end": 0.2,
        },
        "normalization": {
            "method": "rddm_window_minmax_neg1_1",
            "stats_scope": "per_window_per_modality",
            "feature_range": [-1.0, 1.0],
            "generated_inverse_policy": "normalized_domain_only",
        },
        "provenance": {"upstream_commit": EXPECTED_UPSTREAM_COMMIT},
    }


def test_checkpoint_contract_accepts_only_frozen_rddm_endpoint():
    checkpoint = _checkpoint()
    _validate_checkpoint(checkpoint)

    checkpoint["config"]["seed"] = 32
    _validate_checkpoint(checkpoint, expected_training_seed=32)
    with pytest.raises(ValueError, match="seed"):
        _validate_checkpoint(checkpoint, expected_training_seed=33)

    checkpoint["config"]["nT"] = 50
    with pytest.raises(ValueError, match="nT"):
        _validate_checkpoint(checkpoint)


class IdentityCondition(nn.Module):
    def forward(self, values):
        return {"values": values}


class RandomSampler(nn.Module):
    def forward(self, cond1, cond2, mode, window_size, output_channels):
        assert mode == "sample"
        assert window_size == 512
        shape = (len(cond1["values"]), output_channels, window_size)
        return torch.randn(shape, device=cond1["values"].device)


def test_batch_generation_replays_recorded_seed_schedule():
    conditions = np.zeros((5, 1, 512), dtype=np.float32)
    args = (RandomSampler(), IdentityCondition(), IdentityCondition(), conditions, 3, 2025, torch.device("cpu"))

    first, first_indices, first_seeds = _generate_batches(*args)
    second, second_indices, second_seeds = _generate_batches(*args)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_indices, [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(first_indices, second_indices)
    np.testing.assert_array_equal(first_seeds, [2025, 2026])
    np.testing.assert_array_equal(first_seeds, second_seeds)


def test_batch_generation_supports_multichannel_rddm_output():
    conditions = np.zeros((2, 1, 512), dtype=np.float32)
    generated, _, _ = _generate_batches(
        RandomSampler(), IdentityCondition(), IdentityCondition(), conditions,
        2, 2025, torch.device("cpu"), output_channels=11,
    )
    assert generated.shape == (2, 11, 512)


def test_example_selection_is_stable():
    assert _select_examples(np.array([0.3, 0.2, 0.5, 0.1])) == {
        "best": 3,
        "median": 0,
        "worst": 2,
    }
