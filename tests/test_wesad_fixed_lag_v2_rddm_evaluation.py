import copy

import pytest

from scripts.evaluate_wesad_fixed_lag_v2 import (
    EXPECTED_ALIGNMENT,
    EXPECTED_SPLIT,
    EXPECTED_VERSION,
)
from scripts.evaluate_wesad_fixed_lag_v2_rddm import (
    UPSTREAM_COMMIT,
    validate_rddm_checkpoint,
)


def _checkpoint():
    return {
        "schema_version": 1,
        "kind": "independent_rddm_reproduction",
        "epoch": 500,
        "global_step": 68500,
        "config": {
            "task": "ppg2ecg",
            "datasets": ["WESAD"],
            "dataset_version": EXPECTED_VERSION,
            "split_hash": EXPECTED_SPLIT,
            "normalization_id": "window_minmax_neg1_1_v1",
            "alignment_id": EXPECTED_ALIGNMENT,
            "heldout_split": "test",
            "expected_train_windows": 17494,
            "expected_test_windows": 4213,
            "window_size": 4,
            "target_channels": 1,
            "attention_heads": 8,
            "nT": 10,
            "seed": 31,
            "reproduction_label": "RDDM-PPG (matched-protocol reproduction)",
            "beta_start": 1e-4,
            "beta_end": 0.2,
        },
        "provenance": {"upstream_commit": UPSTREAM_COMMIT},
    }


def test_accepts_completed_fixed_lag_v2_rddm_endpoint():
    validate_rddm_checkpoint(_checkpoint())


def test_rejects_v1_alignment_and_wrong_endpoint():
    bad = copy.deepcopy(_checkpoint())
    bad["config"]["alignment_id"] = "native_common_start_same_window_no_delay_correction_subject_fold1_v1"
    with pytest.raises(ValueError, match="alignment_id"):
        validate_rddm_checkpoint(bad)
    bad = copy.deepcopy(_checkpoint())
    bad["epoch"] = 499
    with pytest.raises(ValueError, match="contract"):
        validate_rddm_checkpoint(bad)
