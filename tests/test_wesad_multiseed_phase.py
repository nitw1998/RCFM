import numpy as np
import pytest

from scripts.analyze_wesad_multiseed_phase import _metric_values
from scripts.evaluate_wesad_multiseed_checkpoint import (
    EXPECTED_ALIGNMENT,
    EXPECTED_DATASET_VERSION,
    EXPECTED_SPLIT_HASH,
    _validate_flow_contracts,
    _validate_rddm_checkpoint,
)


def _flow(kind, region, ot, seed=32):
    config = {
        "task": "ppg2ecg", "datasets": ["WESAD"], "dataset_version": EXPECTED_DATASET_VERSION,
        "split_hash": EXPECTED_SPLIT_HASH, "normalization_id": "window_minmax_neg1_1_v1",
        "alignment_id": EXPECTED_ALIGNMENT, "window_size": 4, "attention_heads": 8,
        "flow_matcher": "conditional", "sigma": 0.0, "seed": seed,
        "region_weight": region, "use_minibatch_ot": ot, "ot_method": "exact",
    }
    return {"kind": kind, "epoch": 500, "global_step": 68500, "config": config,
            "normalization": {"normalization_id": "window_minmax_neg1_1_v1"},
            "output_spec": {"channels": 1, "length": 512}}


def test_wesad_multiseed_contracts_and_metrics():
    contracts = {"cfm": _flow("canonical_multistep_cfm", 0.0, False),
                 "rcfm": _flow("canonical_multistep_rcfm", 0.01, False),
                 "rcfm_ot": _flow("canonical_multistep_rcfm", 0.01, True)}
    _validate_flow_contracts(contracts, 32)
    base = np.linspace(-1, 1, 480, dtype=np.float32)[None, None, :]
    metrics = _metric_values(np.repeat(base, 4, axis=0), np.repeat(base, 4, axis=0))
    assert metrics["rmse"] == 0 and metrics["waveform_fd"] == 0
    assert np.isclose(metrics["pearson_window_median"], 1)


def test_wesad_rddm_contract_rejects_wrong_seed():
    checkpoint = {"schema_version": 1, "kind": "independent_rddm_reproduction",
                  "epoch": 500, "global_step": 68500,
                  "config": {"task": "ppg2ecg", "datasets": ["WESAD"],
                             "dataset_version": EXPECTED_DATASET_VERSION, "split_hash": EXPECTED_SPLIT_HASH,
                             "normalization_id": "window_minmax_neg1_1_v1", "alignment_id": EXPECTED_ALIGNMENT,
                             "heldout_split": "test", "expected_train_windows": 17494,
                             "expected_test_windows": 4213, "window_size": 4, "target_channels": 1,
                             "nT": 10, "seed": 33,
                             "reproduction_label": "RDDM-PPG (matched-protocol reproduction)"},
                  "provenance": {"upstream_commit": "7d5348843c3985c211a23ae5105a2d9497d5156a"}}
    _validate_rddm_checkpoint(checkpoint, 33)
    with pytest.raises(ValueError, match="WESAD"):
        _validate_rddm_checkpoint(checkpoint, 32)
