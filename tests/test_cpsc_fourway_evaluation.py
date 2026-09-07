import json

import pytest

from scripts.evaluate_cpsc_fourway import (
    EXPECTED_SPLIT_HASH,
    _validate_rddm_checkpoint,
    _validate_manifest,
)


def test_cpsc_manifest_is_bound_by_schema_split_and_validation_count(tmp_path):
    path = tmp_path / "dataset_manifest.json"
    payload = {
        "schema_version": 2,
        "split_hash": EXPECTED_SPLIT_HASH,
        "splits": {"val": {"records": 686}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _validate_manifest(path, 686)["splits"]["val"]["records"] == 686
    payload["splits"]["val"]["records"] = 685
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="record count"):
        _validate_manifest(path, 686)


def test_cpsc_rddm_validator_accepts_explicit_training_seed():
    checkpoint = {
        "kind": "independent_rddm_reproduction",
        "epoch": 500,
        "global_step": 21500,
        "config": {
            "task": "ecg2ecg",
            "datasets": ["CPSC2018"],
            "dataset_version": "cpsc2018-source-derived-all12lead-qc-v3-record-minmax-neg1-1",
            "split_hash": EXPECTED_SPLIT_HASH,
            "normalization_id": "record_minmax_neg1_1_v1",
            "alignment_id": "same_record_simultaneous_channels_first_4s_lead_II_to_other_11_v3",
            "condition_lead_index": 1,
            "target_lead_indices": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "target_channels": 11,
            "window_size": 4,
            "nT": 10,
            "attention_heads": 8,
            "seed": 32,
            "reproduction_label": "RDDM-ECG (adapted)",
        },
        "provenance": {"upstream_commit": "7d5348843c3985c211a23ae5105a2d9497d5156a"},
    }
    _validate_rddm_checkpoint(checkpoint, expected_training_seed=32)
    with pytest.raises(ValueError, match="contract"):
        _validate_rddm_checkpoint(checkpoint, expected_training_seed=31)
