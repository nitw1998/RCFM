import numpy as np

from scripts.analyze_mmecg_multiseed_phase import _metric_values
from scripts.evaluate_mmecg_fourway import (
    _validate_flow_contracts,
    _validate_rddm_checkpoint,
)


def test_phase_metric_values_are_exact_for_identical_waveforms():
    base = np.linspace(-1, 1, 480, dtype=np.float32)
    values = np.stack([np.sin((index + 1) * base) for index in range(8)])[:, None, :]
    metrics = _metric_values(values, values.copy())
    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    assert metrics["waveform_fd"] == 0
    assert np.isclose(metrics["pearson_window_median"], 1)


def test_mmecg_validators_accept_explicit_training_seed():
    base_config = {
        "task": "rcg2ecg", "datasets": ["mmECG"],
        "dataset_version": "mmecg-public-20221108-subject-split-window-minmax-v1",
        "split_hash": "e26fc81121cfd3b0e457608e37a7aa496ac7c48a0e0e7583a465bd069cc9da9f",
        "normalization_id": "window_minmax_neg1_1_v1",
        "alignment_id": "same_record_same_window_no_additional_phase_correction_subject_split_v1",
        "window_size": 4, "attention_heads": 8, "flow_matcher": "exact_ot",
        "sigma": 0.0, "seed": 32,
    }
    contracts = {}
    for model, kind, region, ot in (
        ("cfm", "canonical_multistep_cfm", 0.0, False),
        ("rcfm", "canonical_multistep_rcfm", 1.0, False),
        ("rcfm_ot", "canonical_multistep_rcfm", 1.0, True),
    ):
        config = {**base_config, "region_weight": region, "use_minibatch_ot": ot}
        if ot:
            config["ot_method"] = "exact"
        contracts[model] = {
            "kind": kind, "epoch": 500, "global_step": 37500, "config": config,
            "normalization": {"method": "window_minmax"},
            "output_spec": {"channels": 1, "length": 512},
        }
    _validate_flow_contracts(contracts, expected_training_seed=32)
    rddm = {
        "schema_version": 1, "kind": "independent_rddm_reproduction",
        "epoch": 500, "global_step": 37500,
        "config": {
            **base_config, "heldout_split": "test", "expected_train_windows": 9590,
            "expected_test_windows": 2877, "target_channels": 1, "nT": 10,
            "reproduction_label": "RDDM-RCG (adapted)",
        },
        "normalization": {},
        "provenance": {"upstream_commit": "7d5348843c3985c211a23ae5105a2d9497d5156a"},
    }
    _validate_rddm_checkpoint(rddm, expected_training_seed=32)
