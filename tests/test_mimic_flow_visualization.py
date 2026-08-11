import numpy as np
import pytest
import torch

from scripts.visualize_mimic_flow_predictions import (
    _baseline_summary,
    _lag_diagnostic,
    _select_examples,
    _validate_contracts,
    _validation_noise,
)


def _contract(kind, family, region_weight, use_ot, sampling="multinomial"):
    return {
        "kind": kind,
        "epoch": 500,
        "config": {
            "task": "ppg2ecg",
            "datasets": ["MIMIC-AFib"],
            "dataset_version": "mimic-v1",
            "split_hash": "split-123",
            "normalization_id": "rddm_window_minmax_neg1_1_v1",
            "condition_unit": "normalized",
            "target_unit": "normalized",
            "alignment_id": "paired",
            "condition_lead": "PPG",
            "target_lead": "ECG",
            "condition_lead_index": None,
            "target_lead_index": None,
            "window_size": 4,
            "attention_heads": 8,
            "flow_matcher": "conditional",
            "sigma": 0.0,
            "seed": 31,
            "model_family": family,
            "region_weight": region_weight,
            "use_minibatch_ot": use_ot,
            "ot_method": "exact",
            "ot_sampling_strategy": sampling,
        },
        "normalization": {
            "method": "rddm_window_minmax_neg1_1",
            "normalization_id": "rddm_window_minmax_neg1_1_v1",
        },
        "output_spec": {
            "channels": 1,
            "length": 512,
            "sampling_rate_hz": 128,
            "target_lead": "ECG",
        },
    }


def _contracts():
    return {
        "cfm": _contract("canonical_multistep_cfm", "CFM", 0.0, False),
        "rcfm": _contract("canonical_multistep_rcfm", "RCFM", 0.01, False),
        "rcfm_ot": _contract(
            "canonical_multistep_rcfm", "RCFM", 0.01, True, "assignment"
        ),
    }


def test_contract_validation_accepts_only_matched_mimic_factors():
    contracts = _contracts()
    contracts["cfm"]["output_spec"].update(
        {"target_leads": ["ECG"], "target_lead_indices": None}
    )
    _validate_contracts(contracts)

    contracts["rcfm_ot"]["config"]["split_hash"] = "wrong"
    with pytest.raises(ValueError, match="split_hash"):
        _validate_contracts(contracts)


def test_validation_noise_matches_independently_seeded_validation_batches():
    actual = _validation_noise(5, 1, 4, batch_size=3, seed=2025)
    expected = []
    for batch_index, count in enumerate((3, 2)):
        generator = torch.Generator(device="cpu").manual_seed(2025 + batch_index)
        expected.append(torch.randn((count, 1, 4), generator=generator).numpy())

    np.testing.assert_array_equal(actual, np.concatenate(expected))


def test_example_selection_uses_shared_error_and_is_stable():
    values = {
        "cfm": np.array([0.3, 0.2, 0.5, 0.1]),
        "rcfm": np.array([0.3, 0.2, 0.4, 0.1]),
        "rcfm_ot": np.array([0.3, 0.2, 0.6, 0.1]),
    }

    assert _select_examples(values) == {"best": 3, "median": 0, "worst": 2}


def test_lag_diagnostic_recovers_delay_applied_to_early_prediction():
    reference = np.array([[[0.0, 0.0, 1.0, -0.5, 0.2, 0.0, 0.0, 0.0]]])
    generated = np.array([[[1.0, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]])

    summary, per_record = _lag_diagnostic(reference, generated, 3, sampling_rate=100)

    assert per_record["best_lag_samples"].tolist() == [2]
    assert per_record["lag_adjusted_pearson_r"][0] == pytest.approx(1.0)
    assert summary["shift_definition"] == "positive values delay the generated waveform"


def test_zero_baseline_serializes_undefined_correlation_as_none():
    reference = np.arange(24, dtype=np.float64).reshape(3, 1, 8)
    summary = _baseline_summary(reference, np.zeros_like(reference))

    assert summary["per_record_pearson"]["usable_records"] == 0
    assert summary["per_record_pearson"]["mean"] is None
    assert summary["per_record_pearson"]["median"] is None
