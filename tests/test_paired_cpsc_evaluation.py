from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_cpsc_zscore_paired import (
    _clinical_model_comparison,
    _fixed_noise,
    _p_wave_applicability,
    _validate_contracts,
    _waveform_metrics,
)


def _contract(kind: str, family: str, region_weight: float):
    config = {
        "task": "ecg2ecg",
        "datasets": ["CPSC2018"],
        "dataset_version": "synthetic-v1",
        "split_hash": "split-123",
        "normalization_id": "record_zscore_v1",
        "condition_unit": "unknown_source_unit",
        "target_unit": "unknown_source_unit",
        "alignment_id": "aligned",
        "condition_lead": "lead-2",
        "target_lead": "lead-10",
        "condition_lead_index": 2,
        "target_lead_index": 10,
        "window_size": 4,
        "attention_heads": 8,
        "flow_matcher": "conditional",
        "sigma": 0.0,
        "use_minibatch_ot": False,
        "region_weight": region_weight,
        "model_family": family,
    }
    return {
        "kind": kind,
        "config": config,
        "normalization": {
            "method": "record_zscore",
            "normalization_id": "record_zscore_v1",
        },
        "output_spec": {"channels": 1, "length": 512},
    }


def test_paired_contract_accepts_only_intended_region_difference():
    cfm = _contract("canonical_multistep_cfm", "CFM", 0.0)
    rcfm = _contract("canonical_multistep_rcfm", "RCFM", 0.01)
    _validate_contracts(cfm, rcfm)

    rcfm["config"]["split_hash"] = "other-split"
    with pytest.raises(ValueError, match="split_hash"):
        _validate_contracts(cfm, rcfm)


def test_historical_paired_clinical_evaluator_rejects_multilead_checkpoints():
    cfm = _contract("canonical_multistep_cfm", "CFM", 0.0)
    rcfm = _contract("canonical_multistep_rcfm", "RCFM", 0.01)
    cfm["output_spec"]["channels"] = 11
    rcfm["output_spec"]["channels"] = 11

    with pytest.raises(ValueError, match="only single-target-lead"):
        _validate_contracts(cfm, rcfm)


def test_fixed_noise_is_exactly_reproducible():
    first = _fixed_noise(7, 1, 16, seed=2025)
    second = _fixed_noise(7, 1, 16, seed=2025)
    different = _fixed_noise(7, 1, 16, seed=2026)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_waveform_metrics_report_recordwise_and_pointwise_agreement():
    reference = np.arange(24, dtype=np.float32).reshape(3, 1, 8)
    generated = reference + 0.5
    summary, records = _waveform_metrics(reference, generated)

    assert summary["rmse"] == pytest.approx(0.5)
    assert summary["mae"] == pytest.approx(0.5)
    assert summary["pointwise_correlation_descriptive_only"]["r"] == pytest.approx(1.0)
    assert summary["pointwise_bland_altman_descriptive_only"]["bias"] == pytest.approx(0.5)
    np.testing.assert_allclose(records["pearson_r"], 1.0)


def test_af_label_policy_blocks_p_wave_metrics(tmp_path: Path):
    labels = np.zeros((4, 3), dtype=np.uint8)
    labels[[1, 3], 1] = 1
    np.save(tmp_path / "labels_test.npy", labels)
    (tmp_path / "dataset_manifest.json").write_text(
        '{"label_names":{"1":"normal","2":"atrial_fibrillation","3":"other"}}',
        encoding="utf-8",
    )

    applicable, policy = _p_wave_applicability(tmp_path, count=4)

    np.testing.assert_array_equal(applicable, [True, False, True, False])
    assert policy["not_applicable_af_records"] == 2


def test_clinical_model_comparison_uses_three_way_common_records():
    cfm = {
        "unit_summaries": {
            "real": {"a": {"rr_ms": 1000.0}, "b": {"rr_ms": 900.0}},
            "generated": {"a": {"rr_ms": 1010.0}, "b": {"rr_ms": 920.0}},
        }
    }
    rcfm = {
        "unit_summaries": {
            "real": {"a": {"rr_ms": 1000.0}, "b": {"rr_ms": 900.0}},
            "generated": {"a": {"rr_ms": 1005.0}, "b": {"rr_ms": 910.0}},
        }
    }

    comparison = _clinical_model_comparison(cfm, rcfm)["parameters"]["rr_ms"]

    assert comparison["n"] == 2
    assert comparison["cfm_mean_absolute_error"] == pytest.approx(15.0)
    assert comparison["rcfm_mean_absolute_error"] == pytest.approx(7.5)
    assert comparison["fraction_absolute_error_favoring_rcfm"] == 1.0
