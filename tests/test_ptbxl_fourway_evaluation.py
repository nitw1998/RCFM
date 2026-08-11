import json

import numpy as np
import pytest
import torch

from scripts.evaluate_ptbxl_fourway import (
    EXPECTED_DATASET_VERSION,
    EXPECTED_SPLIT_HASH,
    TARGET_INDICES,
    TARGET_LEADS,
    _lag_diagnostic,
    _metric_summary,
    _sample_rddm,
    _validate_dataset_manifest,
)


class _Identity(torch.nn.Module):
    def forward(self, values, conditions, time):
        return values


class _Zero(torch.nn.Module):
    def forward(self, values, conditions, time):
        return torch.zeros_like(values)


class _DummyRDDM:
    n_T = 1
    region_model = _Identity()
    eps_model = _Zero()
    oneover_sqrta = torch.ones(2)
    mab_over_sqrtmab = torch.zeros(2)
    sqrt_beta_t = torch.zeros(2)


def test_rddm_sampler_preserves_explicit_multilead_shape():
    conditions = {"down_conditions": [torch.zeros(3, 2, 8)]}
    output = _sample_rddm(_DummyRDDM(), conditions, conditions, (3, 11, 32))
    assert output.shape == (3, 11, 32)


def test_multilead_metrics_use_macro_lead_fd_definition():
    generator = np.random.default_rng(3)
    target = generator.normal(size=(4, 11, 16)).astype(np.float32)
    generated = target + 0.1
    summary, per_record = _metric_summary(target, generated)
    assert summary["rmse"] == pytest.approx(0.1)
    assert len(summary["per_lead"]) == len(TARGET_LEADS)
    assert summary["waveform_fd_macro_lead"] == pytest.approx(
        np.mean([row["waveform_fd"] for row in summary["per_lead"].values()])
    )
    assert per_record["rmse"].shape == (4,)


def test_lag_search_is_diagnostic_and_does_not_return_corrected_arrays():
    target = np.zeros((2, 11, 64), dtype=np.float32)
    target[:, :, 24] = 1
    generated = np.roll(target, 3, axis=-1)
    result = _lag_diagnostic(target, generated, 8)
    assert result["status"].startswith("diagnostic_only")
    assert result["median_absolute_lag_samples"] == 3
    assert "aligned_predictions" not in result


def test_manifest_requires_official_fold_10(tmp_path):
    payload = {
        "dataset_version": EXPECTED_DATASET_VERSION,
        "split_hash": EXPECTED_SPLIT_HASH,
        "split_method": "official_ptbxl_strat_fold_1_8_train_9_val_10_test",
        "patient_disjoint_verified": True,
        "splits": {"test": {"folds": [10], "records": 2203}},
    }
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _validate_dataset_manifest(path, 2203)["splits"]["test"]["records"] == 2203
    payload["splits"]["test"]["folds"] = [9]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fold-10"):
        _validate_dataset_manifest(path, 2203)


def test_frozen_target_order_excludes_only_lead_ii():
    assert TARGET_INDICES == (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    assert TARGET_LEADS[0:2] == ("I", "III")


def test_condition_channel_axis_is_required_by_generation_contract():
    condition_storage = np.zeros((7, 512), dtype=np.float32)
    conditions = condition_storage[:, None, :]
    assert conditions.shape == (7, 1, 512)


def test_zero_baseline_uses_null_for_undefined_record_correlation():
    target = np.arange(4 * 11 * 16, dtype=np.float32).reshape(4, 11, 16)
    summary, _ = _metric_summary(target, np.zeros_like(target), include_fd=False)
    assert summary["per_record_pearson_mean"] is None
    assert all(row["per_record_pearson_median"] is None for row in summary["per_lead"].values())
