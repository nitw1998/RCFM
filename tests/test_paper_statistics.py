import numpy as np
import pytest

from src.rcfm.metrics.paper_statistics import (
    ELEVEN_TARGET_LEADS,
    apply_fixed_lag,
    estimate_training_fixed_lag,
    hierarchical_pearson,
    raw_waveform_summary,
    waveform_fd_summary,
)


def test_hierarchical_pearson_uses_fisher_mean_within_equal_groups():
    x = np.stack([np.arange(16), np.arange(16), np.arange(16), np.arange(16)])
    y = np.stack([x[0], x[1], -x[2], -x[3]])
    summary, per_sample, per_group = hierarchical_pearson(
        x[:, None], y[:, None], ["a", "a", "b", "b"]
    )
    np.testing.assert_allclose(per_sample, [1, 1, -1, -1])
    assert per_group["a"] == pytest.approx(1.0, abs=1e-6)
    assert per_group["b"] == pytest.approx(-1.0, abs=1e-6)
    assert summary["group_count"] == 2
    assert summary["group_median_primary"] == pytest.approx(0.0, abs=1e-6)


def test_fixed_11_lead_wfd_is_unweighted_macro_over_full_split():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(20, 11, 16))
    generated = reference + np.arange(11)[None, :, None] / 100.0
    summary = waveform_fd_summary(reference, generated, ELEVEN_TARGET_LEADS)
    assert summary["aggregation"] == "macro_11_target_leads"
    assert list(summary["per_lead"]) == list(ELEVEN_TARGET_LEADS)
    assert summary["value"] == pytest.approx(np.mean(list(summary["per_lead"].values())))
    with pytest.raises(ValueError, match="frozen 11-lead order"):
        waveform_fd_summary(reference, generated, tuple(reversed(ELEVEN_TARGET_LEADS)))


def test_training_fixed_lag_recovers_delay_without_test_inputs():
    source = np.zeros((10, 1, 64), dtype=np.float64)
    source[:, 0, [12, 31, 45]] = [1.0, -0.5, 0.75]
    target = np.roll(source, 3, axis=-1)
    result = estimate_training_fixed_lag(source, target, max_lag_samples=8)
    assert result["lag_samples"] == 3
    assert result["test_target_used_for_selection"] is False

    reference, unshifted, aligned = apply_fixed_lag(target, source, 3, 8)
    assert reference.shape[-1] == 48
    assert not np.array_equal(reference, unshifted)
    np.testing.assert_array_equal(reference, aligned)


def test_raw_summary_keeps_full_window_and_group_primary_pearson():
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(12, 1, 32))
    generated = reference + rng.normal(scale=0.05, size=reference.shape)
    summary, detail = raw_waveform_summary(
        reference, generated, np.repeat(["s1", "s2", "s3"], 4)
    )
    assert summary["support"] == "raw_full_window"
    assert summary["time_samples"] == 32
    assert summary["pearson"]["group_count"] == 3
    assert summary["wfd"]["aggregation"] == "single_target_lead"
    assert len(detail["per_sample_pearson"]) == 12
