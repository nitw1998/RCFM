import numpy as np
import pytest

from scripts.plot_ptbxl_wesad_rmse_record_audit import (
    _per_record_pearson,
    _per_record_rmse,
    _select_extremes,
)


def test_select_extremes_uses_exact_stable_ranks():
    scores = np.asarray([0.4, 0.1, 0.8, 0.2, 0.6, 0.3], dtype=np.float64)
    selected = _select_extremes(scores)
    assert [row["selection"] for row in selected] == ["best", "mid", "worst"]
    assert [row["dataset_row"] for row in selected] == [1, 0, 2]
    assert [row["rank_zero_based"] for row in selected] == [0, 3, 5]
    assert [row["selection_score"] for row in selected] == pytest.approx([0.1, 0.4, 0.8])


def test_select_extremes_breaks_ties_by_dataset_row():
    selected = _select_extremes(np.asarray([0.1, 0.1, 0.2, 0.3]))
    assert selected[0]["dataset_row"] == 0


def test_per_record_metrics_and_shared_mean_score():
    reference = np.zeros((3, 1, 512), dtype=np.float32)
    first = np.stack([np.full((1, 512), value) for value in (0.1, 0.3, 0.5)])
    second = np.stack([np.full((1, 512), value) for value in (0.3, 0.1, 0.7)])
    first_rmse = _per_record_rmse(reference, first)
    second_rmse = _per_record_rmse(reference, second)
    np.testing.assert_allclose(first_rmse, [0.1, 0.3, 0.5], atol=1e-7)
    np.testing.assert_allclose(second_rmse, [0.3, 0.1, 0.7], atol=1e-7)
    np.testing.assert_allclose(np.mean(np.stack([first_rmse, second_rmse]), axis=0), [0.2, 0.2, 0.6], atol=1e-7)

    ramp = np.linspace(-1.0, 1.0, 512, dtype=np.float32)[None, None, :]
    correlations = _per_record_pearson(ramp, -ramp)
    np.testing.assert_allclose(correlations, [-1.0])
