import numpy as np

from scripts.analyze_mmecg_fourway import _negative_dominant, _paired_rows


def test_negative_dominant_qc_uses_median_centered_excursions():
    signals = np.asarray([[[0.0, 0.1, 1.0]], [[0.0, -0.1, -1.0]]])
    assert _negative_dominant(signals).tolist() == [False, True]


def test_paired_rows_preserve_left_minus_right_and_win_fraction():
    values = {
        "cfm": np.asarray([1.0, 3.0]),
        "rcfm": np.asarray([2.0, 2.0]),
        "rcfm_ot": np.asarray([3.0, 3.0]),
        "rddm": np.asarray([4.0, 1.0]),
    }
    row = _paired_rows(values)[0]
    assert row["left_model"] == "cfm" and row["right_model"] == "rcfm"
    assert row["mean_difference"] == 0.0
    assert row["left_lower_rmse_fraction"] == 0.5
