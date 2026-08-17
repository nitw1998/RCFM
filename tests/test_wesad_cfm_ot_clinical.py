import numpy as np

from scripts.evaluate_wesad_cfm_ot_clinical import _five_minute_blocks, _hrv_summary


def test_five_minute_blocks_do_not_cross_subject_or_label_boundaries():
    subjects = np.asarray(["S1"] * 150 + ["S2"] * 80)
    labels = np.asarray([1] * 75 + [2] * 75 + [1] * 80)
    blocks = _five_minute_blocks(subjects, labels)
    assert len(blocks) == 3
    assert [(block[0], block[-1]) for block in blocks] == [(0, 74), (75, 149), (150, 224)]


def test_hrv_summary_censors_window_boundary_differences():
    measurements = [
        {"rr_intervals_ms": [800.0, 810.0, 790.0]},
        {"rr_intervals_ms": [1000.0, 1010.0, 990.0]},
    ]
    summary = _hrv_summary(measurements, np.asarray([0, 1]))
    assert summary is not None
    assert summary["n_rr"] == 6
    assert summary["rmssd_ms"] == np.sqrt((10**2 + 20**2 + 10**2 + 20**2) / 4)
