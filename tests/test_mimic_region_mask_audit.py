import numpy as np

from scripts.audit_mimic_region_masks import (
    _fiducial_masks,
    _refine_peaks_to_local_extrema,
    _soft_alignment_metrics,
)
from src.rcfm.metrics.clinical import ECGFiducials


def test_peak_refinement_uses_local_morphological_extremum_for_both_polarities():
    signal = np.zeros(128, dtype=np.float32)
    signal[43] = 2.0
    signal[91] = -3.0
    refined = _refine_peaks_to_local_extrema(signal, np.array([48, 87]), 128, search_ms=50)
    np.testing.assert_array_equal(refined, [43, 91])


def test_fiducial_masks_keep_p_qrs_t_and_qt_on_the_same_sample_grid():
    fiducials = ECGFiducials(
        r_peaks=np.array([50]),
        p_onsets=np.array([30]), p_peaks=np.array([35]), p_offsets=np.array([40]),
        qrs_onsets=np.array([46]), qrs_offsets=np.array([54]),
        t_peaks=np.array([75]), t_offsets=np.array([90]),
    )
    masks = _fiducial_masks(fiducials, 128)
    assert masks["p"][30] == 1 and masks["p"][40] == 1
    assert masks["qrs"][46] == 1 and masks["qrs"][54] == 1
    assert masks["qt"][46] == 1 and masks["qt"][90] == 1
    assert masks["morphology"][50] == 1


def test_alignment_metric_recovers_known_sample_lag():
    reference = np.zeros(128, dtype=np.float32)
    reference[50:60] = 1.0
    delayed = np.zeros(128, dtype=np.float32)
    delayed[57:67] = 1.0
    metrics = _soft_alignment_metrics(delayed, reference, max_lag=16)
    assert metrics["best_lag_samples"] == -7
    assert metrics["best_lag_correlation"] > 0.99

    degenerate = _soft_alignment_metrics(np.zeros(128), reference, max_lag=16)
    assert degenerate["mask_degenerate"] == 1.0
    assert degenerate["best_lag_samples"] == 0
    assert degenerate["best_lag_correlation"] == 0.0
