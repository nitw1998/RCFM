from __future__ import annotations

import numpy as np

from scripts.audit_crossmodal_ecgdeli_wfdb_bland_altman import summarize_fiducials


def test_summary_applies_order_and_physiology_qc() -> None:
    signal = np.zeros(480)
    signal[100] = 0.2
    signal[140] = 1.0
    signal[180] = 0.3
    rows = [{
        "p_onset": 95, "p_peak": 100, "p_offset": 108, "qrs_onset": 120,
        "r_peak": 125, "qrs_offset": 135, "t_peak": 180, "t_offset": 210,
    }, {
        "p_onset": 218, "p_peak": 220, "p_offset": 225, "qrs_onset": 240,
        "r_peak": 253, "qrs_offset": 263, "t_peak": 310, "t_offset": 340,
    }]
    result = summarize_fiducials(rows, signal, 128.0, afib=False)
    assert result["qrs_order_valid_fraction"] == 1.0
    assert result["qc_qrs_ms_beats"] == 2
    assert result["qc_pr_ms_beats"] == 2
    assert result["qc_rr_ms_beats"] == 1


def test_afib_excludes_p_and_pr_but_keeps_qrs() -> None:
    signal = np.zeros(480)
    rows = [{
        "p_onset": 95, "p_peak": 100, "p_offset": 108, "qrs_onset": 120,
        "r_peak": 125, "qrs_offset": 135, "t_peak": 180, "t_offset": 210,
    }]
    result = summarize_fiducials(rows, signal, 128.0, afib=True)
    assert result["raw_pr_ms"] is None
    assert result["qc_p_amplitude"] is None
    assert result["qc_qrs_ms"] is not None
