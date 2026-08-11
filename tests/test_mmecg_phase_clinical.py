import numpy as np

from scripts.evaluate_mmecg_phase_clinical import (
    _configure_ieee_fonts,
    _record_measurement,
    _subject_agreement,
)


def test_subject_agreement_aggregates_paired_windows_before_comparison():
    subjects = np.asarray(["S1", "S1", "S2", "S2", "S3", "S3"])
    rows = np.arange(6)
    reference = np.asarray([1.0, 3.0, 2.0, 4.0, 4.0, 6.0])
    generated = reference + 1.0
    summary, detail = _subject_agreement(
        subjects, rows, reference, generated, "cfm", "unshifted", "rr_ms"
    )
    assert summary["n_subjects"] == 3
    assert summary["n_windows_contributing"] == 6
    assert summary["bland_altman_bias"] == 1.0
    assert [row["real_mean"] for row in detail] == [2.0, 3.0, 5.0]


def test_mmecg_short_signal_measurement_uses_center_fiducials_without_crashing():
    sampling_rate = 128
    time = np.arange(480) / sampling_rate
    signal = 0.04 * np.sin(2 * np.pi * 1.2 * time)
    for center in np.arange(0.4, 3.6, 1 / 1.2):
        signal += np.exp(-0.5 * ((time - center) / 0.018) ** 2)
    result = _record_measurement(signal.astype(np.float32), sampling_rate)
    assert "success" in result
    assert "summary" in result


def test_ieee_font_configuration_embeds_truetype():
    import matplotlib.pyplot as plt

    _configure_ieee_fonts()
    assert plt.rcParams["font.family"] == ["serif"]
    assert plt.rcParams["pdf.fonttype"] == 42
