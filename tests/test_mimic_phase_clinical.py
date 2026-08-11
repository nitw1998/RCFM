import numpy as np

from scripts.evaluate_mimic_phase_clinical import (
    _agreement_record,
    _parameter_pairs,
    _parameter_triplets,
    _record_measurement,
    _waveform_summary,
)


def test_parameter_pairs_require_same_record_measurements():
    real = [
        {"summary": {"rr_ms": 900.0}},
        {"summary": {"rr_ms": 1000.0}},
        {"summary": {}},
    ]
    generated = [
        {"summary": {"rr_ms": 910.0}},
        {"summary": {}},
        {"summary": {"rr_ms": 800.0}},
    ]
    rows, reference, prediction = _parameter_pairs(real, generated, "rr_ms")
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_allclose(reference, [900.0])
    np.testing.assert_allclose(prediction, [910.0])


def test_agreement_record_reports_generated_minus_reference():
    result = _agreement_record(
        "rcfm_ot",
        "oracle_aligned",
        "qrs_ms",
        np.array([80.0, 90.0, 100.0]),
        np.array([85.0, 95.0, 105.0]),
    )
    assert result["n"] == 3
    assert result["unit"] == "ms"
    assert result["bland_altman_bias"] == 5.0
    assert result["inference_status"] == "descriptive_only_no_subject_ids"


def test_parameter_triplets_freeze_same_rows_before_and_after_alignment():
    real = [
        {"summary": {"rr_ms": 900.0}},
        {"summary": {"rr_ms": 1000.0}},
        {"summary": {"rr_ms": 1100.0}},
    ]
    unshifted = [
        {"summary": {"rr_ms": 910.0}},
        {"summary": {"rr_ms": 990.0}},
        {"summary": {}},
    ]
    aligned = [
        {"summary": {"rr_ms": 905.0}},
        {"summary": {}},
        {"summary": {"rr_ms": 1090.0}},
    ]
    rows, reference, before, after = _parameter_triplets(
        real, unshifted, aligned, "rr_ms"
    )
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_allclose(reference, [900.0])
    np.testing.assert_allclose(before, [910.0])
    np.testing.assert_allclose(after, [905.0])


def test_waveform_summary_detects_phase_match_and_bland_altman_bias():
    target = np.tile(np.linspace(-1.0, 1.0, 32), (3, 1, 1)).astype(np.float32)
    generated = target + 0.25
    result = _waveform_summary(target, generated)
    np.testing.assert_allclose(result["per_record_pearson_median"], 1.0)
    np.testing.assert_allclose(result["pointwise_bland_altman"]["bias"], 0.25)
    assert result["pointwise_inference_status"] == "descriptive_only_autocorrelated_samples"


def test_record_measurement_keeps_hrv_blocked_for_short_noncontinuous_window():
    sampling_rate = 128
    time = np.arange(480) / sampling_rate
    signal = 0.05 * np.sin(2 * np.pi * time)
    for peak in (64, 192, 320, 448):
        signal[peak - 1 : peak + 2] += np.array([0.4, 1.0, 0.4])
    result = _record_measurement(signal, sampling_rate)
    assert result["success"]
    assert result["summary"]["rr_ms"] == 1000.0
    assert result["summary"]["heart_rate_bpm"] == 60.0
    assert result["hrv_status"] == "blocked_non_continuous"
