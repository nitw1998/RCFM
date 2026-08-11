import numpy as np

from src.rcfm.metrics.bland_altman import bland_altman, paired_correlation
from src.rcfm.metrics.clinical import (
    ECGFiducials,
    compute_hrv,
    delineate_ecg,
    measure_ecg_parameters,
)
from scripts.evaluate_clinical import _paired_records


def _synthetic_signal_and_fiducials():
    signal = np.zeros(3000, dtype=np.float64)
    r_peaks = np.array([500, 1500, 2500])
    p_peaks = r_peaks - 100
    t_peaks = r_peaks + 200
    qrs_offsets = r_peaks + 40
    signal[p_peaks] = 0.2
    signal[r_peaks] = 1.0
    signal[t_peaks] = 0.3
    signal[qrs_offsets + 60] = 0.1
    fiducials = ECGFiducials(
        r_peaks=r_peaks,
        p_onsets=r_peaks - 150,
        p_peaks=p_peaks,
        p_offsets=r_peaks - 50,
        qrs_onsets=r_peaks - 20,
        qrs_offsets=qrs_offsets,
        t_peaks=t_peaks,
        t_offsets=r_peaks + 350,
    )
    return signal, fiducials


def test_known_intervals_qtc_amplitudes_and_st():
    signal, fiducials = _synthetic_signal_and_fiducials()
    result = measure_ecg_parameters(
        signal,
        sampling_rate=1000,
        fiducials=fiducials,
        amplitude_unit="mV",
        inverse_transformed=True,
        qtc_formula="fridericia",
        st_offset_ms=60,
    )
    parameters = result["parameters"]

    np.testing.assert_allclose(parameters["rr_ms"], [1000, 1000])
    np.testing.assert_allclose(parameters["pr_ms"], [130, 130, 130])
    np.testing.assert_allclose(parameters["qrs_ms"], [60, 60, 60])
    np.testing.assert_allclose(parameters["qt_ms"], [370, 370, 370])
    np.testing.assert_allclose(parameters["qtc_ms"], [370, 370])
    np.testing.assert_allclose(parameters["p_amplitude"], [0.2, 0.2, 0.2])
    np.testing.assert_allclose(parameters["r_amplitude"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(parameters["t_amplitude"], [0.3, 0.3, 0.3])
    np.testing.assert_allclose(parameters["st_deviation"], [0.1, 0.1, 0.1])


def test_missing_p_waves_are_not_applicable():
    signal, fiducials = _synthetic_signal_and_fiducials()
    missing = np.full(3, -1)
    no_p = ECGFiducials(
        r_peaks=fiducials.r_peaks,
        p_onsets=missing,
        p_peaks=missing,
        p_offsets=missing,
        qrs_onsets=fiducials.qrs_onsets,
        qrs_offsets=fiducials.qrs_offsets,
        t_peaks=fiducials.t_peaks,
        t_offsets=fiducials.t_offsets,
    )
    result = measure_ecg_parameters(
        signal,
        1000,
        no_p,
        amplitude_unit="mV",
        inverse_transformed=True,
    )

    assert result["parameters"]["pr_ms"] == []
    assert result["parameters"]["p_amplitude"] == []
    assert result["p_wave_status"] == "not_applicable_or_not_delineated"
    assert len(result["parameters"]["qrs_ms"]) == 3


def test_normalized_amplitudes_are_blocked():
    signal, fiducials = _synthetic_signal_and_fiducials()
    result = measure_ecg_parameters(signal, 1000, fiducials)

    assert result["amplitude_status"] == "blocked_requires_inverse_physical_units"
    assert result["parameters"]["r_amplitude"] == []
    assert result["parameters"]["st_deviation"] == []


def test_normalized_amplitudes_require_explicit_exploratory_opt_in():
    signal, fiducials = _synthetic_signal_and_fiducials()
    result = measure_ecg_parameters(
        signal,
        1000,
        fiducials,
        allow_normalized_amplitudes=True,
    )

    assert result["amplitude_status"] == "exploratory_normalized_units_not_physical"
    assert not result["amplitude_claim_allowed"]
    np.testing.assert_allclose(result["parameters"]["r_amplitude"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(result["parameters"]["st_deviation"], [0.1, 0.1, 0.1])


def test_record_policy_excludes_pr_and_p_amplitude_for_af():
    signal, fiducials = _synthetic_signal_and_fiducials()
    result = measure_ecg_parameters(
        signal,
        1000,
        fiducials,
        allow_normalized_amplitudes=True,
        p_wave_applicable=False,
    )

    assert result["p_wave_status"] == "not_applicable_by_record_policy"
    assert result["parameters"]["pr_ms"] == []
    assert result["parameters"]["p_amplitude"] == []
    assert len(result["parameters"]["qrs_ms"]) == 3
    assert len(result["parameters"]["r_amplitude"]) == 3


def test_hrv_requires_continuity_and_minimum_duration():
    rr = [0.9, 1.0, 1.1, 1.0]
    assert compute_hrv(rr, 60, continuous=False)["status"] == "blocked_non_continuous"
    assert compute_hrv(rr, 4, continuous=True)["status"] == "blocked_insufficient_duration"

    result = compute_hrv(rr, 60, continuous=True)
    assert result["status"] == "ok"
    np.testing.assert_allclose(result["sdnn_ms"], np.std(rr, ddof=1) * 1000)
    np.testing.assert_allclose(result["rmssd_ms"], np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000)


def test_delineation_failure_is_explicit():
    result = delineate_ecg(np.zeros(32), sampling_rate=128)
    assert not result.success
    assert result.fiducials is None
    assert result.failure_reason


def test_bland_altman_and_correlation_use_paired_finite_values():
    reference = [1.0, 2.0, 3.0, np.nan]
    generated = [1.2, 2.2, 3.2, 4.0]
    agreement = bland_altman(reference, generated)
    correlation = paired_correlation(reference, generated)

    assert agreement["n"] == 3
    np.testing.assert_allclose(agreement["bias"], 0.2)
    assert agreement["difference_definition"] == "generated_minus_reference"
    np.testing.assert_allclose(correlation["r"], 1.0)


def test_clinical_agreement_uses_only_jointly_successful_records():
    measured = {
        "real": {
            0: {"subject_id": "a", "summary": {"rr_ms": 1000.0}},
            1: {"subject_id": "a", "summary": {"rr_ms": 900.0}},
        },
        "generated": {
            0: {"subject_id": "a", "summary": {"rr_ms": 1010.0}},
            2: {"subject_id": "b", "summary": {"rr_ms": 800.0}},
        },
    }

    indices, paired = _paired_records(measured)

    assert indices == [0]
    assert [record["summary"]["rr_ms"] for record in paired["real"]] == [1000.0]
    assert [record["summary"]["rr_ms"] for record in paired["generated"]] == [1010.0]
