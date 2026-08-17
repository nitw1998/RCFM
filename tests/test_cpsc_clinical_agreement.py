import numpy as np
import pytest

from scripts.evaluate_cpsc_clinical_agreement import (
    AMPLITUDE_PARAMETERS,
    _agreement_row,
    _count_summary,
    _record_pairs,
)


def test_record_pairs_require_parameter_on_both_independent_measurements():
    real = [{"summary": {"rr_ms": 800.0}}, {"summary": {"rr_ms": 900.0}}]
    generated = [{"summary": {"rr_ms": 810.0}}, {"summary": {}}]
    indices, reference, prediction = _record_pairs(real, generated, "rr_ms")
    np.testing.assert_array_equal(indices, [0])
    np.testing.assert_allclose(reference, [800.0])
    np.testing.assert_allclose(prediction, [810.0])


def test_normalized_amplitude_agreement_is_not_labeled_physical():
    row = _agreement_row("rcfm_ot", "I", "r_amplitude", np.array([0.2, 0.4]), np.array([0.3, 0.5]))
    assert row["unit"] == "normalized"
    assert row["ba_bias"] == pytest.approx(0.1)
    assert row["difference_definition"] == "generated_minus_real"


def test_nn_count_audit_reports_short_window_density_without_hrv_values():
    result = _count_summary([3, 4, 5, 9])
    assert result["median"] == 4.5
    assert result["fraction_at_least_5_rr"] == pytest.approx(0.5)


def test_cpsc_clinical_protocol_includes_qrs_peak_to_peak():
    assert "qrs_peak_to_peak_amplitude" in AMPLITUDE_PARAMETERS


def test_cpsc_cfm_ot_label_is_available():
    from scripts.evaluate_cpsc_clinical_agreement import MODEL_LABELS

    assert MODEL_LABELS["cfm_ot"] == "CFM+OT"
