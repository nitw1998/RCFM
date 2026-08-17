import numpy as np
import pytest
import matplotlib.pyplot as plt

from scripts.evaluate_ptbxl_clinical_agreement import (
    AMPLITUDE_PARAMETERS,
    MODEL_LABELS,
    _agreement_row,
    _configure_ieee_style,
    _inverse_oracle_minmax,
    _load_p_wave_applicability,
    _macro_rows,
    _patient_parameter_pairs,
    _valid_source_protocol,
)
from scripts.plot_ptbxl_representative_lead import _cached_parameters, _patient_pairs


def test_oracle_inverse_minmax_restores_multilead_values():
    values = np.linspace(-1, 1, 48, dtype=np.float32).reshape(2, 3, 8)
    minima = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    ranges = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.float32)
    restored = _inverse_oracle_minmax(values, minima, ranges)
    np.testing.assert_allclose(restored, (values + 1) * ranges[..., None] / 2 + minima[..., None])


def test_patient_pairs_average_repeated_records_before_agreement():
    real = [{"summary": {"rr_ms": 800}}, {"summary": {"rr_ms": 1000}}, {"summary": {"rr_ms": 700}}]
    generated = [{"summary": {"rr_ms": 820}}, {"summary": {"rr_ms": 1020}}, {"summary": {"rr_ms": 710}}]
    patients, reference, prediction = _patient_parameter_pairs(real, generated, np.array([1, 1, 2]), "rr_ms")
    np.testing.assert_array_equal(patients, ["1", "2"])
    np.testing.assert_allclose(reference, [900, 700])
    np.testing.assert_allclose(prediction, [920, 710])


def test_agreement_uses_patient_level_generated_minus_real_bland_altman():
    row = _agreement_row("rcfm", "V5", "qrs_ms", np.array([80.0, 90.0, 100.0]), np.array([85.0, 95.0, 105.0]))
    assert row["n_patients"] == 3
    assert row["ba_bias"] == pytest.approx(5.0)
    assert row["unit"] == "ms"


def test_macro_rows_average_leads_without_pooling_amplitudes():
    rows = []
    for model in ("cfm", "rcfm", "rcfm_ot", "rddm"):
        for lead, value in (("I", 1.0), ("V5", 3.0)):
            rows.append({
                "model": model, "lead": lead, "parameter": "rr_ms", "status": "ok", "unit": "ms",
                "n_patients": 10, "mae": value, "rmse": value, "pearson_r": 0.5,
                "ba_bias": 0.0, "ba_difference_sd": 1.0, "ba_lower": -2.0,
                "ba_upper": 2.0, "ba_loa_width": 4.0,
            })
    macro = _macro_rows(rows)
    cfm_rr = next(row for row in macro if row["model"] == "cfm" and row["parameter"] == "rr_ms")
    assert cfm_rr["mae"] == pytest.approx(2.0)
    assert cfm_rr["usable_leads"] == 2
    assert cfm_rr["pearson_aggregation"] == "unweighted_Fisher_z_mean_across_leads"


def test_p_wave_applicability_excludes_official_afib_and_aflt(tmp_path):
    path = tmp_path / "ptbxl_database.csv"
    path.write_text(
        "ecg_id,scp_codes\n1,\"{'NORM': 100.0}\"\n2,\"{'AFIB': 100.0}\"\n3,\"{'AFLT': 50.0, 'NORM': 50.0}\"\n",
        encoding="utf-8",
    )
    applicable, policy = _load_p_wave_applicability(path, np.array([1, 2, 3]))
    np.testing.assert_array_equal(applicable, [True, False, False])
    assert policy["not_applicable_records"] == 2


def test_ieee_clinical_style_uses_embedded_times_compatible_font():
    _configure_ieee_style()
    assert plt.rcParams["font.serif"][0] == "Liberation Serif"
    assert plt.rcParams["pdf.fonttype"] == 42


def test_smoke_protocol_requires_an_explicit_record_cap():
    smoke = {"status": "smoke_completed", "protocol": {"phase_correction_applied": False}}
    assert _valid_source_protocol(smoke, 2)
    assert not _valid_source_protocol(smoke, None)
    assert not _valid_source_protocol(
        {"status": "completed", "protocol": {"phase_correction_applied": True}}, None
    )


def test_ptbxl_clinical_protocol_includes_qrs_peak_to_peak():
    assert "qrs_peak_to_peak_amplitude" in AMPLITUDE_PARAMETERS


def test_diagmask_sixway_labels_are_available():
    assert MODEL_LABELS["cfm_ot"] == "CFM+OT"
    assert MODEL_LABELS["diag_ot"] == "RCFM-DiagMask-OT"
    assert MODEL_LABELS["semantic"] == "RCFM-SemanticMask"
    assert "neg. ctrl." in MODEL_LABELS["ecgmamba_diag"]
    assert MODEL_LABELS["ecgmamba_semantic"] == "ECGMamba-Sem"


def test_cached_representative_lead_uses_patient_means_without_imputation():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 3],
            "lead": ["V3", "V3", "V3", "I"],
            "model": ["cfm"] * 4,
            "real_rr_ms": [800.0, 1000.0, 700.0, 600.0],
            "generated_rr_ms": [820.0, 1020.0, 710.0, 605.0],
        }
    )
    assert _cached_parameters(frame) == ("rr_ms",)
    real, generated = _patient_pairs(frame, "cfm", "V3", "rr_ms")
    np.testing.assert_allclose(real, [900.0, 700.0])
    np.testing.assert_allclose(generated, [920.0, 710.0])
