import numpy as np

from scripts.evaluate_wesad_random_window_phase_clinical import CLINICAL_SPECS, subject_agreement


def test_subject_agreement_averages_windows_before_bland_altman():
    subjects = np.asarray(["S1", "S1", "S2"])
    paired = np.asarray([0, 1, 2])
    reference = np.asarray([10.0, 14.0, 20.0])
    generated = np.asarray([12.0, 18.0, 19.0])
    summary, detail = subject_agreement(
        subjects, paired, reference, generated, "oracle_aligned", "rr_ms"
    )
    assert summary["n_subjects"] == 2
    assert summary["n_windows_contributing"] == 3
    assert summary["bland_altman_bias"] == 1.0
    assert [row["reference_mean"] for row in detail] == [12.0, 20.0]


def test_mmecg_clinical_contract_uses_actual_sampling_rate():
    assert CLINICAL_SPECS["mmecg"]["sampling_rate"] == 200.0
    assert CLINICAL_SPECS["mmecg"]["windows"] == 2494
