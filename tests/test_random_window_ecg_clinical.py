import numpy as np

from scripts.evaluate_random_window_ecg_clinical import (
    DATASET_SPECS,
    _agreement_row,
    grouped_parameter_pairs,
    inverse_shared_record_minmax,
)


def test_inverse_shared_record_minmax_broadcasts_one_scaler_to_all_leads():
    normalized = np.asarray([[[-1.0, 1.0], [0.0, 0.5]]], dtype=np.float32)
    actual = inverse_shared_record_minmax(normalized, np.asarray([2.0]), np.asarray([4.0]))
    np.testing.assert_allclose(actual, [[[2.0, 6.0], [4.0, 5.0]]])


def test_grouped_parameter_pairs_averages_paired_windows_before_agreement():
    real = [{"summary": {"rr_ms": value}} for value in (10.0, 14.0, 20.0)]
    fake = [{"summary": {"rr_ms": value}} for value in (12.0, 18.0, 19.0)]
    groups, real_values, fake_values = grouped_parameter_pairs(
        real, fake, np.asarray(["a", "a", "b"]), "rr_ms"
    )
    assert groups.tolist() == ["a", "b"]
    np.testing.assert_allclose(real_values, [12.0, 20.0])
    np.testing.assert_allclose(fake_values, [15.0, 19.0])


def test_mmecg_clinical_contract_is_single_channel_normalized():
    assert DATASET_SPECS["mmecg"] == {
        "name": "mmECG",
        "leads": ("single_channel_ECG",),
        "physical": False,
        "delineation_minimum_seconds": 8.0,
    }


def test_wesad_clinical_contract_is_single_channel_normalized_128hz():
    assert DATASET_SPECS["wesad"] == {
        "name": "WESAD",
        "leads": ("chest_ECG",),
        "physical": False,
        "delineation_minimum_seconds": 4.0,
    }


def test_wesad_agreement_is_subject_level():
    row = _agreement_row(
        "wesad", "chest_ECG", "rr_ms", np.asarray(["S001", "S002"]),
        np.asarray([800.0, 900.0]), np.asarray([810.0, 890.0]), physical=False,
    )
    assert row["status"] == "ok"
    assert row["group_level"] == "subject"
    assert row["n_groups"] == 2
