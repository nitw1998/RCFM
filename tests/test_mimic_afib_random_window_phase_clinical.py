import numpy as np

from scripts.analyze_mimic_afib_random_window_cfm_phase_clinical import (
    grouped_parameter_triplets,
)


def _items(values):
    return [{"summary": {"qrs_ms": value}} for value in values]


def test_grouped_parameter_triplets_average_joint_windows_within_record():
    groups, real, before, after = grouped_parameter_triplets(
        _items([10.0, 14.0, 20.0]),
        _items([12.0, 18.0, 19.0]),
        _items([11.0, 15.0, 21.0]),
        np.asarray(["a", "a", "b"]),
        "qrs_ms",
    )
    assert groups.tolist() == ["a", "b"]
    np.testing.assert_allclose(real, [12.0, 20.0])
    np.testing.assert_allclose(before, [15.0, 19.0])
    np.testing.assert_allclose(after, [13.0, 21.0])


def test_grouped_parameter_triplets_obey_subgroup_and_joint_availability():
    reference = _items([10.0, 20.0, 30.0])
    before = _items([11.0, 21.0, 31.0])
    after = _items([12.0, 22.0, 32.0])
    after[1]["summary"] = {}
    groups, real, unshifted, aligned = grouped_parameter_triplets(
        reference, before, after, np.asarray(["a", "a", "b"]), "qrs_ms",
        selected=np.asarray([True, True, False]),
    )
    assert groups.tolist() == ["a"]
    np.testing.assert_allclose(real, [10.0])
    np.testing.assert_allclose(unshifted, [11.0])
    np.testing.assert_allclose(aligned, [12.0])
