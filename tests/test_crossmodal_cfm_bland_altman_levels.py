import numpy as np

from scripts.plot_crossmodal_cfm_bland_altman_levels import (
    aggregate_subject_pairs,
    finite_parameter_pairs,
)


def _row(subject, reference, generated):
    return {
        "subject_id": subject,
        "reference": {"qrs_ms": reference},
        "generated": {"qrs_ms": generated},
    }


def test_finite_pairs_are_parameter_specific_and_pairwise_complete():
    rows = [
        _row("s1", 10.0, 12.0),
        _row("s1", 20.0, float("nan")),
        _row("s2", float("nan"), 30.0),
        _row("s2", 40.0, 44.0),
    ]
    subjects, reference, generated = finite_parameter_pairs(rows, "qrs_ms")
    assert subjects.tolist() == ["s1", "s2"]
    np.testing.assert_array_equal(reference, [10.0, 40.0])
    np.testing.assert_array_equal(generated, [12.0, 44.0])


def test_subject_level_uses_within_subject_paired_means():
    subjects = np.asarray(["s2", "s1", "s1", "s2"])
    reference = np.asarray([30.0, 10.0, 20.0, 50.0])
    generated = np.asarray([33.0, 12.0, 24.0, 55.0])
    ids, real_means, fake_means, counts = aggregate_subject_pairs(subjects, reference, generated)
    assert ids.tolist() == ["s1", "s2"]
    np.testing.assert_array_equal(real_means, [15.0, 40.0])
    np.testing.assert_array_equal(fake_means, [18.0, 44.0])
    np.testing.assert_array_equal(counts, [2, 2])


def test_subject_level_drops_nonfinite_pairs_without_dropping_subject():
    subjects = np.asarray(["s1", "s1", "s2"])
    reference = np.asarray([10.0, np.nan, 30.0])
    generated = np.asarray([11.0, 100.0, 33.0])
    ids, real_means, fake_means, counts = aggregate_subject_pairs(subjects, reference, generated)
    assert ids.tolist() == ["s1", "s2"]
    np.testing.assert_array_equal(real_means, [10.0, 30.0])
    np.testing.assert_array_equal(fake_means, [11.0, 33.0])
    np.testing.assert_array_equal(counts, [1, 1])
