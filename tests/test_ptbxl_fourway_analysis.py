import numpy as np
import pytest

import matplotlib.pyplot as plt

from scripts.analyze_ptbxl_fourway import (
    MODEL_ORDER,
    _cluster_bootstrap_difference,
    _configure_ieee_style,
    _patient_means,
)


def test_patient_means_do_not_weight_patients_by_record_count():
    patients, means = _patient_means(np.array([1.0, 3.0, 9.0]), np.array([10, 10, 20]))
    np.testing.assert_array_equal(patients, [10, 20])
    np.testing.assert_allclose(means, [2.0, 9.0])


def test_cluster_bootstrap_is_deterministic_and_paired():
    first = np.array([2.0, 4.0, 8.0, 10.0])
    second = first - 1.0
    patients = np.array([1, 1, 2, 3])
    result = _cluster_bootstrap_difference(first, second, patients, seed=5, replicates=100)
    assert result["patients"] == 3
    assert result["mean_difference"] == pytest.approx(1.0)
    assert result["bootstrap_95_ci"] == pytest.approx([1.0, 1.0])


def test_ieee_style_uses_times_compatible_embedded_fonts():
    _configure_ieee_style()
    assert plt.rcParams["font.family"] == ["serif"]
    assert plt.rcParams["font.serif"][0] == "Liberation Serif"
    assert plt.rcParams["pdf.fonttype"] == 42


def test_four_models_define_six_complete_pairwise_comparisons():
    import itertools

    pairs = tuple(itertools.combinations(MODEL_ORDER, 2))
    assert len(pairs) == 6
    assert ("cfm", "rddm") in pairs
