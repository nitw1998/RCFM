import numpy as np

from scripts.analyze_mimic_afib_multiseed import _exact_sign_flip_p_value, _holm_adjust, _paired_test


def test_three_seed_exact_sign_flip_resolution_and_paired_test():
    result = _paired_test(np.array([0.8, 0.72, 0.59]), np.array([1.0, 0.9, 0.8]))
    np.testing.assert_allclose(result["mean_difference"], -0.19666666666666666)
    assert _exact_sign_flip_p_value(np.array([-0.2, -0.2, -0.2])) == 0.25


def test_holm_adjustment_is_monotone_in_sorted_raw_p_values():
    adjusted = _holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}
