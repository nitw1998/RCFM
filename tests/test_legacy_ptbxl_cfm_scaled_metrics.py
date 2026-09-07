import numpy as np

from scripts.analyze_legacy_ptbxl_cfm_scaled_metrics import inverse_minmax, scaled_errors


def test_inverse_minmax_reconstructs_each_record_with_its_own_minimum_and_range():
    normalized = np.asarray([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    actual = inverse_minmax(normalized, np.asarray([-2.0, 10.0]), np.asarray([4.0, 2.0]))
    np.testing.assert_allclose(actual, [[-2.0, 0.0, 2.0], [10.0, 11.0, 12.0]])


def test_scaled_errors_use_record_specific_range_over_two_and_return_mv_rmse():
    reference = np.zeros((2, 2))
    generated = np.asarray([[1.0, -1.0], [0.5, -0.5]])
    scales, error_mv, record_rmse_mv = scaled_errors(reference, generated, np.asarray([2.0, 8.0]))
    np.testing.assert_allclose(scales, [1.0, 4.0])
    np.testing.assert_allclose(error_mv, [[1.0, -1.0], [2.0, -2.0]])
    np.testing.assert_allclose(record_rmse_mv, [1.0, 2.0])


def test_scaled_rmse_is_unchanged_by_full_inverse_translation():
    reference = np.asarray([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]])
    generated = reference + np.asarray([[0.1], [-0.2]])
    minima = np.asarray([-1.5, 0.75])
    ranges = np.asarray([3.0, 0.5])
    _, _, scaled_rmse = scaled_errors(reference, generated, ranges)
    physical_reference = inverse_minmax(reference, minima, ranges)
    physical_generated = inverse_minmax(generated, minima, ranges)
    physical_rmse = np.sqrt(np.mean((physical_generated - physical_reference) ** 2, axis=1))
    np.testing.assert_allclose(scaled_rmse, physical_rmse)
