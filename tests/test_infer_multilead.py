import numpy as np
import pytest

from infer_rcfm import _inverse_record_minmax_neg1_1, _inverse_record_zscore


def test_multilead_record_zscore_inverse_preserves_shape_and_values():
    normalized = np.arange(48, dtype=np.float32).reshape(2, 3, 8) / 10.0
    means = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    scales = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32)

    restored = _inverse_record_zscore(normalized, means, scales)

    assert restored.shape == normalized.shape
    np.testing.assert_allclose(restored, normalized * scales[..., None] + means[..., None])


def test_multilead_record_minmax_inverse_preserves_shape_and_values():
    normalized = np.linspace(-1.0, 1.0, 48, dtype=np.float32).reshape(2, 3, 8)
    offsets = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ranges = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32)

    restored = _inverse_record_minmax_neg1_1(normalized, offsets, ranges)

    assert restored.shape == normalized.shape
    expected = (normalized + 1.0) * ranges[..., None] / 2.0 + offsets[..., None]
    np.testing.assert_allclose(restored, expected)


def test_record_inverse_rejects_mismatched_lead_coefficients():
    with pytest.raises(ValueError, match="must have shape"):
        _inverse_record_zscore(
            np.zeros((2, 11, 8), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
        )
