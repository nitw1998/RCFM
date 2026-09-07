import numpy as np
import pytest

from scripts.verify_ptbxl_per_lead_rmse_wfdb import (
    _normalize_first_window,
    _rmse_from_sse,
)


def test_first_window_normalization_is_independent_per_lead():
    waveform = np.zeros((520, 2), dtype=np.float32)
    waveform[:512, 0] = np.linspace(2.0, 4.0, 512)
    waveform[:512, 1] = np.linspace(-3.0, 1.0, 512)
    normalized, minima, ranges = _normalize_first_window(waveform)
    np.testing.assert_allclose(minima, [2.0, -3.0])
    np.testing.assert_allclose(ranges, [2.0, 4.0])
    np.testing.assert_allclose(normalized[[0, -1]], [[-1.0, -1.0], [1.0, 1.0]])


def test_rmse_accumulator_uses_observation_count_not_macro_average():
    result = _rmse_from_sse(np.asarray([4.0, 9.0]), observations=4)
    np.testing.assert_allclose(result, [1.0, 1.5])
    with pytest.raises(ValueError):
        _rmse_from_sse(np.asarray([1.0]), observations=0)
