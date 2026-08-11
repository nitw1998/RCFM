import argparse

import numpy as np
import pytest

from scripts.evaluate_mimic_phase_corrected import (
    _fixed_support_align,
    _parse_max_lags,
)
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic


def test_fixed_support_alignment_applies_positive_delay_without_padding():
    target = np.zeros((1, 1, 12), dtype=np.float32)
    generated = np.zeros_like(target)
    target[0, 0, 5] = 1.0
    generated[0, 0, 3] = 1.0

    _, lag_values = _lag_diagnostic(target, generated, 2, sampling_rate=100)
    target_center, unshifted, aligned = _fixed_support_align(
        target, generated, lag_values["best_lag_samples"].astype(np.int32), 2
    )

    assert lag_values["best_lag_samples"].tolist() == [2]
    assert target_center.shape == unshifted.shape == aligned.shape == (1, 1, 8)
    np.testing.assert_array_equal(target_center, aligned)
    assert not np.array_equal(target_center, unshifted)


def test_fixed_support_alignment_rejects_shift_outside_margin():
    values = np.zeros((2, 1, 12), dtype=np.float32)
    with pytest.raises(ValueError, match="contain every requested shift"):
        _fixed_support_align(values, values, np.array([0, 3], dtype=np.int32), margin=2)


def test_max_lag_parser_sorts_and_rejects_duplicates():
    assert _parse_max_lags("64,16,32") == (16, 32, 64)
    with pytest.raises(argparse.ArgumentTypeError, match="unique positive"):
        _parse_max_lags("16,16")
