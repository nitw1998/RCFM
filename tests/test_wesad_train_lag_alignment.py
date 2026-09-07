from pathlib import Path

import numpy as np
import pytest

from scripts.preprocess_wesad_train_lag_aligned import (
    ALIGNMENT_ID,
    DATASET_VERSION,
    _align_continuous,
    _first_peak_delays,
)


def test_first_peak_delays_select_only_first_peak_in_physiological_interval():
    ecg = np.asarray([100, 200, 300])
    bvp = np.asarray([105, 125, 140, 229, 370])
    delays = _first_peak_delays(ecg, bvp, 10, 50)
    np.testing.assert_array_equal(delays, [25, 29])


def test_continuous_alignment_advances_bvp_and_crops_without_wraparound():
    ecg = np.arange(8, dtype=np.float32)
    bvp = np.arange(100, 108, dtype=np.float32)
    labels = np.arange(20, 28, dtype=np.int16)
    aligned_bvp, aligned_ecg, aligned_labels = _align_continuous(bvp, ecg, labels, 2)
    np.testing.assert_array_equal(aligned_bvp, [102, 103, 104, 105, 106, 107])
    np.testing.assert_array_equal(aligned_ecg, [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(aligned_labels, [20, 21, 22, 23, 24, 25])


def test_alignment_rejects_nonpositive_or_signal_consuming_lag():
    values = np.arange(4)
    with pytest.raises(ValueError, match="positive"):
        _align_continuous(values, values, values, 0)
    with pytest.raises(ValueError, match="no common"):
        _align_continuous(values, values, values, 4)


def test_v2_contract_names_training_derived_alignment():
    assert "v2" in DATASET_VERSION
    assert "train" in ALIGNMENT_ID
    assert "fixed_lag" in ALIGNMENT_ID
