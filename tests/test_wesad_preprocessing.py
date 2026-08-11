import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_wesad.py"
SPEC = importlib.util.spec_from_file_location("preprocess_wesad", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_fold_one_is_subject_disjoint_and_matches_historical_subjects():
    train, test = MODULE._subject_folds(fold_index=0, seed=42)
    assert train.isdisjoint(test)
    assert train | test == set(MODULE.SUBJECTS)
    assert test == {2, 11, 14}


def test_linear_resampling_and_nonoverlapping_windows_have_expected_length():
    source = np.arange(64 * 8, dtype=np.float32)
    resampled = MODULE._resample_linear(source, source_rate=64)
    windows = MODULE._windows(resampled, window_samples=512)
    assert windows.shape == (1, 512)
    np.testing.assert_allclose(windows[0, :5], [0.0, 0.5, 1.0, 1.5, 2.0])


def test_window_labels_use_deterministic_smallest_value_for_ties():
    labels = np.asarray([2, 2, 3, 3, 1, 1, 1, 2])
    result = MODULE._window_labels(labels, window_samples=4)
    np.testing.assert_array_equal(result, [2, 1])


def test_nonempty_output_is_rejected(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "existing.txt").write_text("preserve", encoding="ascii")
    with pytest.raises(FileExistsError, match="not empty"):
        MODULE._refuse_nonempty_output(output)
