import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_mmecg.py"
SPEC = importlib.util.spec_from_file_location("preprocess_mmecg", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_subject_split_has_no_overlap_and_is_deterministic():
    records = [
        {"subject_id": subject}
        for subject in ("1", "1", "2", "2", "3", "3", "4", "4", "5", "5")
    ]
    first = MODULE._split_records(records, test_fraction=0.4, seed=31)
    second = MODULE._split_records(records, test_fraction=0.4, seed=31)
    assert first == second
    train, test = first
    train_subjects = {records[index]["subject_id"] for index in train}
    test_subjects = {records[index]["subject_id"] for index in test}
    assert train_subjects.isdisjoint(test_subjects)


def test_windowing_uses_expected_overlap_without_cross_record_windows():
    signal = np.arange(12, dtype=np.float32)
    windows = MODULE._windows(signal, size=4, overlap=0.5)
    np.testing.assert_array_equal(
        windows,
        np.asarray([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9], [8, 9, 10, 11]]),
    )


def test_rcg_fusion_is_energy_weighted():
    mmwave = np.column_stack(
        [np.arange(1, 5, dtype=np.float64) * (index + 1) for index in range(50)]
    )
    energy = np.sum(mmwave**2, axis=0)
    expected = mmwave @ (energy / energy.sum())
    np.testing.assert_allclose(MODULE._fuse_rcg(mmwave), expected, rtol=1e-6)
