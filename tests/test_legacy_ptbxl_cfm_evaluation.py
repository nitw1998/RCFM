import numpy as np
from argparse import Namespace

from scripts.evaluate_legacy_ptbxl_cfm_fold9 import _minmax_per_record, load_data


def test_legacy_ptbxl_checkpoint_era_normalization_is_per_record_minmax():
    values = np.asarray([[2.0, 4.0, 6.0], [-5.0, 0.0, 5.0]], dtype=np.float32)
    normalized = _minmax_per_record(values)
    np.testing.assert_allclose(normalized, [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    np.testing.assert_allclose(normalized.min(axis=1), -1.0)
    np.testing.assert_allclose(normalized.max(axis=1), 1.0)


def test_official_fold10_loads_test_waveforms_and_patient_ids(tmp_path):
    values = np.zeros((2, 512, 12), dtype=np.float32)
    values[:, :, 2] = np.asarray([[0.0, 1.0] * 256, [1.0, 3.0] * 256])
    values[:, :, 10] = np.asarray([[2.0, 4.0] * 256, [-2.0, 2.0] * 256])
    np.save(tmp_path / "X_test_resampled.npy", values)
    np.save(tmp_path / "patient_ids_test.npy", np.asarray([101, 202]))
    args = Namespace(
        split_source="official_fold10",
        historical_root=tmp_path / "unused",
        official_root=tmp_path,
        max_records=None,
    )

    target, source, patient_ids, _, path = load_data(args)

    assert path == tmp_path / "X_test_resampled.npy"
    assert target.shape == source.shape == (2, 1, 512)
    np.testing.assert_array_equal(patient_ids, [101, 202])
    np.testing.assert_allclose(target.min(axis=2), -1.0)
    np.testing.assert_allclose(target.max(axis=2), 1.0)
