import numpy as np

from scripts.evaluate_legacy_ptbxl_cfm_fold9 import _minmax_per_record


def test_legacy_ptbxl_checkpoint_era_normalization_is_per_record_minmax():
    values = np.asarray([[2.0, 4.0, 6.0], [-5.0, 0.0, 5.0]], dtype=np.float32)
    normalized = _minmax_per_record(values)
    np.testing.assert_allclose(normalized, [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    np.testing.assert_allclose(normalized.min(axis=1), -1.0)
    np.testing.assert_allclose(normalized.max(axis=1), 1.0)
