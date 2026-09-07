import numpy as np

from scripts.analyze_wesad_random_window_cfm_phase import DATASET_SPECS, VARIANTS, _metric_view
from scripts.evaluate_cpsc_zscore_paired import _waveform_metrics
from scripts.evaluate_mimic_phase_corrected import _fixed_support_align


def test_fixed_support_phase_alignment_recovers_integer_shift():
    target = np.sin(np.linspace(0, 12 * np.pi, 512, dtype=np.float32))[None, None, :]
    generated = np.roll(target, 7, axis=-1)
    center, before, after = _fixed_support_align(
        target, generated, np.asarray([-7], dtype=np.int32), margin=16
    )
    assert center.shape == before.shape == after.shape == (1, 1, 480)
    np.testing.assert_allclose(after, center, atol=1e-6)


def test_phase_metric_view_keeps_required_waveform_statistics():
    target = np.stack([
        np.linspace(-1, 1, 32, dtype=np.float32),
        np.linspace(1, -1, 32, dtype=np.float32),
    ])[:, None, :]
    summary, _ = _waveform_metrics(target, target.copy())
    view = _metric_view(summary)
    assert view["rmse"] == 0.0
    assert view["mae"] == 0.0
    assert view["waveform_fd"] == 0.0
    assert np.isclose(view["per_window_pearson_median"], 1.0)


def test_record_minmax_phase_variant_has_frozen_contract():
    assert VARIANTS["record_minmax"] == {
        "dataset_version": "wesad-all-windows-random80-20-subject-overlap-linear-resample-source-record-minmax-v2",
        "normalization_id": "source_record_minmax_neg1_1_v1",
    }


def test_mmecg_phase_variant_has_actual_200hz_contract():
    assert DATASET_SPECS["mmecg"]["sampling_rate"] == 200
    assert DATASET_SPECS["mmecg"]["windows"] == 2494
