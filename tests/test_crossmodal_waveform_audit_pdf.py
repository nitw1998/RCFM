import numpy as np

from scripts.plot_crossmodal_waveform_audit_pdf import (
    MODELS,
    _per_window_pearson,
    _select_rows,
)


def test_selection_uses_fixed_consensus_error_quantiles():
    target = np.zeros((10, 1, 512), dtype=np.float32)
    predictions = {
        model: np.arange(10, dtype=np.float32)[:, None, None] * np.ones((1, 1, 512), dtype=np.float32)
        for model in MODELS
    }
    selected, scores = _select_rows(target, predictions)
    assert [row[0] for row in selected] == ["easier", "typical", "harder"]
    assert [row[2] for row in selected] == [1, 4, 8]
    np.testing.assert_allclose(scores, np.arange(10))


def test_per_window_pearson_detects_equal_and_inverted_waveforms():
    signal = np.linspace(-1, 1, 512, dtype=np.float32)[None, None, :]
    values = _per_window_pearson(
        np.concatenate((signal, signal)), np.concatenate((signal, -signal))
    )
    np.testing.assert_allclose(values, [1, -1], atol=1e-12)
