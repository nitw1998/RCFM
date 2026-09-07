import numpy as np

from scripts.analyze_rddm_vs_rcfm_ot_random_window_multiseed import (
    _batch_wfd,
    _phase_correct,
)


def test_batch_wfd_uses_batch_then_lead_unweighted_means():
    targets = np.arange(6 * 2 * 4, dtype=np.float32).reshape(6, 2, 4) / 20
    predictions = targets.copy()
    predictions[:4, 0] += 0.1
    predictions[4:, 1] += 0.3
    value, rows = _batch_wfd(
        "synthetic", "RDDM", 31, targets, predictions, batch_size=4
    )
    assert len(rows) == 4
    lead_means = [
        np.mean([row["waveform_fd"] for row in rows if row["lead_index"] == lead])
        for lead in (0, 1)
    ]
    assert value == np.mean(lead_means)


def test_phase_correction_returns_common_480_support():
    x = np.sin(np.linspace(0, 10, 512, dtype=np.float32))[None, None]
    shifted = np.roll(x, 3, axis=-1)
    target, aligned = _phase_correct(x, shifted, sampling_rate=128)
    assert target.shape == aligned.shape == (1, 1, 480)
    assert np.corrcoef(target.ravel(), aligned.ravel())[0, 1] > 0.999
