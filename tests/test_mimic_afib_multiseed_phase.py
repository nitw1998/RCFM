import numpy as np

from scripts.analyze_mimic_afib_multiseed_phase import _phase_metrics


def test_phase_metrics_use_matched_support_and_recover_known_shift():
    time = np.linspace(0.0, 4.0 * np.pi, 512, endpoint=False)
    target = np.stack((np.sin(time), np.cos(0.7 * time)))[:, None, :].astype(np.float32)
    generated = np.zeros_like(target)
    generated[..., :-7] = target[..., 7:]

    metrics, diagnostic = _phase_metrics(target, generated, max_lag=16, sampling_rate=128)

    assert diagnostic["shift_samples_median"] == 7.0
    assert metrics["oracle_aligned_fixed_support"]["rmse"] < 1e-7
    assert (
        metrics["oracle_aligned_fixed_support"]["pearson_window_median"]
        > metrics["unshifted_fixed_support"]["pearson_window_median"]
    )
