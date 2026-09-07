import numpy as np

from scripts.analyze_legacy_ptbxl_cfm_waveform import waveform_summary


def test_waveform_summary_uses_generated_minus_reference_bland_altman():
    reference = np.asarray([[[0.0, 1.0]], [[1.0, 2.0]]])
    generated = reference + 0.25
    result = waveform_summary(reference, generated)
    assert result["rmse"] == 0.25
    assert result["mae"] == 0.25
    assert result["record_pearson_median"] == 1.0
    assert result["pointwise_bland_altman_descriptive_only"]["bias"] == 0.25
