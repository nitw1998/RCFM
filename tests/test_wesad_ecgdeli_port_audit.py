import numpy as np

from scripts.audit_wesad_ecgdeli_port_delineation import decode_fpt_samples


def test_decode_fpt_samples_maps_columns_and_padding():
    samples = np.asarray([[40, 45, 51, 58, 61, 65, 69, 77, 80, 85, 96, 103]], dtype=float)
    rows = decode_fpt_samples(samples, pad=16, signal_length=480)
    assert rows == [{
        "p_onset": 24,
        "p_peak": 29,
        "p_offset": 35,
        "qrs_onset": 42,
        "r_peak": 49,
        "qrs_offset": 61,
        "t_peak": 80,
        "t_offset": 87,
    }]


def test_decode_fpt_samples_preserves_missing_as_minus_one_and_drops_outer_r():
    samples = np.asarray([
        [np.nan, np.nan, np.nan, 10, 12, 15, 18, 20, 22, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, 58, 61, 65, 69, 77, 80, np.nan, np.nan, np.nan],
    ])
    rows = decode_fpt_samples(samples, pad=16, signal_length=480)
    assert len(rows) == 1
    assert rows[0]["p_onset"] == -1
    assert rows[0]["r_peak"] == 49
