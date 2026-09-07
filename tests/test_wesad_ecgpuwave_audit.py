import numpy as np

from scripts.audit_wesad_ecgpuwave_delineation import parse_ecgpuwave_annotations


def test_parse_ecgpuwave_annotations_maps_wave_types_and_removes_padding():
    samples = np.asarray([40, 45, 51, 58, 65, 77, 85, 96, 103])
    symbols = ["(", "p", ")", "(", "N", ")", "(", "t", ")"]
    nums = np.asarray([0, 0, 0, 1, 0, 1, 2, 0, 2])

    rows = parse_ecgpuwave_annotations(
        samples, symbols, nums, pad_samples=16, signal_length=480
    )

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


def test_parse_ecgpuwave_annotations_does_not_reuse_previous_beat_p_wave():
    samples = np.asarray([40, 45, 51, 58, 65, 77, 164, 170, 186])
    symbols = ["(", "p", ")", "(", "N", ")", "(", "N", ")"]
    nums = np.asarray([0, 0, 0, 1, 0, 1, 1, 0, 1])

    rows = parse_ecgpuwave_annotations(
        samples, symbols, nums, pad_samples=16, signal_length=480
    )

    assert rows[1]["p_onset"] == -1
    assert rows[1]["p_peak"] == -1
    assert rows[1]["p_offset"] == -1
    assert rows[1]["qrs_onset"] == 148
    assert rows[1]["r_peak"] == 154
    assert rows[1]["qrs_offset"] == 170
