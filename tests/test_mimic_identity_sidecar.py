from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_mimic_identity_sidecar import (
    _header_metadata,
    _match_source_windows,
)


def test_header_metadata_extracts_deidentified_subject_and_record(tmp_path: Path):
    path = tmp_path / "record.hea"
    path.write_text(
        "record 2 125 100\n"
        "#<Original Subject ID>: p000123 <Original Recording ID>: p000123-2100-01-02-03-04\n",
        encoding="utf-8",
    )
    assert _header_metadata(path) == ("p000123", "p000123-2100-01-02-03-04")


def test_mapping_allows_ecg_only_only_for_zero_upstream_ppg():
    upstream_ecg = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    upstream_ppg = np.asarray([[5.0, 6.0], [0.0, 0.0]])
    source = [
        {"source_record_name": "a", "window_index": 0, "ecg": upstream_ecg[0], "ppg": upstream_ppg[0]},
        {"source_record_name": "b", "window_index": 0, "ecg": upstream_ecg[1], "ppg": np.asarray([9.0, 9.0])},
    ]
    rows, counts = _match_source_windows(upstream_ecg, upstream_ppg, source)
    assert [row["upstream_index"] for row in rows] == [0, 1]
    assert counts == {"exact_ecg_ppg_windows": 1, "ecg_only_all_zero_ppg_windows": 1}

    bad_ppg = upstream_ppg.copy()
    bad_ppg[1] = 0.5
    with pytest.raises(ValueError, match="only for an all-zero upstream PPG"):
        _match_source_windows(upstream_ecg, bad_ppg, source)
