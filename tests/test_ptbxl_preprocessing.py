import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.preprocess_ptbxl import (
    LEAD_ORDER,
    minmax_neg1_1,
    official_split_indices,
    record_minmax_statistics,
    resample_waveform,
    run,
)


def test_official_ptbxl_folds_are_patient_disjoint():
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3, 4],
            "patient_id": [11, 12, 13, 14],
            "strat_fold": [1, 8, 9, 10],
        }
    )
    splits = official_split_indices(metadata)
    assert splits["train"].tolist() == [0, 1]
    assert splits["val"].tolist() == [2]
    assert splits["test"].tolist() == [3]


def test_official_ptbxl_folds_reject_patient_overlap():
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "patient_id": [11, 11, 13],
            "strat_fold": [1, 9, 10],
        }
    )
    with pytest.raises(ValueError, match="patient overlap"):
        official_split_indices(metadata)


def test_ptbxl_resampling_and_record_minmax_are_reversible():
    time = np.arange(5000, dtype=np.float64) / 500.0
    signal = np.stack(
        [np.sin(2 * np.pi * (index + 1) * time / 10.0) + index for index in range(12)],
        axis=1,
    )
    resampled = resample_waveform(signal)
    assert resampled.shape == (1280, 12)
    records = resampled[None, :, :]
    minima, ranges = record_minmax_statistics(records, model_window_samples=512)
    normalized = minmax_neg1_1(records[:, :512], minima, ranges)
    restored = (normalized + 1.0) * ranges[:, None, :] / 2.0 + minima[:, None, :]
    np.testing.assert_allclose(normalized.min(axis=1), -1.0, atol=2e-5)
    np.testing.assert_allclose(normalized.max(axis=1), 1.0, atol=2e-5)
    np.testing.assert_allclose(restored, records[:, :512], atol=1e-6)


def test_ptbxl_record_minmax_rejects_constant_lead():
    values = np.ones((2, 1280, 12), dtype=np.float32)
    with pytest.raises(ValueError, match="constant or near-constant"):
        record_minmax_statistics(values, model_window_samples=512)


def test_ptbxl_preprocessor_writes_rcfm_compatible_arrays(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "artifact"
    source.mkdir()
    rows = []
    for ecg_id, patient_id, fold in [(1, 11, 1), (2, 12, 9), (3, 13, 10), (4, 14, 1)]:
        relative = f"records500/{ecg_id:05d}_hr"
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".hea").write_text("synthetic\n")
        path.with_suffix(".dat").write_bytes(b"synthetic")
        rows.append(
            {
                "ecg_id": ecg_id,
                "patient_id": patient_id,
                "strat_fold": fold,
                "filename_hr": relative,
            }
        )
    pd.DataFrame(rows).to_csv(source / "ptbxl_database.csv", index=False)
    (source / "scp_statements.csv").write_text("code,diagnostic\nNORM,1\n")

    def fake_rdsamp(path):
        ecg_id = int(Path(path).stem.split("_")[0])
        time = np.arange(5000, dtype=np.float64) / 500.0
        signal = np.stack(
            [np.sin(2 * np.pi * (lead + 1) * time / 10.0) + ecg_id + lead for lead in range(12)],
            axis=1,
        )
        if ecg_id == 4:
            signal[:, 10] = 0.0
        fields = {"fs": 500, "sig_name": LEAD_ORDER, "units": ["mV"] * 12}
        return signal, fields

    monkeypatch.setattr("scripts.preprocess_ptbxl.wfdb.rdsamp", fake_rdsamp)
    args = argparse.Namespace(
        source_root=source,
        output_dir=output,
        source_rate=500,
        output_rate=128,
        duration_seconds=10,
        model_window_seconds=4,
        minimum_lead_range=1e-6,
    )
    run(args)

    for split, ecg_id in [("train", 1), ("val", 2), ("test", 3)]:
        waveforms = np.load(output / f"X_{split}_resampled.npy", allow_pickle=False)
        assert waveforms.shape == (1, 1280, 12)
        assert np.load(output / f"record_ids_{split}.npy").tolist() == [ecg_id]
        minima = np.load(output / f"record_minima_{split}.npy")
        ranges = np.load(output / f"record_ranges_{split}.npy")
        scaled = minmax_neg1_1(waveforms[:, :512], minima, ranges)
        np.testing.assert_allclose(scaled.min(axis=1), -1.0, atol=2e-5)
        np.testing.assert_allclose(scaled.max(axis=1), 1.0, atol=2e-5)

    manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert manifest["split_method"] == "official_ptbxl_strat_fold_1_8_train_9_val_10_test"
    assert manifest["patient_disjoint_verified"] is True
    assert manifest["eligible_records"] == 3
    assert manifest["signal_qc"]["excluded_record_count"] == 1
    assert manifest["excluded_records"][0]["ecg_id"] == 4
    assert manifest["excluded_records"][0]["lead_names"] == ["V5"]
    assert manifest["stored_waveforms"]["layout"] == ["records", "samples", "leads"]
    assert manifest["normalization"]["model_input"] == "per_record_per_lead_minmax_neg1_1"
