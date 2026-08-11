import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.prepare_ptbxl_plus_delineation import _annotation_relative_path, run
from src.rcfm.interpretability.delineation_dataset import PTBXLPlusDelineationDataset


def _notes_and_samples():
    notes = []
    samples = []
    for offset in (0, 500):
        for sample, note in [
            (10, "p-wave onset"),
            (20, "p-wave peak"),
            (30, "p-wave offset"),
            (40, "QRS onset"),
            (50, "R peak"),
            (60, "QRS offset"),
            (80, "t-wave onset"),
            (100, "t-wave peak"),
            (130, "t-wave offset"),
        ]:
            samples.append(offset + sample)
            notes.append(note)
    return np.asarray(samples), notes


def test_preprocessor_and_memory_mapped_dataset_align_by_record(tmp_path, monkeypatch):
    waveform_root = tmp_path / "waveforms"
    ptbxl_root = tmp_path / "ptbxl"
    plus_root = tmp_path / "plus"
    output_root = tmp_path / "sidecar"
    waveform_root.mkdir()
    ptbxl_root.mkdir()
    plus_root.mkdir()
    split_ids = {"train": [1], "val": [2], "test": [3]}
    waveform_manifest = {
        "dataset_version": "synthetic-ptbxl",
        "split_hash": "synthetic-split",
        "split_method": "official_ptbxl_strat_fold_1_8_train_9_val_10_test",
        "patient_disjoint_verified": True,
        "stored_waveforms": {"sampling_rate_hz": 128, "samples_per_record": 1280},
    }
    (waveform_root / "dataset_manifest.json").write_text(json.dumps(waveform_manifest))
    for split, ids in split_ids.items():
        np.save(waveform_root / f"record_ids_{split}.npy", np.asarray(ids, dtype=np.int32))
        waveform = np.stack(
            [np.sin(np.linspace(0, 20 * np.pi, 1280) + lead) for lead in range(12)], axis=1
        )[None].astype(np.float32)
        np.save(waveform_root / f"X_{split}_resampled.npy", waveform)
    pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "scp_codes": ["{'NORM': 100}", "{'AFIB': 100}", "{'NORM': 100}"],
        }
    ).to_csv(ptbxl_root / "ptbxl_database.csv", index=False)
    checksum_lines = []
    for ecg_id in (1, 2, 3):
        for lead in ("I", "II"):
            relative = _annotation_relative_path(ecg_id, lead)
            path = plus_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic")
            checksum_lines.append(f"{hashlib.sha256(b'synthetic').hexdigest()} {relative}\n")
    (plus_root / "SHA256SUMS.txt").write_text("".join(checksum_lines))
    samples, notes = _notes_and_samples()
    monkeypatch.setattr(
        "scripts.prepare_ptbxl_plus_delineation.wfdb.rdann",
        lambda *_args, **_kwargs: SimpleNamespace(fs=500, sample=samples, aux_note=notes),
    )
    args = argparse.Namespace(
        waveform_root=waveform_root,
        ptbxl_root=ptbxl_root,
        ptbxl_plus_root=plus_root,
        output_dir=output_root,
        leads=["I", "II"],
        source_rate=500,
        output_rate=128,
        output_samples=1280,
        window_samples=512,
        crop_starts=[0],
        max_events=8,
        workers=2,
        allow_incomplete_download=False,
        verify_checksums=True,
    )
    run(args)

    manifest = json.loads((output_root / "dataset_manifest.json").read_text())
    assert manifest["patient_disjoint_verified"] is True
    assert manifest["splits"]["train"]["eligible_windows"] == 2
    assert manifest["download"]["unavailable_selected_record_leads"] == 0
    assert np.load(output_root / "wave_valid_val.npy")[:, :, 0].sum() == 0
    dataset = PTBXLPlusDelineationDataset(
        output_root,
        waveform_root,
        "train",
        crop_starts=[0],
    )
    sample = dataset[0]
    assert len(dataset) == 2
    assert sample["signal"].shape == (1, 512)
    assert sample["regions"].shape == (3, 512)
    assert sample["heatmaps"].shape == (9, 512)
    assert torch_is_finite(sample)


def torch_is_finite(sample):
    return all(bool(np.all(np.isfinite(value.numpy()))) for value in sample.values())
