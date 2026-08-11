import json
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_rddm_ppg_artifact import (
    all_zero_ppg_mask,
    filter_paired_windows,
    prepare,
)


def test_all_zero_filter_matches_upstream_nan_to_num_and_keeps_nonzero_constant():
    ppg = np.stack(
        [
            np.zeros(512),
            np.full(512, np.nan),
            np.ones(512),
            np.concatenate([np.zeros(511), np.ones(1)]),
        ]
    )

    assert all_zero_ppg_mask(ppg).tolist() == [True, True, False, False]


def test_paired_filter_removes_matching_ecg_rows():
    ecg = np.repeat(np.arange(4)[:, None], 512, axis=1)
    ppg = np.ones((4, 512))
    ppg[[1, 3]] = 0.0

    filtered_ecg, filtered_ppg, kept, rejected = filter_paired_windows(ecg, ppg)

    assert kept.tolist() == [0, 2]
    assert rejected.tolist() == [1, 3]
    assert filtered_ecg[:, 0].tolist() == [0, 2]
    assert np.all(filtered_ppg == 1.0)


def test_prepare_writes_reproducible_manifest_and_preserves_sources(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    arrays = {
        "ecg_train_4sec.npy": np.repeat(np.arange(3)[:, None], 512, axis=1),
        "ppg_train_4sec.npy": np.stack([np.ones(512), np.zeros(512), np.full(512, 2)]),
        "ecg_test_4sec.npy": np.repeat(np.arange(2)[:, None], 512, axis=1),
        "ppg_test_4sec.npy": np.stack([np.zeros(512), np.ones(512)]),
    }
    for name, values in arrays.items():
        np.save(source / name, values, allow_pickle=False)
    source_bytes = {name: (source / name).read_bytes() for name in arrays}

    prepare(source, output)

    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1"
    assert manifest["splits"]["train"]["retained_windows"] == 2
    assert manifest["splits"]["test"]["retained_windows"] == 1
    assert np.load(output / "kept_indices_train.npy").tolist() == [0, 2]
    assert np.load(output / "filtered_all_zero_ppg_indices_test.npy").tolist() == [0]
    assert np.load(output / "ecg_train_4sec.npy")[:, 0].tolist() == [0, 2]
    assert all((source / name).read_bytes() == contents for name, contents in source_bytes.items())

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare(source, output)


def test_paired_filter_rejects_misaligned_arrays():
    with pytest.raises(ValueError, match="same number"):
        filter_paired_windows(np.zeros((2, 512)), np.zeros((1, 512)))
