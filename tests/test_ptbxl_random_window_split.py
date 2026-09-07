import argparse
import json
from pathlib import Path

import numpy as np

from scripts.prepare_ptbxl_random_window_split import (
    DATASET_VERSION,
    WINDOW_SAMPLES,
    random_window_membership,
    run,
)
from train_cfm_compare import parse_args_with_config


def _source_artifact(root, records_per_split=4):
    root.mkdir()
    output_sha256 = {}
    next_id = 100
    for split_index, split in enumerate(("train", "val", "test")):
        values = np.empty((records_per_split, 1280, 12), dtype=np.float32)
        for row in range(records_per_split):
            time = np.arange(1280, dtype=np.float32)
            for lead in range(12):
                values[row, :, lead] = time * (lead + 1) / 1000 + 10 * split_index + row + lead
        ids = np.arange(next_id, next_id + records_per_split, dtype=np.int32)
        patients = ids // 2
        next_id += records_per_split
        np.save(root / f"X_{split}_resampled.npy", values)
        np.save(root / f"record_ids_{split}.npy", ids)
        np.save(root / f"patient_ids_{split}.npy", patients)
    manifest = {
        "dataset_version": "ptbxl-1.0.1-official-folds-record-joint12-minmax-neg1-1-v1",
        "split_hash": "source-split-hash",
        "stored_waveforms": {"physical_unit": "mV"},
        "output_sha256": output_sha256,
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_random_window_membership_is_deterministic_complete_and_record_overlapping():
    train_a, val_a = random_window_membership(100, validation_fraction=0.2, seed=31)
    train_b, val_b = random_window_membership(100, validation_fraction=0.2, seed=31)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(val_a, val_b)
    assert len(train_a) == 160
    assert len(val_a) == 40
    assert len(set((train_a // 2).tolist()) & set((val_a // 2).tolist())) > 0
    np.testing.assert_array_equal(np.sort(np.concatenate([train_a, val_a])), np.arange(200))


def test_random_window_artifact_shares_full_record_joint12_coefficients(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _source_artifact(source)
    run(
        argparse.Namespace(
            source_dir=source,
            output_dir=output,
            validation_fraction=0.2,
            seed=31,
            minimum_range_mv=1e-6,
        )
    )
    train = np.load(output / "X_train_resampled.npy")
    val = np.load(output / "X_val_resampled.npy")
    assert train.shape == (19, WINDOW_SAMPLES, 12)
    assert val.shape == (5, WINDOW_SAMPLES, 12)
    minima = np.load(output / "record_joint_minima_train.npy")
    ranges = np.load(output / "record_joint_ranges_train.npy")
    normalized = 2 * (train - minima[:, None, None]) / ranges[:, None, None] - 1
    restored = (normalized + 1) * ranges[:, None, None] / 2 + minima[:, None, None]
    np.testing.assert_allclose(restored, train, atol=2e-6)
    assert normalized.min() >= -1.0 - 1e-6
    assert normalized.max() <= 1.0 + 1e-6
    assert np.any(normalized.max(axis=(1, 2)) < 0.99)

    source_codes = np.load(output / "source_split_codes_train.npy")
    source_rows = np.load(output / "source_local_rows_train.npy")
    for index in range(len(train)):
        split = ("train", "val", "test")[int(source_codes[index])]
        full_record = np.load(source / f"X_{split}_resampled.npy", mmap_mode="r")[
            int(source_rows[index])
        ]
        assert minima[index] == np.min(full_record)
        np.testing.assert_allclose(ranges[index], np.max(full_record) - np.min(full_record))

    train_ids = np.load(output / "record_ids_train.npy")
    val_ids = np.load(output / "record_ids_val.npy")
    val_minima = np.load(output / "record_joint_minima_val.npy")
    val_ranges = np.load(output / "record_joint_ranges_val.npy")
    shared_id = next(iter(set(train_ids.tolist()) & set(val_ids.tolist())))
    train_row = int(np.flatnonzero(train_ids == shared_id)[0])
    val_row = int(np.flatnonzero(val_ids == shared_id)[0])
    assert minima[train_row] == val_minima[val_row]
    assert ranges[train_row] == val_ranges[val_row]
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert manifest["dataset_version"] == DATASET_VERSION
    assert manifest["overlap"]["record_disjoint"] is False
    assert manifest["overlap"]["patient_disjoint"] is False
    assert manifest["overlap"]["records_in_both_train_and_val"] > 0
    assert manifest["normalization"]["preserves_interlead_relative_amplitudes_and_offsets"] is True
    assert manifest["normalization"]["preserves_within_record_interwindow_scale"] is True
    assert "full_10_second_source_record" in manifest["normalization"]["coefficient_scope"]


def test_random_window_cfm_config_is_frozen_to_200_epochs_and_non_grouped_artifact():
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/ptbxl/cfm_compare_random_window80_20_joint12_seed31.yaml"
    args = parse_args_with_config(["--config", str(config)])
    assert args.epochs == 200
    assert args.condition_lead_index == 1
    assert args.target_lead_indices == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert args.use_minibatch_ot is False
    assert args.region_weight == 0.0
    assert args.checkpoint_policy == "latest_only"
    assert args.normalization_id == "source_record_joint12_minmax_neg1_1_v1"
    assert args.split_hash == "9ba296dc33ef6f29f9368ae4d1dd61feceb9366100b7b4afbc8698ea7592012c"
    launcher = (root / "scripts/train_ptbxl_random_window80_20_cfm.sh").read_text()
    assert "34939" in launcher and "8735" in launcher
    assert "record_disjoint" in launcher and "patient_disjoint" in launcher
    assert "--query-gpu=memory.free" in launcher
    assert "MIN_FREE_MEMORY_MIB=20480" in launcher
    assert "FREE_MEMORY_MIB <= MIN_FREE_MEMORY_MIB" in launcher
    assert "--query-compute-apps=pid" not in launcher
