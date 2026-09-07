import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from scripts.prepare_cpsc2018_random_window_split import (
    DATASET_VERSION,
    NORMALIZATION_ID,
    _load_resampled_record,
    random_window_membership,
    run,
)
from train_cfm_compare import parse_args_with_config


def _source_artifact(root: Path, records: int = 10) -> None:
    training = root / "TrainingSet1"
    training.mkdir(parents=True)
    with (root / "REFERENCE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Recording", "First_label", "Second_label", "Third_label"])
        for index in range(records):
            record_id = f"A{index:04d}"
            writer.writerow([record_id, index % 9 + 1, "", ""])
            time = np.arange(96, dtype=np.float64) / 8.0
            signal = np.stack(
                [
                    (lead + 1) * np.sin(2 * np.pi * (0.2 + lead / 100) * time)
                    + index
                    + time / (lead + 2)
                    for lead in range(12)
                ]
            )
            savemat(training / f"{record_id}.mat", {"ECG": {"data": signal}})


def test_random_membership_is_deterministic_complete_and_disjoint_by_window():
    train_a, val_a = random_window_membership(30, validation_fraction=0.2, seed=31)
    train_b, val_b = random_window_membership(30, validation_fraction=0.2, seed=31)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(val_a, val_b)
    np.testing.assert_array_equal(np.sort(np.concatenate([train_a, val_a])), np.arange(30))
    assert not set(train_a.tolist()) & set(val_a.tolist())


def test_cpsc_random_artifact_uses_full_record_joint12_coefficients(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _source_artifact(source)
    run(
        argparse.Namespace(
            source_root=source,
            output_dir=output,
            source_rate=8,
            output_rate=4,
            validation_fraction=0.2,
            seed=31,
            minimum_range=1e-6,
        )
    )

    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == DATASET_VERSION
    assert manifest["normalization"]["normalization_id"] == NORMALIZATION_ID
    assert manifest["windowing"]["total_windows"] == 30
    assert manifest["splits"]["train"]["windows"] == 24
    assert manifest["splits"]["val"]["windows"] == 6
    assert manifest["overlap"]["record_disjoint"] is False
    assert manifest["overlap"]["records_in_both_train_and_val"] > 0
    assert manifest["normalization"]["preserves_within_record_interwindow_scale"] is True

    train = np.load(output / "X_train_resampled.npy")
    train_ids = np.load(output / "record_ids_train.npy")
    minima = np.load(output / "record_joint_minima_train.npy")
    ranges = np.load(output / "record_joint_ranges_train.npy")
    normalized = 2 * (train - minima[:, None, None]) / ranges[:, None, None] - 1
    restored = (normalized + 1) * ranges[:, None, None] / 2 + minima[:, None, None]
    np.testing.assert_allclose(restored, train, atol=2e-5)
    assert normalized.min() >= -1.0 - 1e-6
    assert normalized.max() <= 1.0 + 1e-6
    assert np.any(normalized.max(axis=(1, 2)) < 0.99)

    first_id = str(train_ids[0])
    full_record = _load_resampled_record(
        source / "TrainingSet1" / f"{first_id}.mat", source_rate_hz=8, output_rate_hz=4
    )
    assert minima[0] == np.min(full_record)
    np.testing.assert_allclose(ranges[0], np.max(full_record) - np.min(full_record))

    val_ids = np.load(output / "record_ids_val.npy")
    val_minima = np.load(output / "record_joint_minima_val.npy")
    val_ranges = np.load(output / "record_joint_ranges_val.npy")
    shared_id = next(iter(set(train_ids.tolist()) & set(val_ids.tolist())))
    train_row = int(np.flatnonzero(train_ids == shared_id)[0])
    val_row = int(np.flatnonzero(val_ids == shared_id)[0])
    assert minima[train_row] == val_minima[val_row]
    assert ranges[train_row] == val_ranges[val_row]


def test_cpsc_random_window_cfm_config_and_launcher_are_frozen():
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/cpsc2018/cfm_compare_random_window80_20_joint12_seed31.yaml"
    args = parse_args_with_config(["--config", str(config)])
    assert args.datasets == "CPSC2018"
    assert args.epochs == 200
    assert args.condition_lead_index == 1
    assert args.target_lead_indices == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert args.normalization_id == NORMALIZATION_ID
    assert args.use_minibatch_ot is False
    assert args.region_weight == 0.0
    assert args.checkpoint_policy == "latest_only"
    assert args.split_hash == "b7902b112219541e795bac4f020ef268b2951f0c3f80709f0a06f18132a743d8"
    launcher = (root / "scripts/train_cpsc2018_random_window80_20_cfm.sh").read_text()
    assert "19364" in launcher and "4842" in launcher
    assert "--query-gpu=memory.free" in launcher
    assert "MIN_FREE_MEMORY_MIB=20480" in launcher
    assert "--query-compute-apps=pid" not in launcher
