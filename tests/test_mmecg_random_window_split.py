import argparse
import json
from pathlib import Path

import numpy as np

from data import get_ppg2ecg_datasets
from scripts.prepare_mmecg_random_window_split import (
    DATASET_VERSION,
    NORMALIZATION_ID,
    SOURCE_SPLIT_HASH,
    SOURCE_VERSION,
    random_window_membership,
    run,
)
from train_cfm_compare import parse_args_with_config


def _source_artifact(root: Path) -> None:
    root.mkdir(parents=True)
    definitions = {
        "train": [("S001", "1.mat"), ("S002", "2.mat")],
        "test": [("S003", "3.mat"), ("S004", "4.mat")],
    }
    cursor = 0
    for split, records in definitions.items():
        subjects = np.repeat([subject for subject, _ in records], 5)
        files = np.repeat([source_file for _, source_file in records], 5)
        rows = len(subjects)
        time = np.arange(512, dtype=np.float32)
        rcg = np.stack([np.sin(time / (11 + row)) + cursor + row for row in range(rows)])
        ecg = np.stack([2 * np.cos(time / (7 + row)) - cursor - row for row in range(rows)])
        np.save(root / f"ppg_{split}_4sec.npy", rcg.astype(np.float32))
        np.save(root / f"ecg_{split}_4sec.npy", ecg.astype(np.float32))
        np.save(root / f"subject_ids_{split}.npy", subjects)
        np.save(root / f"source_files_{split}.npy", files)
        cursor += rows
    manifest = {
        "dataset_version": SOURCE_VERSION,
        "split_hash": SOURCE_SPLIT_HASH,
        "overlap": 0.5,
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_mmecg_random_membership_is_deterministic_complete_and_window_disjoint():
    train_a, test_a = random_window_membership(100, validation_fraction=0.2, seed=31)
    train_b, test_b = random_window_membership(100, validation_fraction=0.2, seed=31)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(test_a, test_b)
    np.testing.assert_array_equal(np.sort(np.concatenate([train_a, test_a])), np.arange(100))
    assert not set(train_a.tolist()) & set(test_a.tolist())


def test_mmecg_random_artifact_preserves_windows_and_declares_raw_sample_overlap(
    tmp_path: Path,
):
    source = tmp_path / "source"
    output_root = tmp_path / "output"
    output = output_root / "mmECG"
    _source_artifact(source)
    run(
        argparse.Namespace(
            source_dir=source,
            output_dir=output,
            validation_fraction=0.2,
            seed=31,
        )
    )

    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == DATASET_VERSION
    assert manifest["source_windows"] == 20
    assert manifest["splits"]["train"]["windows"] == 16
    assert manifest["splits"]["test"]["windows"] == 4
    assert manifest["overlap"]["subject_disjoint"] is False
    assert manifest["overlap"]["source_record_disjoint"] is False
    assert manifest["overlap"]["cross_split_adjacent_window_pairs"] > 0
    assert manifest["overlap"]["shared_raw_samples_per_cross_split_adjacent_pair"] == 256
    assert manifest["normalization"]["normalization_id"] == NORMALIZATION_ID

    identities = set()
    for split in ("train", "test"):
        files = np.load(output / f"source_files_{split}.npy").astype(str)
        ordinals = np.load(output / f"source_record_window_ordinals_{split}.npy")
        identities.update(zip(files.tolist(), ordinals.tolist()))
    assert len(identities) == 20

    train_set, test_set = get_ppg2ecg_datasets(
        DATA_PATH=str(output_root),
        datasets=["mmECG"],
        normalization_id=NORMALIZATION_ID,
        return_region_mask_train=False,
    )
    assert len(train_set) == 16 and len(test_set) == 4
    for values in (
        train_set.target_ecg,
        train_set.condition_signal,
        test_set.target_ecg,
        test_set.condition_signal,
    ):
        np.testing.assert_allclose(values.min(axis=1), -1.0, atol=3e-6)
        np.testing.assert_allclose(values.max(axis=1), 1.0, atol=3e-6)


def test_mmecg_random_window_cfm_config_and_launcher_are_frozen():
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/mmecg/cfm_random_window80_20_seed31.yaml"
    args = parse_args_with_config(["--config", str(config)])
    assert args.task == "rcg2ecg"
    assert args.datasets == "mmECG"
    assert args.epochs == 200
    assert args.normalization_id == NORMALIZATION_ID
    assert args.heldout_role == "validation"
    assert args.use_minibatch_ot is False
    assert args.region_weight == 0.0
    assert args.checkpoint_policy == "latest_only"
    assert args.split_hash == "6e5365be9b71c3815907eeabab2ee6b83a11a280521243a1f79c4f90da570dc2"
    launcher = (root / "scripts/train_mmecg_random_window80_20_cfm.sh").read_text()
    assert "9973" in launcher and "2494" in launcher
    assert "3977" in launcher
    assert "--query-gpu=memory.free" in launcher
    assert "MIN_FREE_MEMORY_MIB=20480" in launcher
    assert "--query-compute-apps=pid" not in launcher
