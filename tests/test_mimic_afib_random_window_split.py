import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from data import get_ppg2ecg_datasets
from scripts.prepare_mimic_afib_random_window_split import (
    DATASET_VERSION,
    NORMALIZATION_ID,
    SOURCE_SPLIT_HASH,
    SOURCE_VERSION,
    random_window_membership,
    run,
)
from train_cfm_compare import parse_args_with_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_artifacts(source: Path, identity: Path) -> None:
    source.mkdir(parents=True)
    identity.mkdir(parents=True)
    cursor = 0
    for split, subjects in {
        "train": np.repeat(np.asarray(["S01", "S02"]), 5),
        "test": np.repeat(np.asarray(["S03", "S04"]), 5),
    }.items():
        rows = len(subjects)
        time = np.arange(512, dtype=np.float64)
        ppg = np.stack([np.sin(time / (11 + row)) + cursor + row for row in range(rows)])
        ecg = np.stack([np.cos(time / (7 + row)) - cursor - row for row in range(rows)])
        records = np.asarray([f"R{subject[1:]}" for subject in subjects])
        within_record = np.tile(np.arange(5, dtype=np.int16), 2)
        arrays = {
            f"ppg_{split}_4sec.npy": ppg,
            f"ecg_{split}_4sec.npy": ecg,
        }
        for name, values in arrays.items():
            np.save(source / name, values, allow_pickle=False)
        sidecars = {
            "subject_ids": subjects,
            "record_ids": records,
            "source_record_names": records,
            "window_indices": within_record,
            "start_samples_128hz": within_record.astype(np.int32) * 512,
            "afib_labels": np.asarray([(cursor + row) % 2 for row in range(rows)], dtype=bool),
        }
        for field, values in sidecars.items():
            np.save(identity / f"{field}_{split}.npy", values, allow_pickle=False)
        cursor += rows
    manifest = {
        "dataset_version": SOURCE_VERSION,
        "split_membership_hash": SOURCE_SPLIT_HASH,
    }
    (source / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    identity_manifest = {
        "status": "completed",
        "subject_disjoint_verified": True,
        "qc_manifest_sha256": _sha256(source / "dataset_manifest.json"),
    }
    (identity / "identity_manifest.json").write_text(
        json.dumps(identity_manifest), encoding="utf-8"
    )


def test_mimic_random_membership_is_deterministic_complete_and_window_disjoint():
    train_a, test_a = random_window_membership(100, validation_fraction=0.2, seed=31)
    train_b, test_b = random_window_membership(100, validation_fraction=0.2, seed=31)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(test_a, test_b)
    np.testing.assert_array_equal(np.sort(np.concatenate([train_a, test_a])), np.arange(100))
    assert not set(train_a.tolist()) & set(test_a.tolist())


def test_mimic_random_artifact_preserves_pairs_and_exposes_overlap(tmp_path: Path):
    source = tmp_path / "source"
    identity = tmp_path / "identity"
    output_root = tmp_path / "output"
    output = output_root / "MIMIC-AFib"
    _source_artifacts(source, identity)
    source_pairs = {
        tuple(np.concatenate([ecg, ppg]).tobytes())
        for split in ("train", "test")
        for ecg, ppg in zip(
            np.load(source / f"ecg_{split}_4sec.npy"),
            np.load(source / f"ppg_{split}_4sec.npy"),
        )
    }
    run(
        argparse.Namespace(
            source_dir=source,
            identity_dir=identity,
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
    assert manifest["overlap"]["subjects_in_both_train_and_test"] > 0
    assert manifest["normalization"]["normalization_id"] == NORMALIZATION_ID

    output_pairs = {
        tuple(np.concatenate([ecg, ppg]).tobytes())
        for split in ("train", "test")
        for ecg, ppg in zip(
            np.load(output / f"ecg_{split}_4sec.npy"),
            np.load(output / f"ppg_{split}_4sec.npy"),
        )
    }
    assert output_pairs == source_pairs

    train_set, test_set = get_ppg2ecg_datasets(
        DATA_PATH=str(output_root),
        datasets=["MIMIC-AFib"],
        normalization_id=NORMALIZATION_ID,
        return_region_mask_train=False,
    )
    assert len(train_set) == 16 and len(test_set) == 4
    assert train_set.target_ecg.shape == (16, 512)
    assert test_set.condition_signal.shape == (4, 512)
    assert np.all(np.isfinite(train_set.target_ecg))
    assert np.all(np.isfinite(test_set.condition_signal))


def test_mimic_random_window_cfm_config_and_launcher_are_frozen():
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/mimic_afib/cfm_random_window80_20_seed31.yaml"
    args = parse_args_with_config(["--config", str(config)])
    assert args.task == "ppg2ecg"
    assert args.datasets == "MIMIC-AFib"
    assert args.epochs == 200
    assert args.normalization_id == NORMALIZATION_ID
    assert args.heldout_role == "validation"
    assert args.use_minibatch_ot is False
    assert args.region_weight == 0.0
    assert args.checkpoint_policy == "latest_only"
    assert args.split_hash == "8b862a432969db8e13dd5cec18928f96486b983f6147fbc2ad2d8cfb4fc96232"
    launcher = (root / "scripts/train_mimic_afib_random_window80_20_cfm.sh").read_text()
    assert "8160" in launcher and "2040" in launcher
    assert "MIN_FREE_MEMORY_MIB=20480" in launcher
    assert "--query-gpu=memory.free" in launcher
