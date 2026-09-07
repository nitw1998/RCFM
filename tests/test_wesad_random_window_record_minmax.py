import argparse
import json
from pathlib import Path

import numpy as np

from data import get_ppg2ecg_datasets
from scripts.prepare_wesad_random_window_record_minmax import (
    DATASET_VERSION,
    NORMALIZATION_ID,
    run,
    source_record_coefficients,
)
from scripts.prepare_wesad_random_window_split import SOURCE_SPLIT_HASH, SOURCE_VERSION
from train_cfm_compare import parse_args_with_config


def _source_artifact(root: Path) -> None:
    root.mkdir(parents=True)
    split_subjects = {
        "train": np.repeat(np.asarray(["S001", "S002", "S003"]), 4),
        "test": np.repeat(np.asarray(["S004", "S005"]), 4),
    }
    cursor = 0
    for split, subjects in split_subjects.items():
        rows = len(subjects)
        time = np.arange(512, dtype=np.float32)
        ppg = np.stack(
            [np.sin(time / (13 + row)) + 5 * (cursor + row) for row in range(rows)]
        )
        ecg = np.stack(
            [2 * np.cos(time / (9 + row)) - 7 * (cursor + row) for row in range(rows)]
        )
        labels = np.arange(cursor, cursor + rows, dtype=np.int16) % 8
        np.save(root / f"ppg_{split}_4sec.npy", ppg.astype(np.float32))
        np.save(root / f"ecg_{split}_4sec.npy", ecg.astype(np.float32))
        np.save(root / f"labels_{split}.npy", labels)
        np.save(root / f"subject_ids_{split}.npy", subjects)
        cursor += rows
    manifest = {
        "dataset_version": SOURCE_VERSION,
        "split_hash": SOURCE_SPLIT_HASH,
        "alignment": "native_common_start_and_duration_same_window_boundaries_no_delay_correction",
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_source_record_coefficients_are_shared_only_within_subject():
    signals = np.asarray([[0, 1], [3, 5], [-4, 2]], dtype=np.float32)
    subjects = np.asarray(["A", "A", "B"])
    minima, ranges = source_record_coefficients(signals, subjects)
    np.testing.assert_array_equal(minima, [0, 0, -4])
    np.testing.assert_array_equal(ranges, [5, 5, 6])


def test_record_minmax_artifact_reuses_full_subject_coefficients_across_split(
    tmp_path: Path,
):
    source = tmp_path / "source"
    output_root = tmp_path / "output"
    output = output_root / "WESAD"
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
    assert manifest["normalization"]["normalization_id"] == NORMALIZATION_ID
    assert manifest["normalization"]["computed_before_random_window_split"] is True
    assert manifest["normalization"]["target_and_condition_scalers_shared"] is False

    raw_by_split = {}
    for split in ("train", "test"):
        subjects = np.load(output / f"subject_ids_{split}.npy").astype(str)
        raw_target = np.load(output / f"ecg_{split}_4sec.npy")
        target_minima = np.load(output / f"target_record_minima_{split}.npy")
        target_ranges = np.load(output / f"target_record_ranges_{split}.npy")
        condition_minima = np.load(output / f"condition_record_minima_{split}.npy")
        condition_ranges = np.load(output / f"condition_record_ranges_{split}.npy")
        assert target_minima.shape == target_ranges.shape == (len(subjects),)
        assert condition_minima.shape == condition_ranges.shape == (len(subjects),)
        raw_by_split[split] = (subjects, raw_target, target_minima, target_ranges)

    common = set(raw_by_split["train"][0]) & set(raw_by_split["test"][0])
    assert common
    for subject in common:
        coefficients = []
        for split in ("train", "test"):
            subjects, _, minima, ranges = raw_by_split[split]
            selected = subjects == subject
            assert len(np.unique(minima[selected])) == 1
            assert len(np.unique(ranges[selected])) == 1
            coefficients.append((minima[selected][0], ranges[selected][0]))
        assert coefficients[0] == coefficients[1]

    train_set, test_set = get_ppg2ecg_datasets(
        DATA_PATH=str(output_root),
        datasets=["WESAD"],
        normalization_id=NORMALIZATION_ID,
        return_region_mask_train=False,
    )
    assert len(train_set) == 16 and len(test_set) == 4
    assert train_set.record_ids is not None and test_set.record_ids is not None
    np.testing.assert_array_equal(train_set.record_ids.astype(str), raw_by_split["train"][0])
    np.testing.assert_array_equal(test_set.record_ids.astype(str), raw_by_split["test"][0])
    np.testing.assert_array_equal(train_set.target_offsets, raw_by_split["train"][2])
    np.testing.assert_array_equal(train_set.target_scales, raw_by_split["train"][3])
    assert any(
        not np.isclose(window.min(), -1.0) or not np.isclose(window.max(), 1.0)
        for window in train_set.target_ecg
    )
    for split, dataset in (("train", train_set), ("test", test_set)):
        _, raw, minima, ranges = raw_by_split[split]
        restored = (dataset.target_ecg + 1.0) * ranges[:, None] / 2.0 + minima[:, None]
        np.testing.assert_allclose(restored, raw, rtol=2e-6, atol=2e-5)

    combined_subjects = np.concatenate(
        [raw_by_split["train"][0], raw_by_split["test"][0]]
    )
    combined_scaled = np.concatenate([train_set.target_ecg, test_set.target_ecg])
    for subject in np.unique(combined_subjects):
        selected = combined_subjects == subject
        np.testing.assert_allclose(combined_scaled[selected].min(), -1.0, atol=3e-6)
        np.testing.assert_allclose(combined_scaled[selected].max(), 1.0, atol=3e-6)


def test_record_minmax_cfm_config_and_launcher_are_frozen():
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/wesad/cfm_random_window80_20_record_minmax_seed31.yaml"
    args = parse_args_with_config(["--config", str(config)])
    assert args.task == "ppg2ecg"
    assert args.datasets == "WESAD"
    assert args.epochs == 200
    assert args.normalization_id == NORMALIZATION_ID
    assert args.heldout_role == "validation"
    assert args.use_minibatch_ot is False
    assert args.region_weight == 0.0
    assert args.checkpoint_policy == "latest_only"
    assert args.split_hash == "ef5687b00e5cc3809a8ac3d6b95d05671ee18b37182e04fd7635fe6657a3906c"
    launcher = (
        root / "scripts/train_wesad_random_window80_20_record_minmax_cfm.sh"
    ).read_text()
    assert "17365" in launcher and "4342" in launcher
    assert "target_record_minima_train.npy" in launcher
    assert "--query-gpu=memory.free" in launcher
    assert "MIN_FREE_MEMORY_MIB=20480" in launcher
