#!/usr/bin/env python3
"""Re-split all frozen native-alignment WESAD windows randomly 80:20."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


SOURCE_VERSION = "wesad-subject-fold1-linear-resample-window-minmax-v1"
SOURCE_SPLIT_HASH = "0b90bffe7c3032243c803e534de3592618284d26792c2b612ce7af17a81a85cd"
DATASET_VERSION = "wesad-all-windows-random80-20-subject-overlap-linear-resample-window-minmax-v1"
NORMALIZATION_ID = "window_minmax_neg1_1_v1"
WINDOW_SAMPLES = 512
SOURCE_SPLITS = ("train", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def random_window_membership(
    windows: int, *, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if windows <= 1 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("windows and validation_fraction must define a nonempty split")
    indices = np.arange(windows, dtype=np.int64)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
    )
    if len(np.intersect1d(train_indices, test_indices)):
        raise AssertionError("random splits overlap by window identity")
    if len(train_indices) + len(test_indices) != windows:
        raise AssertionError("random splits do not cover all windows")
    return np.asarray(train_indices), np.asarray(test_indices)


def _window_ordinals(subjects: np.ndarray) -> np.ndarray:
    counts: Counter[str] = Counter()
    ordinals = np.empty(len(subjects), dtype=np.int32)
    for index, subject in enumerate(subjects.astype(str)):
        ordinals[index] = counts[subject]
        counts[subject] += 1
    return ordinals


def _split_hash(
    subjects: np.ndarray,
    ordinals: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> str:
    lines = []
    for split, indices in (("train", train_indices), ("test", test_indices)):
        for index in indices:
            lines.append(f"{subjects[index]}\t{int(ordinals[index])}\t{split}")
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()


def _load_source(
    source_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, str]]:
    manifest_path = source_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != SOURCE_VERSION:
        raise ValueError("source must be the frozen native-alignment WESAD artifact")
    if manifest.get("split_hash") != SOURCE_SPLIT_HASH:
        raise ValueError("source WESAD subject-fold hash changed")
    if manifest.get("alignment") != "native_common_start_and_duration_same_window_boundaries_no_delay_correction":
        raise ValueError("source WESAD alignment changed")
    arrays: dict[str, list[np.ndarray]] = {
        "ppg": [],
        "ecg": [],
        "labels": [],
        "subjects": [],
        "source_split_codes": [],
        "source_rows": [],
    }
    source_hashes = {"dataset_manifest.json": _sha256(manifest_path)}
    for split_code, split in enumerate(SOURCE_SPLITS):
        paths = {
            "ppg": source_dir / f"ppg_{split}_4sec.npy",
            "ecg": source_dir / f"ecg_{split}_4sec.npy",
            "labels": source_dir / f"labels_{split}.npy",
            "subjects": source_dir / f"subject_ids_{split}.npy",
        }
        loaded = {key: np.load(path, allow_pickle=False) for key, path in paths.items()}
        rows = len(loaded["ppg"])
        if loaded["ppg"].shape != (rows, WINDOW_SAMPLES):
            raise ValueError(f"source {split} PPG shape changed")
        if loaded["ecg"].shape != loaded["ppg"].shape:
            raise ValueError(f"source {split} ECG/PPG pairing changed")
        if len(loaded["labels"]) != rows or len(loaded["subjects"]) != rows:
            raise ValueError(f"source {split} sidecars do not align")
        if not np.all(np.isfinite(loaded["ppg"])) or not np.all(np.isfinite(loaded["ecg"])):
            raise ValueError(f"source {split} contains nonfinite waveforms")
        if np.any(np.ptp(loaded["ppg"], axis=1) <= 0) or np.any(
            np.ptp(loaded["ecg"], axis=1) <= 0
        ):
            raise ValueError(f"source {split} contains constant windows")
        for key in ("ppg", "ecg", "labels", "subjects"):
            arrays[key].append(loaded[key])
            source_hashes[paths[key].name] = _sha256(paths[key])
        arrays["source_split_codes"].append(np.full(rows, split_code, dtype=np.uint8))
        arrays["source_rows"].append(np.arange(rows, dtype=np.int32))
    return {key: np.concatenate(parts) for key, parts in arrays.items()}, manifest, source_hashes


def run(args: argparse.Namespace) -> Path:
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    repository_root = Path(__file__).resolve().parents[1]
    if output_dir == repository_root or repository_root in output_dir.parents:
        raise ValueError("preprocessed datasets must be written outside the Git repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays, source_manifest, source_hashes = _load_source(source_dir)
    ordinals = _window_ordinals(arrays["subjects"])
    identities = list(zip(arrays["subjects"].astype(str).tolist(), ordinals.tolist()))
    if len(set(identities)) != len(identities):
        raise ValueError("source subject/window identities are not unique")
    train_indices, test_indices = random_window_membership(
        len(ordinals), validation_fraction=args.validation_fraction, seed=args.seed
    )
    summaries = {}
    output_hashes = {}
    for split, indices in (("train", train_indices), ("test", test_indices)):
        outputs = {
            f"ppg_{split}_4sec.npy": arrays["ppg"][indices].astype(np.float32, copy=False),
            f"ecg_{split}_4sec.npy": arrays["ecg"][indices].astype(np.float32, copy=False),
            f"labels_{split}.npy": arrays["labels"][indices].astype(np.int16, copy=False),
            f"subject_ids_{split}.npy": arrays["subjects"][indices],
            f"subject_window_ordinals_{split}.npy": ordinals[indices],
            f"source_split_codes_{split}.npy": arrays["source_split_codes"][indices],
            f"source_rows_{split}.npy": arrays["source_rows"][indices],
        }
        for name, values in outputs.items():
            path = output_dir / name
            np.save(path, values, allow_pickle=False)
            output_hashes[name] = _sha256(path)
        subjects = outputs[f"subject_ids_{split}.npy"]
        labels = outputs[f"labels_{split}.npy"]
        summaries[split] = {
            "windows": int(len(indices)),
            "unique_subjects": int(len(np.unique(subjects))),
            "label_counts": {
                str(label): int(count)
                for label, count in zip(*np.unique(labels, return_counts=True))
            },
        }

    train_subjects = set(
        np.load(output_dir / "subject_ids_train.npy", allow_pickle=False).astype(str).tolist()
    )
    test_subjects = set(
        np.load(output_dir / "subject_ids_test.npy", allow_pickle=False).astype(str).tolist()
    )
    split_hash = _split_hash(arrays["subjects"], ordinals, train_indices, test_indices)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "dataset": "WESAD",
        "dataset_version": DATASET_VERSION,
        "source_dataset_version": source_manifest["dataset_version"],
        "source_split_hash": source_manifest["split_hash"],
        "source_file_sha256": dict(sorted(source_hashes.items())),
        "subject_count": int(len(np.unique(arrays["subjects"]))),
        "source_windows": int(len(ordinals)),
        "split_method": "sklearn_train_test_split_over_all_windows_without_subject_grouping",
        "split_seed": args.seed,
        "train_fraction": 1.0 - args.validation_fraction,
        "validation_fraction": args.validation_fraction,
        "heldout_file_role": "test_files_used_as_training_validation_only",
        "split_hash": split_hash,
        "splits": summaries,
        "overlap": {
            "subject_disjoint": False,
            "subjects_in_both_train_and_test": len(train_subjects & test_subjects),
            "subject_ids_in_both": sorted(train_subjects & test_subjects),
            "same_continuous_subject_record_can_cross_splits": True,
            "allowed_by_protocol": True,
        },
        "condition": "wrist_BVP_64Hz_linearly_resampled_to_128Hz",
        "target": "chest_ECG_700Hz_linearly_resampled_to_128Hz",
        "alignment_id": "native_common_start_same_window_no_delay_correction_random80_20_v1",
        "alignment": "native common start/duration and same window boundaries; no delay correction",
        "window_seconds": 4,
        "window_samples": WINDOW_SAMPLES,
        "window_overlap": 0.0,
        "normalization": {
            "normalization_id": NORMALIZATION_ID,
            "model_input": "independent_per_window_per_modality_minmax_neg1_1",
            "application": "deferred_to_loader",
            "preserves_cross_modality_amplitude": False,
            "reason_not_shared": "BVP and ECG have different unverified sensor units",
        },
        "signal_cleaning": "none",
        "output_sha256": dict(sorted(output_hashes.items())),
        "claim_boundary": (
            "This non-grouped comparison permits the same subject and continuous recording "
            "in training and validation. It does not measure subject-independent generalization."
        ),
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"split_hash": split_hash, "splits": summaries, "overlap": manifest["overlap"]},
            indent=2,
        )
    )
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=31)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
