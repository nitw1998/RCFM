"""Prepare subject-isolated WESAD wrist-BVP-to-chest-ECG windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold


SUBJECTS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17)
BVP_RATE_HZ = 64
ECG_RATE_HZ = 700
OUTPUT_RATE_HZ = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--window_seconds", type=int, default=4)
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--fold_seed", type=int, default=42)
    return parser.parse_args()


def _resample_linear(signal: np.ndarray, source_rate: int) -> np.ndarray:
    values = np.asarray(signal).reshape(-1)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("signals must contain at least two finite samples")
    source_time = np.arange(len(values), dtype=np.float64) / source_rate
    target_time = np.arange(0.0, source_time[-1], 1.0 / OUTPUT_RATE_HZ)
    interpolator = interp1d(
        source_time,
        values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return np.asarray(interpolator(target_time), dtype=np.float32)


def _resample_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels).reshape(-1)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("labels must contain at least two finite samples")
    source_time = np.arange(len(values), dtype=np.float64) / ECG_RATE_HZ
    target_time = np.arange(0.0, source_time[-1], 1.0 / OUTPUT_RATE_HZ)
    interpolator = interp1d(
        source_time,
        values.astype(np.int16),
        kind="nearest",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return np.asarray(interpolator(target_time), dtype=np.int16)


def _windows(signal: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    count = len(signal) // window_samples
    if count == 0:
        raise ValueError("signal is shorter than one window")
    return np.asarray(signal[: count * window_samples]).reshape(count, window_samples)


def _window_labels(labels: np.ndarray, window_samples: int) -> np.ndarray:
    windows = _windows(labels, window_samples)
    modes = []
    for window in windows:
        values, counts = np.unique(window, return_counts=True)
        modes.append(values[np.argmax(counts)])
    return np.asarray(modes, dtype=np.int16)


def _subject_folds(fold_index: int, seed: int) -> tuple[set[int], set[int]]:
    if not 0 <= fold_index < 5:
        raise ValueError("fold_index must be in [0, 4]")
    subjects = np.asarray(SUBJECTS)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(subjects))
    train_indices, test_indices = folds[fold_index]
    return set(subjects[train_indices].tolist()), set(subjects[test_indices].tolist())


def _load_subject(source_root: Path, subject: int) -> dict[str, np.ndarray]:
    path = source_root / f"S{subject}" / f"S{subject}.pkl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    bvp = np.asarray(payload["signal"]["wrist"]["BVP"]).reshape(-1)
    ecg = np.asarray(payload["signal"]["chest"]["ECG"]).reshape(-1)
    labels = np.asarray(payload["label"]).reshape(-1)
    bvp_duration = len(bvp) / BVP_RATE_HZ
    ecg_duration = len(ecg) / ECG_RATE_HZ
    if not np.isclose(bvp_duration, ecg_duration, atol=1e-9, rtol=0):
        raise ValueError(f"S{subject} BVP/ECG durations differ")
    return {"bvp": bvp, "ecg": ecg, "labels": labels}


def _prepare_subject(
    source_root: Path, subject: int, window_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = _load_subject(source_root, subject)
    bvp = _resample_linear(raw["bvp"], BVP_RATE_HZ)
    ecg = _resample_linear(raw["ecg"], ECG_RATE_HZ)
    labels = _resample_labels(raw["labels"])
    common_length = min(len(bvp), len(ecg), len(labels))
    bvp_windows = _windows(bvp[:common_length], window_samples)
    ecg_windows = _windows(ecg[:common_length], window_samples)
    label_windows = _window_labels(labels[:common_length], window_samples)
    count = min(len(bvp_windows), len(ecg_windows), len(label_windows))
    return bvp_windows[:count], ecg_windows[:count], label_windows[:count]


def _refuse_nonempty_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.window_seconds != 4:
        raise ValueError("the frozen WESAD protocol requires four-second windows")
    _refuse_nonempty_output(args.output_dir)
    train_subjects, test_subjects = _subject_folds(args.fold_index, args.fold_seed)
    pseudonyms = {
        subject: f"S{index:03d}" for index, subject in enumerate(SUBJECTS, start=1)
    }
    split_arrays: dict[str, dict[str, list[np.ndarray]]] = {
        "train": {"bvp": [], "ecg": [], "labels": [], "subjects": []},
        "test": {"bvp": [], "ecg": [], "labels": [], "subjects": []},
    }
    windows_per_subject: dict[str, int] = {}
    window_samples = args.window_seconds * OUTPUT_RATE_HZ
    for subject in SUBJECTS:
        bvp, ecg, labels = _prepare_subject(args.source_root, subject, window_samples)
        split = "train" if subject in train_subjects else "test"
        split_arrays[split]["bvp"].append(bvp)
        split_arrays[split]["ecg"].append(ecg)
        split_arrays[split]["labels"].append(labels)
        split_arrays[split]["subjects"].append(
            np.repeat(pseudonyms[subject], len(bvp))
        )
        windows_per_subject[pseudonyms[subject]] = len(bvp)

    split_counts = {}
    label_counts = {}
    for split, values in split_arrays.items():
        bvp = np.concatenate(values["bvp"]).astype(np.float32)
        ecg = np.concatenate(values["ecg"]).astype(np.float32)
        labels = np.concatenate(values["labels"]).astype(np.int16)
        subjects = np.concatenate(values["subjects"])
        if bvp.shape != ecg.shape or len(labels) != len(bvp) or len(subjects) != len(bvp):
            raise AssertionError(f"{split} paired arrays or sidecars are misaligned")
        if not np.all(np.isfinite(bvp)) or not np.all(np.isfinite(ecg)):
            raise ValueError(f"{split} contains non-finite samples")
        if np.any(np.ptp(bvp, axis=1) <= 0) or np.any(np.ptp(ecg, axis=1) <= 0):
            raise ValueError(f"{split} contains constant windows")
        np.save(args.output_dir / f"ppg_{split}_4sec.npy", bvp)
        np.save(args.output_dir / f"ecg_{split}_4sec.npy", ecg)
        np.save(args.output_dir / f"labels_{split}.npy", labels)
        np.save(args.output_dir / f"subject_ids_{split}.npy", subjects)
        split_counts[split] = len(bvp)
        label_counts[split] = {
            str(label): int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        }

    assignment = {
        "train_subject_ids": sorted(pseudonyms[subject] for subject in train_subjects),
        "test_subject_ids": sorted(pseudonyms[subject] for subject in test_subjects),
    }
    split_hash = hashlib.sha256(
        json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    manifest = {
        "dataset": "WESAD",
        "dataset_version": "wesad-subject-fold1-linear-resample-window-minmax-v1",
        "source_release": "WESAD public dataset",
        "source_root": str(args.source_root),
        "subject_count": len(SUBJECTS),
        "split_method": "five_fold_KFold_on_subject_before_windowing",
        "fold_index_zero_based": args.fold_index,
        "fold_seed": args.fold_seed,
        "split_hash": split_hash,
        **assignment,
        "train_windows": split_counts["train"],
        "test_windows": split_counts["test"],
        "windows_per_subject": dict(sorted(windows_per_subject.items())),
        "label_counts": label_counts,
        "labels_retained": "all_source_labels_0_through_7",
        "condition": "wrist_BVP_64Hz_linearly_resampled_to_128Hz",
        "target": "chest_ECG_700Hz_linearly_resampled_to_128Hz",
        "alignment": "native_common_start_and_duration_same_window_boundaries_no_delay_correction",
        "window_seconds": args.window_seconds,
        "window_samples": window_samples,
        "window_overlap": 0.0,
        "normalization": "deferred_to_loader_window_minmax_neg1_1_v1",
        "signal_cleaning": "none",
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
