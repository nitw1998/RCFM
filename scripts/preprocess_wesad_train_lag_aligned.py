"""Prepare WESAD BVP-to-ECG windows with a training-derived fixed lag.

The lag is calibrated exclusively from training subjects.  For each training
subject, ECG R peaks and wrist-BVP systolic peaks are detected on continuous
signals.  The subject estimate is the median delay from each R peak to the
first BVP peak in a physiological 80--500 ms interval; the applied dataset lag
is the median of the subject estimates.  The single lag is then applied to all
subjects before non-overlapping four-second windows are extracted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import neurokit2 as nk
import numpy as np

try:
    from scripts.preprocess_wesad import (
        OUTPUT_RATE_HZ,
        SUBJECTS,
        _load_subject,
        _refuse_nonempty_output,
        _resample_labels,
        _resample_linear,
        _subject_folds,
        _window_labels,
        _windows,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/<name>.py
    from preprocess_wesad import (
        OUTPUT_RATE_HZ,
        SUBJECTS,
        _load_subject,
        _refuse_nonempty_output,
        _resample_labels,
        _resample_linear,
        _subject_folds,
        _window_labels,
        _windows,
    )


DATASET_VERSION = "wesad-subject-fold1-train-fixed-lag-aligned-v2"
ALIGNMENT_ID = "train_subjects_peak_median_fixed_lag_crop_before_window_subject_fold1_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--window_seconds", type=int, default=4)
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--fold_seed", type=int, default=42)
    parser.add_argument("--minimum_delay_ms", type=float, default=80.0)
    parser.add_argument("--maximum_delay_ms", type=float, default=500.0)
    parser.add_argument("--minimum_matched_beats", type=int, default=100)
    return parser.parse_args()


def _continuous_subject(source_root: Path, subject: int) -> dict[str, np.ndarray]:
    raw = _load_subject(source_root, subject)
    bvp = _resample_linear(raw["bvp"], source_rate=64)
    ecg = _resample_linear(raw["ecg"], source_rate=700)
    labels = _resample_labels(raw["labels"])
    common = min(len(bvp), len(ecg), len(labels))
    if common < OUTPUT_RATE_HZ * 4:
        raise ValueError(f"S{subject} is shorter than one four-second window")
    return {
        "bvp": np.asarray(bvp[:common], dtype=np.float32),
        "ecg": np.asarray(ecg[:common], dtype=np.float32),
        "labels": np.asarray(labels[:common], dtype=np.int16),
    }


def _first_peak_delays(
    ecg_peaks: np.ndarray,
    bvp_peaks: np.ndarray,
    minimum_delay_samples: int,
    maximum_delay_samples: int,
) -> np.ndarray:
    """Return the first BVP peak after each R peak in the accepted interval."""
    if minimum_delay_samples <= 0 or maximum_delay_samples < minimum_delay_samples:
        raise ValueError("invalid delay interval")
    ecg_peaks = np.asarray(ecg_peaks, dtype=np.int64)
    bvp_peaks = np.asarray(bvp_peaks, dtype=np.int64)
    if np.any(np.diff(ecg_peaks) < 0) or np.any(np.diff(bvp_peaks) < 0):
        raise ValueError("peak indices must be sorted")
    delays: list[int] = []
    for peak in ecg_peaks:
        index = int(np.searchsorted(bvp_peaks, peak + minimum_delay_samples))
        if index < len(bvp_peaks):
            delay = int(bvp_peaks[index] - peak)
            if delay <= maximum_delay_samples:
                delays.append(delay)
    return np.asarray(delays, dtype=np.int32)


def _estimate_subject_lag(
    signals: dict[str, np.ndarray],
    minimum_delay_samples: int,
    maximum_delay_samples: int,
    minimum_matched_beats: int,
) -> dict[str, int | float]:
    ecg_clean = nk.ecg_clean(
        signals["ecg"], sampling_rate=OUTPUT_RATE_HZ, method="pantompkins1985"
    )
    bvp_clean = nk.ppg_clean(signals["bvp"], sampling_rate=OUTPUT_RATE_HZ)
    _, ecg_info = nk.ecg_peaks(
        ecg_clean,
        sampling_rate=OUTPUT_RATE_HZ,
        method="pantompkins1985",
        correct_artifacts=True,
        show=False,
    )
    _, bvp_info = nk.ppg_peaks(
        bvp_clean, sampling_rate=OUTPUT_RATE_HZ, method="elgendi", show=False
    )
    ecg_peaks = np.asarray(ecg_info.get("ECG_R_Peaks", []), dtype=np.int64)
    bvp_peaks = np.asarray(bvp_info.get("PPG_Peaks", []), dtype=np.int64)
    delays = _first_peak_delays(
        ecg_peaks, bvp_peaks, minimum_delay_samples, maximum_delay_samples
    )
    if len(delays) < minimum_matched_beats:
        raise ValueError(
            f"only {len(delays)} matched beats; need at least {minimum_matched_beats}"
        )
    return {
        "lag_samples": int(np.rint(np.median(delays))),
        "lag_ms": float(np.median(delays) * 1000.0 / OUTPUT_RATE_HZ),
        "matched_beats": int(len(delays)),
        "detected_ecg_peaks": int(len(ecg_peaks)),
        "detected_bvp_peaks": int(len(bvp_peaks)),
        "matched_fraction_of_ecg_peaks": float(len(delays) / max(1, len(ecg_peaks))),
        "delay_q10_samples": float(np.quantile(delays, 0.1)),
        "delay_q90_samples": float(np.quantile(delays, 0.9)),
    }


def _align_continuous(
    bvp: np.ndarray, ecg: np.ndarray, labels: np.ndarray, lag_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance delayed BVP without padding or circular wraparound."""
    if lag_samples <= 0:
        raise ValueError("the fixed BVP delay must be positive")
    common = min(len(bvp), len(ecg), len(labels))
    if common <= lag_samples:
        raise ValueError("lag leaves no common signal support")
    # BVP[t + lag] is paired with ECG[t]; labels remain on the ECG timeline.
    return (
        np.asarray(bvp[lag_samples:common], dtype=np.float32),
        np.asarray(ecg[: common - lag_samples], dtype=np.float32),
        np.asarray(labels[: common - lag_samples], dtype=np.int16),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.window_seconds != 4:
        raise ValueError("the frozen WESAD protocol requires four-second windows")
    minimum_delay_samples = int(np.ceil(args.minimum_delay_ms * OUTPUT_RATE_HZ / 1000.0))
    maximum_delay_samples = int(np.floor(args.maximum_delay_ms * OUTPUT_RATE_HZ / 1000.0))
    if minimum_delay_samples <= 0 or maximum_delay_samples < minimum_delay_samples:
        raise ValueError("the requested physiological delay interval is empty")
    _refuse_nonempty_output(args.output_dir)
    train_subjects, test_subjects = _subject_folds(args.fold_index, args.fold_seed)
    pseudonyms = {
        subject: f"S{index:03d}" for index, subject in enumerate(SUBJECTS, start=1)
    }

    # Calibration is a separate first pass over training subjects only.  Keeping
    # the resampled arrays avoids re-reading the large source pickles.
    cached_train: dict[int, dict[str, np.ndarray]] = {}
    calibration: dict[str, dict[str, int | float]] = {}
    for subject in sorted(train_subjects):
        signals = _continuous_subject(args.source_root, subject)
        cached_train[subject] = signals
        calibration[pseudonyms[subject]] = _estimate_subject_lag(
            signals,
            minimum_delay_samples,
            maximum_delay_samples,
            args.minimum_matched_beats,
        )
        print(
            f"calibrated {pseudonyms[subject]}: "
            f"{calibration[pseudonyms[subject]]['lag_samples']} samples",
            flush=True,
        )
    subject_lags = [int(row["lag_samples"]) for row in calibration.values()]
    fixed_lag = int(np.rint(np.median(subject_lags)))
    if not minimum_delay_samples <= fixed_lag <= maximum_delay_samples:
        raise AssertionError("aggregate lag falls outside the calibration interval")

    split_arrays: dict[str, dict[str, list[np.ndarray]]] = {
        "train": {"bvp": [], "ecg": [], "labels": [], "subjects": []},
        "test": {"bvp": [], "ecg": [], "labels": [], "subjects": []},
    }
    windows_per_subject: dict[str, int] = {}
    discarded_tail_samples: dict[str, int] = {}
    window_samples = args.window_seconds * OUTPUT_RATE_HZ
    for subject in SUBJECTS:
        signals = cached_train.pop(subject) if subject in cached_train else _continuous_subject(
            args.source_root, subject
        )
        bvp, ecg, labels = _align_continuous(
            signals["bvp"], signals["ecg"], signals["labels"], fixed_lag
        )
        bvp_windows = _windows(bvp, window_samples)
        ecg_windows = _windows(ecg, window_samples)
        label_windows = _window_labels(labels, window_samples)
        count = min(len(bvp_windows), len(ecg_windows), len(label_windows))
        split = "train" if subject in train_subjects else "test"
        split_arrays[split]["bvp"].append(bvp_windows[:count])
        split_arrays[split]["ecg"].append(ecg_windows[:count])
        split_arrays[split]["labels"].append(label_windows[:count])
        split_arrays[split]["subjects"].append(np.repeat(pseudonyms[subject], count))
        windows_per_subject[pseudonyms[subject]] = count
        discarded_tail_samples[pseudonyms[subject]] = int(len(bvp) - count * window_samples)

    split_counts: dict[str, int] = {}
    label_counts: dict[str, dict[str, int]] = {}
    output_hashes: dict[str, str] = {}
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
        outputs = {
            f"ppg_{split}_4sec.npy": bvp,
            f"ecg_{split}_4sec.npy": ecg,
            f"labels_{split}.npy": labels,
            f"subject_ids_{split}.npy": subjects,
        }
        for name, array in outputs.items():
            path = args.output_dir / name
            np.save(path, array, allow_pickle=False)
            output_hashes[name] = _sha256(path)
        split_counts[split] = int(len(bvp))
        label_counts[split] = {
            str(label): int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        }

    assignment = {
        "train_subject_ids": sorted(pseudonyms[s] for s in train_subjects),
        "test_subject_ids": sorted(pseudonyms[s] for s in test_subjects),
    }
    split_hash = hashlib.sha256(
        json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    manifest = {
        "schema_version": 2,
        "status": "completed",
        "dataset": "WESAD",
        "dataset_version": DATASET_VERSION,
        "source_release": "WESAD public dataset",
        "source_root_recorded": False,
        "subject_count": len(SUBJECTS),
        "split_method": "five_fold_KFold_on_subject_before_lag_calibration_and_windowing",
        "fold_index_zero_based": args.fold_index,
        "fold_seed": args.fold_seed,
        "split_hash": split_hash,
        **assignment,
        "train_windows": split_counts["train"],
        "test_windows": split_counts["test"],
        "windows_per_subject": dict(sorted(windows_per_subject.items())),
        "discarded_tail_samples_after_alignment": dict(sorted(discarded_tail_samples.items())),
        "label_counts": label_counts,
        "labels_retained": "all_source_labels_0_through_7",
        "condition": "wrist_BVP_64Hz_linearly_resampled_to_128Hz",
        "target": "chest_ECG_700Hz_linearly_resampled_to_128Hz",
        "alignment_id": ALIGNMENT_ID,
        "alignment": {
            "calibration_split": "training_subjects_only",
            "heldout_target_used_for_calibration": False,
            "method": "median_of_training_subject_median_R_to_first_BVP_peak_delays",
            "ecg_peak_detector": "NeuroKit2 Pan-Tompkins 1985 with artifact correction",
            "bvp_peak_detector": "NeuroKit2 Elgendi",
            "accepted_delay_samples": [minimum_delay_samples, maximum_delay_samples],
            "accepted_delay_ms_requested": [args.minimum_delay_ms, args.maximum_delay_ms],
            "fixed_lag_samples": fixed_lag,
            "fixed_lag_ms": fixed_lag * 1000.0 / OUTPUT_RATE_HZ,
            "application": "BVP advanced by fixed lag on continuous streams; valid crop; no padding or circular shift",
            "training_subject_calibration": dict(sorted(calibration.items())),
        },
        "window_seconds": args.window_seconds,
        "window_samples": window_samples,
        "window_overlap": 0.0,
        "normalization": "deferred_to_loader_window_minmax_neg1_1_v1",
        "signal_cleaning_for_saved_waveforms": "none; cleaning used only for peak calibration",
        "output_sha256": dict(sorted(output_hashes.items())),
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
