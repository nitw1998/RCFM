#!/usr/bin/env python3
"""Build WESAD random 80:20 windows with full-source-record min-max sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.prepare_wesad_random_window_split import (
    SOURCE_SPLIT_HASH,
    SOURCE_VERSION,
    WINDOW_SAMPLES,
    _load_source,
    _sha256,
    _split_hash,
    _window_ordinals,
    random_window_membership,
)


DATASET_VERSION = (
    "wesad-all-windows-random80-20-subject-overlap-linear-resample-"
    "source-record-minmax-v2"
)
NORMALIZATION_ID = "source_record_minmax_neg1_1_v1"


def source_record_coefficients(
    signals: np.ndarray, subjects: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat one full-record scalar minimum/range for every subject window."""

    signals = np.asarray(signals, dtype=np.float32)
    subjects = np.asarray(subjects).astype(str)
    if signals.ndim != 2 or len(signals) != len(subjects):
        raise ValueError("signals and subject identities must align by window")
    if not np.all(np.isfinite(signals)):
        raise ValueError("source records must contain only finite values")
    minima = np.empty(len(signals), dtype=np.float32)
    ranges = np.empty(len(signals), dtype=np.float32)
    for subject in np.unique(subjects):
        selected = subjects == subject
        record = signals[selected]
        minimum = np.float32(record.min())
        value_range = np.float32(record.max() - minimum)
        if not np.isfinite(value_range) or value_range <= 0:
            raise ValueError(f"source record {subject} has no finite positive range")
        minima[selected] = minimum
        ranges[selected] = value_range
    return minima, ranges


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

    target_minima, target_ranges = source_record_coefficients(
        arrays["ecg"], arrays["subjects"]
    )
    condition_minima, condition_ranges = source_record_coefficients(
        arrays["ppg"], arrays["subjects"]
    )
    train_indices, test_indices = random_window_membership(
        len(ordinals), validation_fraction=args.validation_fraction, seed=args.seed
    )

    summaries: dict[str, object] = {}
    output_hashes: dict[str, str] = {}
    for split, indices in (("train", train_indices), ("test", test_indices)):
        outputs = {
            f"ppg_{split}_4sec.npy": arrays["ppg"][indices].astype(np.float32, copy=False),
            f"ecg_{split}_4sec.npy": arrays["ecg"][indices].astype(np.float32, copy=False),
            f"labels_{split}.npy": arrays["labels"][indices].astype(np.int16, copy=False),
            f"subject_ids_{split}.npy": arrays["subjects"][indices],
            f"subject_window_ordinals_{split}.npy": ordinals[indices],
            f"source_split_codes_{split}.npy": arrays["source_split_codes"][indices],
            f"source_rows_{split}.npy": arrays["source_rows"][indices],
            f"target_record_minima_{split}.npy": target_minima[indices],
            f"target_record_ranges_{split}.npy": target_ranges[indices],
            f"condition_record_minima_{split}.npy": condition_minima[indices],
            f"condition_record_ranges_{split}.npy": condition_ranges[indices],
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

    train_subjects = set(arrays["subjects"][train_indices].astype(str).tolist())
    test_subjects = set(arrays["subjects"][test_indices].astype(str).tolist())
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
            "model_input": "independent_full_source_record_per_modality_minmax_neg1_1",
            "application": "deferred_to_loader_using_saved_window_aligned_sidecars",
            "record_definition": "one_subject_complete_retained_resampled_continuous_record",
            "computed_before_random_window_split": True,
            "same_coefficients_reused_for_all_subject_windows": True,
            "target_and_condition_scalers_shared": False,
            "preserves_within_record_interwindow_amplitude_and_offset": True,
            "heldout_target_statistics_used": True,
            "generated_inverse_policy": "ground_truth_target_scaler_is_oracle_only",
            "reason_not_shared": "BVP and ECG have different unverified sensor units",
        },
        "signal_cleaning": "none",
        "output_sha256": dict(sorted(output_hashes.items())),
        "claim_boundary": (
            "This non-grouped paired benchmark permits the same subject and continuous "
            "recording in training and validation. Full-record extrema are computed before "
            "the window split, so it does not measure preprocessing-independent or "
            "subject-independent generalization."
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
