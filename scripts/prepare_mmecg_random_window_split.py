#!/usr/bin/env python3
"""Re-split all frozen mmECG RCG/ECG windows randomly 80:20."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


SOURCE_VERSION = "mmecg-public-20221108-subject-split-window-minmax-v1"
SOURCE_SPLIT_HASH = "e26fc81121cfd3b0e457608e37a7aa496ac7c48a0e0e7583a465bd069cc9da9f"
DATASET_VERSION = "mmecg-all-windows-random80-20-subject-record-overlap-window-minmax-v1"
NORMALIZATION_ID = "window_minmax_neg1_1_v1"
WINDOW_SAMPLES = 512
WINDOW_STEP_SAMPLES = 256
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


def _record_ordinals(source_files: np.ndarray) -> np.ndarray:
    counts: Counter[str] = Counter()
    ordinals = np.empty(len(source_files), dtype=np.int32)
    for index, source_file in enumerate(source_files.astype(str)):
        ordinals[index] = counts[source_file]
        counts[source_file] += 1
    return ordinals


def _split_hash(
    source_files: np.ndarray,
    ordinals: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> str:
    lines = []
    for split, indices in (("train", train_indices), ("test", test_indices)):
        for index in indices:
            lines.append(f"{source_files[index]}\t{int(ordinals[index])}\t{split}")
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()


def _load_source(
    source_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, str]]:
    manifest_path = source_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != SOURCE_VERSION:
        raise ValueError("source must be the frozen subject-split mmECG artifact")
    if manifest.get("split_hash") != SOURCE_SPLIT_HASH:
        raise ValueError("source mmECG subject-fold hash changed")
    if float(manifest.get("overlap", -1)) != 0.5:
        raise ValueError("source mmECG window overlap changed")
    arrays: dict[str, list[np.ndarray]] = {
        "rcg": [],
        "ecg": [],
        "subjects": [],
        "source_files": [],
        "source_split_codes": [],
        "source_rows": [],
    }
    source_hashes = {"dataset_manifest.json": _sha256(manifest_path)}
    for split_code, split in enumerate(SOURCE_SPLITS):
        paths = {
            "rcg": source_dir / f"ppg_{split}_4sec.npy",
            "ecg": source_dir / f"ecg_{split}_4sec.npy",
            "subjects": source_dir / f"subject_ids_{split}.npy",
            "source_files": source_dir / f"source_files_{split}.npy",
        }
        loaded = {key: np.load(path, allow_pickle=False) for key, path in paths.items()}
        rows = len(loaded["rcg"])
        if loaded["rcg"].shape != (rows, WINDOW_SAMPLES):
            raise ValueError(f"source {split} RCG shape changed")
        if loaded["ecg"].shape != loaded["rcg"].shape:
            raise ValueError(f"source {split} ECG/RCG pairing changed")
        if len(loaded["subjects"]) != rows or len(loaded["source_files"]) != rows:
            raise ValueError(f"source {split} sidecars do not align")
        if not np.all(np.isfinite(loaded["rcg"])) or not np.all(np.isfinite(loaded["ecg"])):
            raise ValueError(f"source {split} contains nonfinite waveforms")
        if np.any(np.ptp(loaded["rcg"], axis=1) <= 0) or np.any(
            np.ptp(loaded["ecg"], axis=1) <= 0
        ):
            raise ValueError(f"source {split} contains constant windows")
        for key in ("rcg", "ecg", "subjects", "source_files"):
            arrays[key].append(loaded[key])
            source_hashes[paths[key].name] = _sha256(paths[key])
        arrays["source_split_codes"].append(np.full(rows, split_code, dtype=np.uint8))
        arrays["source_rows"].append(np.arange(rows, dtype=np.int32))
    return {key: np.concatenate(parts) for key, parts in arrays.items()}, manifest, source_hashes


def _cross_split_adjacent_pairs(
    source_files: np.ndarray,
    ordinals: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> int:
    assignment = np.empty(len(source_files), dtype=np.uint8)
    assignment[train_indices] = 0
    assignment[test_indices] = 1
    lookup = {
        (str(source_file), int(ordinal)): int(assignment[index])
        for index, (source_file, ordinal) in enumerate(zip(source_files, ordinals))
    }
    return sum(
        lookup.get((source_file, ordinal + 1)) not in {None, split}
        for (source_file, ordinal), split in lookup.items()
    )


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
    ordinals = _record_ordinals(arrays["source_files"])
    identities = list(zip(arrays["source_files"].astype(str).tolist(), ordinals.tolist()))
    if len(set(identities)) != len(identities):
        raise ValueError("source record/window identities are not unique")
    train_indices, test_indices = random_window_membership(
        len(ordinals), validation_fraction=args.validation_fraction, seed=args.seed
    )
    summaries = {}
    output_hashes = {}
    for split, indices in (("train", train_indices), ("test", test_indices)):
        outputs = {
            f"ppg_{split}_4sec.npy": arrays["rcg"][indices].astype(np.float32, copy=False),
            f"ecg_{split}_4sec.npy": arrays["ecg"][indices].astype(np.float32, copy=False),
            f"subject_ids_{split}.npy": arrays["subjects"][indices],
            f"source_files_{split}.npy": arrays["source_files"][indices],
            f"source_record_window_ordinals_{split}.npy": ordinals[indices],
            f"source_split_codes_{split}.npy": arrays["source_split_codes"][indices],
            f"source_rows_{split}.npy": arrays["source_rows"][indices],
        }
        for name, values in outputs.items():
            path = output_dir / name
            np.save(path, values, allow_pickle=False)
            output_hashes[name] = _sha256(path)
        summaries[split] = {
            "windows": int(len(indices)),
            "unique_subjects": int(len(np.unique(outputs[f"subject_ids_{split}.npy"]))),
            "unique_source_records": int(len(np.unique(outputs[f"source_files_{split}.npy"]))),
        }

    train_subjects = set(
        np.load(output_dir / "subject_ids_train.npy", allow_pickle=False).astype(str).tolist()
    )
    test_subjects = set(
        np.load(output_dir / "subject_ids_test.npy", allow_pickle=False).astype(str).tolist()
    )
    train_records = set(
        np.load(output_dir / "source_files_train.npy", allow_pickle=False).astype(str).tolist()
    )
    test_records = set(
        np.load(output_dir / "source_files_test.npy", allow_pickle=False).astype(str).tolist()
    )
    cross_split_adjacent = _cross_split_adjacent_pairs(
        arrays["source_files"], ordinals, train_indices, test_indices
    )
    split_hash = _split_hash(arrays["source_files"], ordinals, train_indices, test_indices)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "dataset": "mmECG",
        "dataset_version": DATASET_VERSION,
        "source_dataset_version": source_manifest["dataset_version"],
        "source_split_hash": source_manifest["split_hash"],
        "source_file_sha256": dict(sorted(source_hashes.items())),
        "subject_count": int(len(np.unique(arrays["subjects"]))),
        "source_record_count": int(len(np.unique(arrays["source_files"]))),
        "source_windows": int(len(ordinals)),
        "split_method": "sklearn_train_test_split_over_all_windows_without_subject_or_record_grouping",
        "split_seed": args.seed,
        "train_fraction": 1.0 - args.validation_fraction,
        "validation_fraction": args.validation_fraction,
        "heldout_file_role": "test_files_used_as_training_validation_only",
        "split_hash": split_hash,
        "splits": summaries,
        "overlap": {
            "subject_disjoint": False,
            "subjects_in_both_train_and_test": len(train_subjects & test_subjects),
            "source_record_disjoint": False,
            "source_records_in_both_train_and_test": len(train_records & test_records),
            "cross_split_adjacent_window_pairs": int(cross_split_adjacent),
            "shared_raw_samples_per_cross_split_adjacent_pair": WINDOW_SAMPLES
            - WINDOW_STEP_SAMPLES,
            "same_subject_record_and_overlapping_raw_samples_can_cross_splits": True,
            "allowed_by_protocol": True,
        },
        "sampling_rate_hz": 128,
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": 4,
        "window_overlap": 0.5,
        "window_step_samples": WINDOW_STEP_SAMPLES,
        "condition": "energy-weighted 50-channel RCG stored under historical ppg_* filenames",
        "target": "single-channel ECG",
        "alignment_id": "same_record_same_window_no_delay_correction_random80_20_v1",
        "normalization": {
            "normalization_id": NORMALIZATION_ID,
            "model_input": "independent_per_window_per_modality_minmax_neg1_1",
            "application": "deferred_to_loader",
            "preserves_cross_modality_amplitude": False,
            "reason_not_shared": "RCG and ECG have different unverified sensor units",
        },
        "signal_cleaning": "none",
        "output_sha256": dict(sorted(output_hashes.items())),
        "claim_boundary": (
            "This non-grouped comparison permits subject, source-record, and raw-sample overlap "
            "between training and validation. It does not measure independent generalization."
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
