#!/usr/bin/env python3
"""Re-split all QC-retained MIMIC-AFib windows randomly 80:20."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


SOURCE_VERSION = "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1"
SOURCE_SPLIT_HASH = "a7e388293adaa7b48d3493efc505dd8750520730cae9fd7649157866efa86a51"
DATASET_VERSION = (
    "mimic-afib-all-qc-windows-random80-20-subject-record-overlap-"
    "rddm-window-minmax-v1"
)
NORMALIZATION_ID = "rddm_window_minmax_neg1_1_v1"
ALIGNMENT_ID = "paired_source_row_no_phase_correction_random80_20_v1"
WINDOW_SAMPLES = 512
SOURCE_SPLITS = ("train", "test")
IDENTITY_FIELDS = (
    "subject_ids",
    "record_ids",
    "source_record_names",
    "window_indices",
    "start_samples_128hz",
    "afib_labels",
)


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
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
    )
    if len(np.intersect1d(train_indices, validation_indices)):
        raise AssertionError("random splits overlap by window identity")
    if len(train_indices) + len(validation_indices) != windows:
        raise AssertionError("random splits do not cover all windows")
    return np.asarray(train_indices), np.asarray(validation_indices)


def _split_hash(
    record_ids: np.ndarray,
    starts: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> str:
    lines = []
    for split, indices in (("train", train_indices), ("test", validation_indices)):
        for index in indices:
            lines.append(f"{record_ids[index]}\t{int(starts[index])}\t{split}")
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()


def _load_source(
    source_dir: Path, identity_dir: Path
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, str]]:
    manifest_path = source_dir / "dataset_manifest.json"
    identity_manifest_path = identity_dir / "identity_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity_manifest = json.loads(identity_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != SOURCE_VERSION:
        raise ValueError("source must be the frozen all-zero-PPG-QC MIMIC-AFib artifact")
    if manifest.get("split_membership_hash") != SOURCE_SPLIT_HASH:
        raise ValueError("source MIMIC-AFib split hash changed")
    if identity_manifest.get("status") != "completed":
        raise ValueError("MIMIC-AFib identity artifact is incomplete")
    if identity_manifest.get("qc_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("identity sidecars do not match the source QC manifest")
    if not identity_manifest.get("subject_disjoint_verified"):
        raise ValueError("source identity artifact must verify its original subject split")

    arrays: dict[str, list[np.ndarray]] = {
        "ppg": [],
        "ecg": [],
        **{field: [] for field in IDENTITY_FIELDS},
        "source_split_codes": [],
        "source_rows": [],
    }
    source_hashes = {
        "dataset_manifest.json": _sha256(manifest_path),
        "identity_manifest.json": _sha256(identity_manifest_path),
    }
    for split_code, split in enumerate(SOURCE_SPLITS):
        paths = {
            "ppg": source_dir / f"ppg_{split}_4sec.npy",
            "ecg": source_dir / f"ecg_{split}_4sec.npy",
            **{field: identity_dir / f"{field}_{split}.npy" for field in IDENTITY_FIELDS},
        }
        loaded = {key: np.load(path, allow_pickle=False) for key, path in paths.items()}
        rows = len(loaded["ppg"])
        if loaded["ppg"].shape != (rows, WINDOW_SAMPLES):
            raise ValueError(f"source {split} PPG shape changed")
        if loaded["ecg"].shape != loaded["ppg"].shape:
            raise ValueError(f"source {split} ECG/PPG pairing changed")
        if any(len(loaded[field]) != rows for field in IDENTITY_FIELDS):
            raise ValueError(f"source {split} identity sidecars do not align")
        if not np.all(np.isfinite(loaded["ppg"])) or not np.all(np.isfinite(loaded["ecg"])):
            raise ValueError(f"source {split} contains nonfinite waveforms")
        if np.any(np.ptp(loaded["ppg"], axis=1) <= 0) or np.any(
            np.ptp(loaded["ecg"], axis=1) <= 0
        ):
            raise ValueError(f"source {split} contains constant windows")
        for key, path in paths.items():
            arrays[key].append(loaded[key])
            source_hashes[f"{split}/{path.name}"] = _sha256(path)
        arrays["source_split_codes"].append(np.full(rows, split_code, dtype=np.uint8))
        arrays["source_rows"].append(np.arange(rows, dtype=np.int32))
    combined = {key: np.concatenate(parts) for key, parts in arrays.items()}
    identities = list(
        zip(
            combined["record_ids"].astype(str).tolist(),
            combined["start_samples_128hz"].astype(int).tolist(),
        )
    )
    if len(set(identities)) != len(identities):
        raise ValueError("source record/start window identities are not unique")
    return combined, manifest, source_hashes


def run(args: argparse.Namespace) -> Path:
    source_dir = args.source_dir.resolve()
    identity_dir = args.identity_dir.resolve()
    output_dir = args.output_dir.resolve()
    repository_root = Path(__file__).resolve().parents[1]
    if output_dir == repository_root or repository_root in output_dir.parents:
        raise ValueError("preprocessed datasets must be written outside the Git repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays, source_manifest, source_hashes = _load_source(source_dir, identity_dir)
    train_indices, validation_indices = random_window_membership(
        len(arrays["ecg"]), validation_fraction=args.validation_fraction, seed=args.seed
    )
    summaries: dict[str, object] = {}
    output_hashes: dict[str, str] = {}
    for split, indices in (("train", train_indices), ("test", validation_indices)):
        outputs = {
            f"ppg_{split}_4sec.npy": arrays["ppg"][indices],
            f"ecg_{split}_4sec.npy": arrays["ecg"][indices],
            **{f"{field}_{split}.npy": arrays[field][indices] for field in IDENTITY_FIELDS},
            f"source_split_codes_{split}.npy": arrays["source_split_codes"][indices],
            f"source_rows_{split}.npy": arrays["source_rows"][indices],
        }
        for name, values in outputs.items():
            path = output_dir / name
            np.save(path, values, allow_pickle=False)
            output_hashes[name] = _sha256(path)
        labels = np.asarray(outputs[f"afib_labels_{split}.npy"], dtype=bool)
        summaries[split] = {
            "windows": int(len(indices)),
            "unique_subjects": int(len(np.unique(outputs[f"subject_ids_{split}.npy"]))),
            "unique_records": int(len(np.unique(outputs[f"record_ids_{split}.npy"]))),
            "afib_windows": int(np.sum(labels)),
            "non_afib_windows": int(np.sum(~labels)),
        }

    train_subjects = set(arrays["subject_ids"][train_indices].astype(str).tolist())
    validation_subjects = set(arrays["subject_ids"][validation_indices].astype(str).tolist())
    train_records = set(arrays["record_ids"][train_indices].astype(str).tolist())
    validation_records = set(arrays["record_ids"][validation_indices].astype(str).tolist())
    split_hash = _split_hash(
        arrays["record_ids"].astype(str),
        arrays["start_samples_128hz"],
        train_indices,
        validation_indices,
    )
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "dataset": "MIMIC-AFib",
        "dataset_version": DATASET_VERSION,
        "source_dataset_version": source_manifest["dataset_version"],
        "source_split_hash": source_manifest["split_membership_hash"],
        "source_file_sha256": dict(sorted(source_hashes.items())),
        "source_windows": int(len(arrays["ecg"])),
        "source_subjects": int(len(np.unique(arrays["subject_ids"]))),
        "source_records": int(len(np.unique(arrays["record_ids"]))),
        "split_method": "sklearn_train_test_split_over_all_qc_windows_without_subject_or_record_grouping",
        "split_seed": int(args.seed),
        "train_fraction": 1.0 - float(args.validation_fraction),
        "validation_fraction": float(args.validation_fraction),
        "heldout_file_role": "test_files_used_as_training_validation_only",
        "split_hash": split_hash,
        "splits": summaries,
        "overlap": {
            "subject_disjoint": False,
            "record_disjoint": False,
            "subjects_in_both_train_and_test": len(train_subjects & validation_subjects),
            "records_in_both_train_and_test": len(train_records & validation_records),
            "raw_samples_overlap_across_splits": False,
            "same_continuous_record_can_cross_splits": True,
            "allowed_by_protocol": True,
        },
        "condition": "MIMIC_PERform_PPG_upstream_array_channel",
        "target": "MIMIC_PERform_ECG_upstream_array_channel",
        "condition_unit": "upstream_normalized_unit_unverified",
        "target_unit": "upstream_normalized_unit_unverified",
        "alignment_id": ALIGNMENT_ID,
        "alignment": "paired source-array row; same four-second boundaries; no phase correction",
        "window_seconds": 4,
        "window_samples": WINDOW_SAMPLES,
        "window_overlap": 0.0,
        "normalization": {
            "normalization_id": NORMALIZATION_ID,
            "model_input": "independent_per_window_per_modality_minmax_neg1_1_then_neurokit_clean",
            "application": "deferred_to_loader",
            "preserves_cross_modality_amplitude": False,
            "inverse_transform": "unavailable_after_per_window_scaling_and_cleaning",
        },
        "source_qc": "all-zero PPG windows and paired ECG rows removed before pooling",
        "identifiers": "deidentified sidecars retained only in the run artifact and not for Git",
        "output_sha256": dict(sorted(output_hashes.items())),
        "claim_boundary": (
            "This non-grouped comparison permits every subject and source record in both "
            "training and validation. It does not measure patient- or record-independent "
            "generalization. The held-out files are validation, not a final test set."
        ),
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"split_hash": split_hash, "splits": summaries, "overlap": manifest["overlap"]}, indent=2))
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--identity_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=31)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
