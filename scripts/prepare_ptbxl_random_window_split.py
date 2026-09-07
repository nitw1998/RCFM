#!/usr/bin/env python3
"""Create a PTB-XL random 80:20 four-second-window comparison artifact.

Every physical-mV 10-second record contributes windows 0--4 s and 4--8 s.
Windows are pooled across all official source splits and randomly divided at
window level, so patients and source records are intentionally allowed to
occur in both output splits. Each source 10-second record uses one min/range
shared by all 12 leads and both extracted windows; the physical-mV arrays
remain stored for reversible loading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


SOURCE_SPLITS = ("train", "val", "test")
WINDOW_SAMPLES = 512
WINDOW_STARTS = (0, 512)
SOURCE_RECORD_SAMPLES = 1280
LEADS = 12
DATASET_VERSION = (
    "ptbxl-1.0.1-random-window80-20-record-overlap-"
    "source-record-joint12-full10s-minmax-neg1-1-v2"
)
NORMALIZATION_ID = "source_record_joint12_minmax_neg1_1_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def random_window_membership(
    records: int, *, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/validation indices over two windows per record."""

    if records <= 0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("records and validation_fraction must define a nonempty split")
    indices = np.arange(records * len(WINDOW_STARTS), dtype=np.int64)
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
    )
    if len(np.intersect1d(train_indices, validation_indices)):
        raise AssertionError("random window splits overlap by window identity")
    if len(train_indices) + len(validation_indices) != len(indices):
        raise AssertionError("random window splits do not cover all windows")
    return np.asarray(train_indices), np.asarray(validation_indices)


def _split_hash(
    record_ids: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> str:
    lines: list[str] = []
    for split, indices in (("train", train_indices), ("val", validation_indices)):
        for window_index in indices:
            record_index, within_record = divmod(int(window_index), len(WINDOW_STARTS))
            lines.append(
                f"{record_ids[record_index]}\t{WINDOW_STARTS[within_record]}\t{split}"
            )
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()


def _load_source(source_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    manifest_path = source_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("source PTB-XL manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_version = "ptbxl-1.0.1-official-folds-record-joint12-minmax-neg1-1-v1"
    if manifest.get("dataset_version") != expected_version:
        raise ValueError("source artifact must be the physical-mV official joint12 PTB-XL artifact")
    if manifest.get("stored_waveforms", {}).get("physical_unit") != "mV":
        raise ValueError("source PTB-XL arrays must remain in physical mV")
    arrays: dict[str, np.ndarray] = {}
    sidecars: dict[str, list[np.ndarray]] = {
        "record_ids": [],
        "patient_ids": [],
        "source_split_codes": [],
        "source_local_rows": [],
        "source_record_joint_minima": [],
        "source_record_joint_ranges": [],
    }
    for code, split in enumerate(SOURCE_SPLITS):
        array = np.load(source_dir / f"X_{split}_resampled.npy", mmap_mode="r", allow_pickle=False)
        record_ids = np.load(source_dir / f"record_ids_{split}.npy", allow_pickle=False).reshape(-1)
        patient_ids = np.load(source_dir / f"patient_ids_{split}.npy", allow_pickle=False).reshape(-1)
        if (
            array.ndim != 3
            or array.shape[1] != SOURCE_RECORD_SAMPLES
            or array.shape[2] != LEADS
        ):
            raise ValueError(f"source {split} waveform array has the wrong shape")
        if len(array) != len(record_ids) or len(array) != len(patient_ids):
            raise ValueError(f"source {split} identity sidecars do not align")
        arrays[split] = array
        sidecars["record_ids"].append(record_ids)
        sidecars["patient_ids"].append(patient_ids)
        sidecars["source_split_codes"].append(np.full(len(array), code, dtype=np.uint8))
        sidecars["source_local_rows"].append(np.arange(len(array), dtype=np.int32))
        minima = np.empty(len(array), dtype=np.float32)
        ranges = np.empty(len(array), dtype=np.float32)
        for start in range(0, len(array), 512):
            stop = min(start + 512, len(array))
            records = np.asarray(array[start:stop], dtype=np.float32)
            chunk_minima = records.min(axis=(1, 2))
            minima[start:stop] = chunk_minima
            ranges[start:stop] = records.max(axis=(1, 2)) - chunk_minima
        if not np.all(np.isfinite(minima)) or not np.all(np.isfinite(ranges)):
            raise ValueError(f"source {split} contains nonfinite joint 12-lead coefficients")
        sidecars["source_record_joint_minima"].append(minima)
        sidecars["source_record_joint_ranges"].append(ranges)
    concatenated = {key: np.concatenate(values) for key, values in sidecars.items()}
    if len(np.unique(concatenated["record_ids"])) != len(concatenated["record_ids"]):
        raise ValueError("source ECG record IDs must be globally unique")
    return arrays, concatenated, manifest


def _write_split(
    output_dir: Path,
    split: str,
    membership: np.ndarray,
    source_arrays: dict[str, np.ndarray],
    source_sidecars: dict[str, np.ndarray],
    minimum_range_mv: float,
) -> dict[str, object]:
    building = output_dir / f".X_{split}_resampled.npy.building"
    final = output_dir / f"X_{split}_resampled.npy"
    waveforms = np.lib.format.open_memmap(
        building, mode="w+", dtype=np.float32, shape=(len(membership), WINDOW_SAMPLES, LEADS)
    )
    record_ids = np.empty(len(membership), dtype=source_sidecars["record_ids"].dtype)
    patient_ids = np.empty(len(membership), dtype=source_sidecars["patient_ids"].dtype)
    source_split_codes = np.empty(len(membership), dtype=np.uint8)
    source_local_rows = np.empty(len(membership), dtype=np.int32)
    starts = np.empty(len(membership), dtype=np.uint16)
    minima = np.empty(len(membership), dtype=np.float32)
    ranges = np.empty(len(membership), dtype=np.float32)

    for output_index, window_index in enumerate(membership):
        record_index, within_record = divmod(int(window_index), len(WINDOW_STARTS))
        start = WINDOW_STARTS[within_record]
        source_code = int(source_sidecars["source_split_codes"][record_index])
        source_split = SOURCE_SPLITS[source_code]
        source_row = int(source_sidecars["source_local_rows"][record_index])
        window = np.asarray(
            source_arrays[source_split][source_row, start : start + WINDOW_SAMPLES],
            dtype=np.float32,
        )
        if window.shape != (WINDOW_SAMPLES, LEADS) or not np.all(np.isfinite(window)):
            raise ValueError("source window is nonfinite or malformed")
        offset = float(source_sidecars["source_record_joint_minima"][record_index])
        value_range = float(source_sidecars["source_record_joint_ranges"][record_index])
        if value_range < minimum_range_mv or not np.isfinite(value_range):
            raise ValueError("source 10-second record has insufficient joint 12-lead range")
        waveforms[output_index] = window
        record_ids[output_index] = source_sidecars["record_ids"][record_index]
        patient_ids[output_index] = source_sidecars["patient_ids"][record_index]
        source_split_codes[output_index] = source_code
        source_local_rows[output_index] = source_row
        starts[output_index] = start
        minima[output_index] = offset
        ranges[output_index] = value_range
        if (output_index + 1) % 5000 == 0 or output_index + 1 == len(membership):
            print(f"{split}: wrote {output_index + 1}/{len(membership)} windows", flush=True)
    waveforms.flush()
    del waveforms
    os.replace(building, final)

    sidecar_arrays = {
        f"record_ids_{split}.npy": record_ids,
        f"patient_ids_{split}.npy": patient_ids,
        f"source_split_codes_{split}.npy": source_split_codes,
        f"source_local_rows_{split}.npy": source_local_rows,
        f"window_start_samples_{split}.npy": starts,
        f"record_joint_minima_{split}.npy": minima,
        f"record_joint_ranges_{split}.npy": ranges,
    }
    for name, values in sidecar_arrays.items():
        np.save(output_dir / name, values, allow_pickle=False)
    return {
        "windows": int(len(membership)),
        "unique_records": int(len(np.unique(record_ids))),
        "unique_patients": int(len(np.unique(patient_ids))),
        "window_start_counts": {
            str(start): int(np.sum(starts == start)) for start in WINDOW_STARTS
        },
    }


def run(args: argparse.Namespace) -> Path:
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    repository_root = Path(__file__).resolve().parents[1]
    if output_dir == repository_root or repository_root in output_dir.parents:
        raise ValueError("preprocessed datasets must be written outside the Git repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_arrays, source_sidecars, source_manifest = _load_source(source_dir)
    records = len(source_sidecars["record_ids"])
    train_indices, validation_indices = random_window_membership(
        records, validation_fraction=args.validation_fraction, seed=args.seed
    )
    summaries = {
        "train": _write_split(
            output_dir, "train", train_indices, source_arrays, source_sidecars, args.minimum_range_mv
        ),
        "val": _write_split(
            output_dir, "val", validation_indices, source_arrays, source_sidecars, args.minimum_range_mv
        ),
    }

    train_records = set(np.load(output_dir / "record_ids_train.npy", allow_pickle=False).tolist())
    val_records = set(np.load(output_dir / "record_ids_val.npy", allow_pickle=False).tolist())
    train_patients = set(np.load(output_dir / "patient_ids_train.npy", allow_pickle=False).tolist())
    val_patients = set(np.load(output_dir / "patient_ids_val.npy", allow_pickle=False).tolist())
    output_files = sorted(path for path in output_dir.iterdir() if path.is_file())
    output_hashes = {path.name: _sha256(path) for path in output_files}
    split_hash = _split_hash(source_sidecars["record_ids"], train_indices, validation_indices)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "dataset": "PTB-XL",
        "dataset_version": DATASET_VERSION,
        "source_dataset_version": source_manifest["dataset_version"],
        "source_split_hash": source_manifest["split_hash"],
        "source_manifest_sha256": _sha256(source_dir / "dataset_manifest.json"),
        "source_records": records,
        "source_patients": int(len(np.unique(source_sidecars["patient_ids"]))),
        "windowing": {
            "sampling_rate_hz": 128,
            "window_samples": WINDOW_SAMPLES,
            "window_seconds": 4,
            "starts_samples": list(WINDOW_STARTS),
            "starts_seconds": [0, 4],
            "unused_tail_seconds": [8, 10],
            "windows_per_record": 2,
        },
        "split_method": "sklearn_train_test_split_over_all_windows_without_patient_or_record_grouping",
        "split_seed": args.seed,
        "train_fraction": 1.0 - args.validation_fraction,
        "validation_fraction": args.validation_fraction,
        "split_hash": split_hash,
        "splits": summaries,
        "overlap": {
            "record_disjoint": False,
            "patient_disjoint": False,
            "records_in_both_train_and_val": int(len(train_records & val_records)),
            "patients_in_both_train_and_val": int(len(train_patients & val_patients)),
            "allowed_by_protocol": True,
        },
        "task": {
            "condition_lead": "II",
            "condition_lead_index": 1,
            "target_leads": ["I", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            "target_lead_indices": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        "normalization": {
            "normalization_id": NORMALIZATION_ID,
            "model_input": "source_record_joint_12lead_minmax_neg1_1",
            "formula": "x_scaled=2*(x-source_record_joint_min)/source_record_joint_range-1",
            "coefficient_scope": "each_full_10_second_source_record_all_time_and_all_12_leads",
            "coefficient_files": "record_joint_minima_{split}.npy and record_joint_ranges_{split}.npy",
            "preserves_interlead_relative_amplitudes_and_offsets": True,
            "preserves_within_record_interwindow_scale": True,
            "heldout_target_statistics_used": True,
            "deployment_boundary": "paired benchmark only; held-out scale uses the 11 target leads",
        },
        "stored_waveforms": {
            "format": "X_{train,val}_resampled.npy",
            "layout": ["windows", "samples", "leads"],
            "dtype": "float32",
            "physical_unit": "mV",
        },
        "output_sha256": output_hashes,
        "claim_boundary": (
            "This intentionally non-grouped comparison permits patient and source-record overlap. "
            "It does not measure patient-independent or record-independent generalization."
        ),
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"split_hash": split_hash, "splits": summaries, "overlap": manifest["overlap"]}, indent=2))
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--minimum_range_mv", type=float, default=1e-6)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
