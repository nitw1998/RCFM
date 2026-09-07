#!/usr/bin/env python3
"""Create the CPSC2018 full-record joint12 random-window 80:20 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.preprocess_cpsc2018 import (  # noqa: E402
    LABEL_NAMES,
    LEAD_ORDER,
    _inventory_hash,
    _sha256,
    clinical_policy,
    discover_record_paths,
    load_reference,
)


DATASET_VERSION = (
    "cpsc2018-source-fullrecord-joint12-minmax-"
    "all-nonoverlap4s-random80-20-record-overlap-v1"
)
NORMALIZATION_ID = "source_record_joint12_minmax_neg1_1_v1"
WINDOW_SECONDS = 4
LEADS = 12


def _load_resampled_record(
    path: Path, *, source_rate_hz: int, output_rate_hz: int
) -> np.ndarray:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=["ECG"])
    if "ECG" not in payload or not hasattr(payload["ECG"], "data"):
        raise ValueError(f"{path.name} does not contain ECG.data")
    signal = np.asarray(payload["ECG"].data, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] != LEADS:
        raise ValueError(f"{path.name} must contain 12 leads x samples")
    if signal.shape[1] < source_rate_hz * WINDOW_SECONDS or not np.all(np.isfinite(signal)):
        raise ValueError(f"{path.name} is too short, nonfinite, or malformed")
    resampled = resample_poly(
        signal, output_rate_hz, source_rate_hz, axis=1, padtype="line"
    ).T
    values = np.asarray(resampled, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != LEADS or not np.all(np.isfinite(values)):
        raise ValueError(f"resampled {path.name} is nonfinite or malformed")
    return values


def random_window_membership(
    total_windows: int, *, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if total_windows <= 1 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("total_windows and validation_fraction must define a nonempty split")
    indices = np.arange(total_windows, dtype=np.int64)
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
    )
    if len(np.intersect1d(train_indices, validation_indices)):
        raise AssertionError("random window splits overlap by window identity")
    if len(train_indices) + len(validation_indices) != total_windows:
        raise AssertionError("random window splits do not cover all windows")
    return np.asarray(train_indices), np.asarray(validation_indices)


def _split_hash(
    record_ids: np.ndarray,
    record_indices: np.ndarray,
    starts: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> str:
    lines = []
    for split, indices in (("train", train_indices), ("val", validation_indices)):
        for index in indices:
            lines.append(
                f"{record_ids[record_indices[index]]}\t{int(starts[index])}\t{split}"
            )
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()


def _scan_records(
    rows: list[dict[str, object]],
    paths: dict[str, Path],
    *,
    source_rate_hz: int,
    output_rate_hz: int,
    minimum_range: float,
) -> dict[str, np.ndarray]:
    window_samples = output_rate_hz * WINDOW_SECONDS
    record_ids: list[str] = []
    source_rows: list[int] = []
    lengths: list[int] = []
    counts: list[int] = []
    minima: list[float] = []
    ranges: list[float] = []
    excluded: list[str] = []
    for source_row, row in enumerate(rows):
        record_id = str(row["record_id"])
        values = _load_resampled_record(
            paths[record_id], source_rate_hz=source_rate_hz, output_rate_hz=output_rate_hz
        )
        offset = float(np.min(values))
        value_range = float(np.max(values) - offset)
        count = len(values) // window_samples
        if count < 1 or not np.isfinite(value_range) or value_range < minimum_range:
            excluded.append(record_id)
        else:
            record_ids.append(record_id)
            source_rows.append(source_row)
            lengths.append(len(values))
            counts.append(count)
            minima.append(offset)
            ranges.append(value_range)
        if (source_row + 1) % 500 == 0 or source_row + 1 == len(rows):
            print(f"scan: processed {source_row + 1}/{len(rows)} records", flush=True)
    if excluded:
        raise ValueError(
            "full-record joint12 QC excluded records; review before changing the cohort: "
            + ",".join(excluded[:10])
        )
    return {
        "record_ids": np.asarray(record_ids, dtype="U5"),
        "source_rows": np.asarray(source_rows, dtype=np.int32),
        "lengths": np.asarray(lengths, dtype=np.int32),
        "window_counts": np.asarray(counts, dtype=np.int16),
        "minima": np.asarray(minima, dtype=np.float32),
        "ranges": np.asarray(ranges, dtype=np.float32),
    }


def _open_outputs(output_dir: Path, split: str, rows: int, window_samples: int) -> dict[str, np.ndarray]:
    building = output_dir / f".X_{split}_resampled.npy.building"
    waveforms = np.lib.format.open_memmap(
        building, mode="w+", dtype=np.float32, shape=(rows, window_samples, LEADS)
    )
    return {
        "waveforms": waveforms,
        "record_ids": np.empty(rows, dtype="U5"),
        "source_rows": np.empty(rows, dtype=np.int32),
        "window_ordinals": np.empty(rows, dtype=np.int16),
        "starts": np.empty(rows, dtype=np.int32),
        "source_lengths": np.empty(rows, dtype=np.int32),
        "minima": np.empty(rows, dtype=np.float32),
        "ranges": np.empty(rows, dtype=np.float32),
        "labels": np.zeros((rows, len(LABEL_NAMES)), dtype=np.uint8),
    }


def _save_outputs(output_dir: Path, split: str, output: dict[str, np.ndarray]) -> None:
    waveforms = output.pop("waveforms")
    waveforms.flush()
    del waveforms
    os.replace(
        output_dir / f".X_{split}_resampled.npy.building",
        output_dir / f"X_{split}_resampled.npy",
    )
    names = {
        "record_ids": f"record_ids_{split}.npy",
        "source_rows": f"source_record_rows_{split}.npy",
        "window_ordinals": f"window_ordinals_{split}.npy",
        "starts": f"window_start_samples_{split}.npy",
        "source_lengths": f"source_record_resampled_lengths_{split}.npy",
        "minima": f"record_joint_minima_{split}.npy",
        "ranges": f"record_joint_ranges_{split}.npy",
        "labels": f"labels_{split}.npy",
    }
    for key, name in names.items():
        np.save(output_dir / name, output[key], allow_pickle=False)


def run(args: argparse.Namespace) -> Path:
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == REPOSITORY_ROOT or REPOSITORY_ROOT in output_dir.parents:
        raise ValueError("preprocessed datasets must be written outside the Git repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = source_root / "REFERENCE.csv"
    rows = load_reference(reference_path)
    paths = discover_record_paths(source_root)
    if {str(row["record_id"]) for row in rows} != set(paths):
        raise ValueError("CPSC source files and reference rows do not match")
    scanned = _scan_records(
        rows,
        paths,
        source_rate_hz=args.source_rate,
        output_rate_hz=args.output_rate,
        minimum_range=args.minimum_range,
    )
    window_samples = args.output_rate * WINDOW_SECONDS
    record_indices = np.repeat(
        np.arange(len(rows), dtype=np.int32), scanned["window_counts"].astype(np.int64)
    )
    starts = np.concatenate(
        [np.arange(count, dtype=np.int32) * window_samples for count in scanned["window_counts"]]
    )
    total_windows = len(record_indices)
    train_indices, validation_indices = random_window_membership(
        total_windows, validation_fraction=args.validation_fraction, seed=args.seed
    )
    destination_split = np.empty(total_windows, dtype=np.uint8)
    destination_row = np.empty(total_windows, dtype=np.int32)
    destination_split[train_indices] = 0
    destination_split[validation_indices] = 1
    destination_row[train_indices] = np.arange(len(train_indices), dtype=np.int32)
    destination_row[validation_indices] = np.arange(len(validation_indices), dtype=np.int32)
    outputs = {
        "train": _open_outputs(output_dir, "train", len(train_indices), window_samples),
        "val": _open_outputs(output_dir, "val", len(validation_indices), window_samples),
    }

    cursor = 0
    for record_index, source_row in enumerate(scanned["source_rows"]):
        row = rows[int(source_row)]
        record_id = str(row["record_id"])
        values = _load_resampled_record(
            paths[record_id], source_rate_hz=args.source_rate, output_rate_hz=args.output_rate
        )
        if len(values) != int(scanned["lengths"][record_index]):
            raise AssertionError("CPSC source record changed between scan and write")
        label_vector = np.zeros(len(LABEL_NAMES), dtype=np.uint8)
        for label in row["labels"]:
            label_vector[int(label) - 1] = 1
        for ordinal in range(int(scanned["window_counts"][record_index])):
            global_index = cursor + ordinal
            split = ("train", "val")[int(destination_split[global_index])]
            output_row = int(destination_row[global_index])
            start = ordinal * window_samples
            output = outputs[split]
            output["waveforms"][output_row] = values[start : start + window_samples]
            output["record_ids"][output_row] = record_id
            output["source_rows"][output_row] = source_row
            output["window_ordinals"][output_row] = ordinal
            output["starts"][output_row] = start
            output["source_lengths"][output_row] = len(values)
            output["minima"][output_row] = scanned["minima"][record_index]
            output["ranges"][output_row] = scanned["ranges"][record_index]
            output["labels"][output_row] = label_vector
        cursor += int(scanned["window_counts"][record_index])
        if (record_index + 1) % 500 == 0 or record_index + 1 == len(rows):
            print(f"write: processed {record_index + 1}/{len(rows)} records", flush=True)
    if cursor != total_windows:
        raise AssertionError("window write count does not match the scan")
    for split, output in outputs.items():
        _save_outputs(output_dir, split, output)

    train_ids = np.load(output_dir / "record_ids_train.npy", allow_pickle=False)
    val_ids = np.load(output_dir / "record_ids_val.npy", allow_pickle=False)
    train_labels = np.load(output_dir / "labels_train.npy", allow_pickle=False)
    val_labels = np.load(output_dir / "labels_val.npy", allow_pickle=False)
    split_hash = _split_hash(
        scanned["record_ids"], record_indices, starts, train_indices, validation_indices
    )
    output_hashes = {
        path.name: _sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()
    }
    split_summaries = {}
    for split, ids, labels in (
        ("train", train_ids, train_labels), ("val", val_ids, val_labels)
    ):
        primary_counts = Counter(
            int(rows[int(source_row)]["first_label"])
            for source_row in np.load(
                output_dir / f"source_record_rows_{split}.npy", allow_pickle=False
            )
        )
        split_summaries[split] = {
            "windows": int(len(ids)),
            "unique_records": int(len(np.unique(ids))),
            "primary_label_window_counts": {
                str(key): int(primary_counts[key]) for key in sorted(primary_counts)
            },
            "all_label_window_counts": labels.sum(axis=0).astype(int).tolist(),
        }
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "dataset": "CPSC2018",
        "dataset_version": DATASET_VERSION,
        "source_reference_sha256": _sha256(reference_path),
        "source_path_size_inventory_sha256": _inventory_hash(paths, source_root),
        "source_records": len(rows),
        "eligible_records": len(scanned["record_ids"]),
        "source_sampling_rate_hz": args.source_rate,
        "source_physical_unit": "unknown_not_encoded_in_local_mat",
        "source_lead_order": LEAD_ORDER,
        "source_lead_order_provenance": "preprocessing_schema_assumption_not_encoded_in_local_mat",
        "output_sampling_rate_hz": args.output_rate,
        "windowing": {
            "method": "all_complete_nonoverlapping_four_second_windows_per_source_record",
            "window_seconds": WINDOW_SECONDS,
            "window_samples": window_samples,
            "total_windows": total_windows,
            "incomplete_tail_policy": "discard",
            "minimum_windows_per_record": int(np.min(scanned["window_counts"])),
            "maximum_windows_per_record": int(np.max(scanned["window_counts"])),
        },
        "split_method": "sklearn_train_test_split_over_all_windows_without_record_grouping",
        "split_seed": args.seed,
        "train_fraction": 1.0 - args.validation_fraction,
        "validation_fraction": args.validation_fraction,
        "split_hash": split_hash,
        "splits": split_summaries,
        "overlap": {
            "record_disjoint": False,
            "records_in_both_train_and_val": int(
                len(set(train_ids.tolist()) & set(val_ids.tolist()))
            ),
            "patient_disjoint_verified": False,
            "patient_disjoint_reason": "source MAT and REFERENCE.csv expose no patient identifier",
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
            "coefficient_scope": "each_full_variable_length_source_record_all_time_and_all_12_leads",
            "coefficient_files": "record_joint_minima_{split}.npy and record_joint_ranges_{split}.npy",
            "preserves_interlead_relative_amplitudes_and_offsets": True,
            "preserves_within_record_interwindow_scale": True,
            "heldout_target_statistics_used": True,
            "deployment_boundary": "paired benchmark only; coefficients use all 12 real leads",
        },
        "stored_waveforms": {
            "format": "X_{train,val}_resampled.npy",
            "layout": ["windows", "samples", "leads"],
            "dtype": "float32",
            "physical_unit": "unknown_source_unit",
        },
        "output_sha256": output_hashes,
        "clinical_policy": clinical_policy(),
        "claim_boundary": (
            "This non-grouped comparison permits source-record overlap and has no patient IDs. "
            "It does not measure record-independent or verified patient-independent generalization."
        ),
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"split_hash": split_hash, "splits": split_summaries, "overlap": manifest["overlap"]},
            indent=2,
        )
    )
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_rate", type=int, default=500)
    parser.add_argument("--output_rate", type=int, default=128)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--minimum_range", type=float, default=1e-6)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
