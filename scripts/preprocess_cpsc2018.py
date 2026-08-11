"""Build record-disjoint CPSC2018 splits and 4-second 128 Hz ECG arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split


LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LABEL_NAMES = {
    1: "normal",
    2: "atrial_fibrillation",
    3: "first_degree_av_block",
    4: "left_bundle_branch_block",
    5: "right_bundle_branch_block",
    6: "premature_atrial_contraction",
    7: "premature_ventricular_contraction",
    8: "st_depression",
    9: "st_elevation",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected = {"Recording", "First_label", "Second_label", "Third_label"}
    if not rows or set(rows[0]) != expected:
        raise ValueError("REFERENCE.csv has an unexpected schema")
    parsed: list[dict[str, object]] = []
    for row in rows:
        labels = tuple(
            int(row[key])
            for key in ("First_label", "Second_label", "Third_label")
            if row[key]
        )
        if not labels or any(label not in LABEL_NAMES for label in labels):
            raise ValueError(f"invalid labels for record {row['Recording']}")
        parsed.append(
            {
                "record_id": row["Recording"],
                "first_label": labels[0],
                "labels": labels,
            }
        )
    record_ids = [str(row["record_id"]) for row in parsed]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("REFERENCE.csv contains duplicate record IDs")
    return parsed


def discover_record_paths(source_root: Path) -> dict[str, Path]:
    paths = sorted(source_root.glob("TrainingSet*/*.mat"))
    by_id = {path.stem: path for path in paths}
    if len(paths) != len(by_id):
        raise ValueError("CPSC source directories contain duplicate record IDs")
    return by_id


def make_record_splits(
    rows: Sequence[Mapping[str, object]],
    seed: int,
) -> dict[str, list[int]]:
    """Create deterministic 80/10/10 record splits stratified by primary label."""

    indices = np.arange(len(rows), dtype=np.int64)
    primary = np.asarray([int(row["first_label"]) for row in rows], dtype=np.int64)
    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=primary,
    )
    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.5,
        random_state=seed + 1,
        shuffle=True,
        stratify=primary[temporary_indices],
    )
    splits = {
        "train": sorted(train_indices.tolist()),
        "val": sorted(validation_indices.tolist()),
        "test": sorted(test_indices.tolist()),
    }
    sets = [set(values) for values in splits.values()]
    if any(left & right for index, left in enumerate(sets) for right in sets[index + 1 :]):
        raise AssertionError("record split overlap detected")
    if set().union(*sets) != set(indices.tolist()):
        raise AssertionError("record splits do not cover every source record")
    return splits


def preprocess_record(
    path: Path,
    source_rate_hz: int = 500,
    output_rate_hz: int = 128,
    window_seconds: int = 4,
) -> np.ndarray:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=["ECG"])
    if "ECG" not in payload or not hasattr(payload["ECG"], "data"):
        raise ValueError(f"{path.name} does not contain ECG.data")
    signal = np.asarray(payload["ECG"].data, dtype=np.float64)
    required_samples = source_rate_hz * window_seconds
    if signal.ndim != 2 or signal.shape[0] != len(LEAD_ORDER):
        raise ValueError(f"{path.name} must contain 12 leads x samples")
    if signal.shape[1] < required_samples:
        raise ValueError(f"{path.name} is shorter than the configured window")
    window = signal[:, :required_samples]
    if not np.all(np.isfinite(window)):
        raise ValueError(f"{path.name} contains NaN or Inf")
    resampled = resample_poly(window, output_rate_hz, source_rate_hz, axis=1, padtype="line")
    expected_samples = output_rate_hz * window_seconds
    if resampled.shape != (len(LEAD_ORDER), expected_samples):
        raise AssertionError(f"unexpected resampled shape for {path.name}: {resampled.shape}")
    return np.asarray(resampled.T, dtype=np.float32)


def record_zscore_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-record, per-lead coefficients for reversible z-score scaling."""

    if values.ndim != 3 or values.shape[2] != len(LEAD_ORDER):
        raise ValueError("values must have shape (records, samples, 12)")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite samples")
    means = np.mean(values, axis=1, dtype=np.float64).astype(np.float32)
    scales = np.std(values, axis=1, dtype=np.float64).astype(np.float32)
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(scales)):
        raise ValueError("record z-score coefficients must be finite")
    return means, scales


def select_records_by_lead_quality(
    rows: Sequence[Mapping[str, object]],
    values: np.ndarray,
    required_lead_indices: Sequence[int],
    minimum_lead_std: float,
) -> tuple[list[dict[str, object]], np.ndarray, list[dict[str, object]]]:
    """Apply the same required-lead quality gate before any record split."""

    if len(rows) != len(values):
        raise ValueError("rows and values must align by record")
    required_leads = tuple(sorted(set(required_lead_indices)))
    if not required_leads or any(index < 0 or index >= len(LEAD_ORDER) for index in required_leads):
        raise ValueError("required lead indices must be unique values from 0 through 11")
    if minimum_lead_std <= 0:
        raise ValueError("minimum_lead_std must be positive")
    _, scales = record_zscore_statistics(values)
    eligible_rows: list[dict[str, object]] = []
    eligible_indices: list[int] = []
    excluded_records: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        invalid_leads = [
            lead for lead in required_leads if float(scales[index, lead]) < minimum_lead_std
        ]
        if invalid_leads:
            excluded_records.append(
                {
                    "record_id": str(row["record_id"]),
                    "required_lead_indices_below_threshold": invalid_leads,
                    "required_lead_scales": {
                        str(lead): float(scales[index, lead]) for lead in required_leads
                    },
                }
            )
        else:
            eligible_rows.append(dict(row))
            eligible_indices.append(index)
    if not eligible_rows:
        raise ValueError("required-lead quality control excluded every record")
    return eligible_rows, values[eligible_indices], excluded_records


def clinical_policy() -> dict[str, object]:
    return {
        "hrv": {
            "status": "disabled_by_author_protocol",
            "metrics": ["sdnn_ms", "rmssd_ms"],
            "window_concatenation_prohibited": True,
        },
        "interval_metrics": {
            "status": "eligible_after_independent_delineation",
            "metrics": ["rr_ms", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms"],
            "p_wave_missing_policy": "not_applicable",
        },
        "physical_amplitude_metrics": {
            "status": "unavailable_unknown_source_unit",
            "metrics": ["p_amplitude", "r_amplitude", "t_amplitude", "st_deviation"],
            "normalized_values_must_not_be_labeled_mV": True,
        },
    }


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_split_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    paths: Mapping[str, Path],
    indices: Sequence[int],
    source_root: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_id", "source_file", "labels"])
        for index in indices:
            row = rows[index]
            record_id = str(row["record_id"])
            writer.writerow(
                [
                    record_id,
                    paths[record_id].relative_to(source_root).as_posix(),
                    ";".join(str(value) for value in row["labels"]),
                ]
            )


def _inventory_hash(paths: Mapping[str, Path], source_root: Path) -> str:
    lines = [
        f"{record_id}\t{path.relative_to(source_root).as_posix()}\t{path.stat().st_size}"
        for record_id, path in sorted(paths.items())
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def run(args: argparse.Namespace) -> Path:
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    repository_root = Path(__file__).resolve().parents[1]
    if output_dir == repository_root or repository_root in output_dir.parents:
        raise ValueError("preprocessed datasets must be written outside the Git repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = source_root / "REFERENCE.csv"
    rows = load_reference(reference_path)
    paths = discover_record_paths(source_root)
    reference_ids = {str(row["record_id"]) for row in rows}
    if reference_ids != set(paths):
        missing = sorted(reference_ids - set(paths))[:5]
        extra = sorted(set(paths) - reference_ids)[:5]
        raise ValueError(f"source/reference record mismatch; missing={missing}, extra={extra}")
    required_leads = tuple(sorted(set(args.required_lead_indices)))
    if not required_leads or any(index < 0 or index >= len(LEAD_ORDER) for index in required_leads):
        raise ValueError("required lead indices must be unique values from 0 through 11")
    if args.minimum_lead_std <= 0:
        raise ValueError("minimum_lead_std must be positive")

    source_values: list[np.ndarray] = []
    for source_index, row in enumerate(rows):
        record_id = str(row["record_id"])
        values = preprocess_record(
            paths[record_id],
            source_rate_hz=args.source_rate,
            output_rate_hz=args.output_rate,
            window_seconds=args.window_seconds,
        )
        source_values.append(values)
        if (source_index + 1) % 500 == 0:
            print(f"source QC: processed {source_index + 1}/{len(rows)}", flush=True)

    eligible_rows, all_eligible_values, excluded_records = select_records_by_lead_quality(
        rows,
        np.stack(source_values).astype(np.float32, copy=False),
        required_leads,
        args.minimum_lead_std,
    )
    splits = make_record_splits(eligible_rows, args.seed)

    split_hash_lines: list[str] = []
    output_hashes: dict[str, str] = {}
    split_summaries: dict[str, object] = {}
    for split, indices in splits.items():
        values = all_eligible_values[indices]
        record_ids: list[str] = []
        labels = np.zeros((len(indices), len(LABEL_NAMES)), dtype=np.uint8)
        for output_index, source_index in enumerate(indices):
            row = eligible_rows[source_index]
            record_id = str(row["record_id"])
            record_ids.append(record_id)
            for label in row["labels"]:
                labels[output_index, int(label) - 1] = 1
            split_hash_lines.append(
                f"{record_id}\t{split}\t{','.join(str(value) for value in row['labels'])}"
            )
            if (output_index + 1) % 500 == 0:
                print(f"{split}: processed {output_index + 1}/{len(indices)}", flush=True)

        array_path = output_dir / f"X_{split}_resampled.npy"
        ids_path = output_dir / f"record_ids_{split}.npy"
        labels_path = output_dir / f"labels_{split}.npy"
        means_path = output_dir / f"record_means_{split}.npy"
        scales_path = output_dir / f"record_scales_{split}.npy"
        record_means, record_scales = record_zscore_statistics(values)
        if np.any(record_scales[:, required_leads] < args.minimum_lead_std):
            raise AssertionError("a QC-invalid required lead reached a dataset split")
        _atomic_save_npy(array_path, values)
        _atomic_save_npy(ids_path, np.asarray(record_ids, dtype="U5"))
        _atomic_save_npy(labels_path, labels)
        _atomic_save_npy(means_path, record_means)
        _atomic_save_npy(scales_path, record_scales)
        _write_split_csv(
            output_dir / f"records_{split}.csv",
            eligible_rows,
            paths,
            indices,
            source_root,
        )
        output_hashes[array_path.name] = _sha256(array_path)
        output_hashes[ids_path.name] = _sha256(ids_path)
        output_hashes[labels_path.name] = _sha256(labels_path)
        output_hashes[means_path.name] = _sha256(means_path)
        output_hashes[scales_path.name] = _sha256(scales_path)
        primary_counts = Counter(int(eligible_rows[index]["first_label"]) for index in indices)
        split_summaries[split] = {
            "records": len(indices),
            "primary_label_counts": {str(key): primary_counts[key] for key in sorted(primary_counts)},
            "all_label_counts": labels.sum(axis=0).astype(int).tolist(),
        }

    split_hash = hashlib.sha256(("\n".join(sorted(split_hash_lines)) + "\n").encode()).hexdigest()
    policy = clinical_policy()
    manifest = {
        "schema_version": 2,
        "dataset": "CPSC2018",
        "source_reference_sha256": _sha256(reference_path),
        "source_path_size_inventory_sha256": _inventory_hash(paths, source_root),
        "source_records": len(rows),
        "eligible_records": len(eligible_rows),
        "excluded_records": excluded_records,
        "source_sampling_rate_hz": args.source_rate,
        "source_physical_unit": "unknown_not_encoded_in_local_mat",
        "source_lead_order": LEAD_ORDER,
        "source_lead_order_provenance": "preprocessing_schema_assumption_not_encoded_in_local_mat",
        "split_method": "record_disjoint_primary_label_stratified_80_10_10",
        "split_implementation": "sklearn.model_selection.train_test_split",
        "split_seed": args.seed,
        "split_hash": split_hash,
        "subject_disjoint_verified": False,
        "subject_disjoint_reason": "source MAT and REFERENCE.csv expose no patient identifier",
        "window_policy": "first_contiguous_window_per_record",
        "window_seconds": args.window_seconds,
        "output_sampling_rate_hz": args.output_rate,
        "normalization": {
            "stored_waveforms": "unnormalized_source_values",
            "model_input": "per_record_per_lead_zscore",
            "formula": "x_z=(x-record_mean)/record_scale",
            "inverse_formula": "x=x_z*record_scale+record_mean",
            "coefficient_files": "record_means_{split}.npy and record_scales_{split}.npy",
            "coefficient_shape": ["records", 12],
        },
        "signal_qc": {
            "timing": "before_record_split",
            "required_lead_indices": list(required_leads),
            "minimum_lead_std": args.minimum_lead_std,
            "excluded_record_count": len(excluded_records),
        },
        "output_dtype": "float32",
        "output_shape_per_record": [args.output_rate * args.window_seconds, 12],
        "label_names": {str(key): value for key, value in LABEL_NAMES.items()},
        "splits": split_summaries,
        "output_sha256": output_hashes,
        "clinical_policy": policy,
        "command": shlex.join(sys.argv),
    }
    _atomic_json(output_dir / "dataset_manifest.json", manifest)
    _atomic_json(output_dir / "clinical_policy.json", policy)
    print(f"CPSC2018 preprocessing complete: {output_dir}")
    print(f"split_hash={split_hash}")
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--source_rate", type=int, default=500)
    parser.add_argument("--output_rate", type=int, default=128)
    parser.add_argument("--window_seconds", type=int, default=4)
    parser.add_argument(
        "--required_lead_indices",
        type=int,
        nargs="+",
        default=list(range(len(LEAD_ORDER))),
        help=(
            "Leads that must have nonzero variance before splitting. The default requires "
            "all 12 leads for joint Lead-II-to-other-11 training."
        ),
    )
    parser.add_argument("--minimum_lead_std", type=float, default=1e-6)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
