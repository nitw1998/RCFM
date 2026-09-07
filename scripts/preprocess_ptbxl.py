"""Build official-fold PTB-XL arrays compatible with the RCFM ECG loader.

The stored ``X_*_resampled.npy`` arrays remain in physical mV. The selected
normalization protocol only controls the reversible coefficient sidecars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly


LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SPLIT_FOLDS = {
    "train": tuple(range(1, 9)),
    "val": (9,),
    "test": (10,),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_metadata(source_root: Path) -> pd.DataFrame:
    path = source_root / "ptbxl_database.csv"
    metadata = pd.read_csv(path)
    required = {"ecg_id", "patient_id", "strat_fold", "filename_hr"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"PTB-XL metadata is missing columns: {sorted(missing)}")
    if metadata.empty or metadata.ecg_id.duplicated().any() or metadata.filename_hr.duplicated().any():
        raise ValueError("PTB-XL metadata must contain unique ECG IDs and high-resolution filenames")
    if metadata.patient_id.isna().any():
        raise ValueError("PTB-XL metadata contains missing patient IDs")
    folds = set(metadata.strat_fold.astype(int).tolist())
    if not folds or not folds <= set(range(1, 11)):
        raise ValueError(f"PTB-XL strat_fold values are invalid: {sorted(folds)}")
    return metadata


def official_split_indices(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return PTB-XL benchmark folds 1-8/9/10 and verify patient isolation."""

    splits = {
        name: np.flatnonzero(metadata.strat_fold.astype(int).isin(folds).to_numpy())
        for name, folds in SPLIT_FOLDS.items()
    }
    covered = np.concatenate(list(splits.values()))
    if len(covered) != len(metadata) or len(np.unique(covered)) != len(metadata):
        raise ValueError("official PTB-XL splits do not cover each metadata row exactly once")
    patient_sets = {
        name: set(metadata.iloc[indices].patient_id.astype(str))
        for name, indices in splits.items()
    }
    names = list(patient_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = patient_sets[left] & patient_sets[right]
            if overlap:
                raise ValueError(
                    f"PTB-XL patient overlap between {left} and {right}: {len(overlap)}"
                )
    return splits


def validate_waveform_inventory(source_root: Path, metadata: pd.DataFrame) -> str:
    lines: list[str] = []
    missing: list[str] = []
    for row in metadata.itertuples(index=False):
        relative = str(row.filename_hr)
        for extension in (".hea", ".dat"):
            path = source_root / f"{relative}{extension}"
            if not path.is_file():
                missing.append(path.relative_to(source_root).as_posix())
            else:
                lines.append(f"{path.relative_to(source_root).as_posix()}\t{path.stat().st_size}")
    if missing:
        raise FileNotFoundError(
            f"PTB-XL high-resolution waveform inventory is incomplete; missing={missing[:10]}"
        )
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()


def resample_waveform(
    signal: np.ndarray,
    source_rate_hz: int = 500,
    output_rate_hz: int = 128,
    duration_seconds: int = 10,
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    expected_source_shape = (source_rate_hz * duration_seconds, len(LEAD_ORDER))
    if values.shape != expected_source_shape:
        raise ValueError(f"unexpected PTB-XL waveform shape {values.shape}; expected {expected_source_shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("PTB-XL waveform contains NaN or Inf")
    resampled = resample_poly(values, output_rate_hz, source_rate_hz, axis=0, padtype="line")
    expected_output_shape = (output_rate_hz * duration_seconds, len(LEAD_ORDER))
    if resampled.shape != expected_output_shape:
        raise AssertionError(f"unexpected resampled waveform shape: {resampled.shape}")
    return np.asarray(resampled, dtype=np.float32)


def record_minmax_statistics(
    values: np.ndarray,
    model_window_samples: int,
    minimum_lead_range: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute coefficients over exactly the fixed window consumed by RCFM."""

    waveforms = np.asarray(values, dtype=np.float32)
    if waveforms.ndim != 3 or waveforms.shape[2] != len(LEAD_ORDER):
        raise ValueError("values must have shape (records, samples, 12)")
    if model_window_samples <= 0 or model_window_samples > waveforms.shape[1]:
        raise ValueError("model window is outside the stored waveform duration")
    model_view = waveforms[:, :model_window_samples]
    if not np.all(np.isfinite(model_view)):
        raise ValueError("model waveform window contains NaN or Inf")
    minima = model_view.min(axis=1)
    ranges = model_view.max(axis=1) - minima
    if np.any(ranges < minimum_lead_range) or not np.all(np.isfinite(ranges)):
        bad = np.argwhere(ranges < minimum_lead_range)
        raise ValueError(f"constant or near-constant PTB-XL model lead windows: {bad[:10].tolist()}")
    return minima.astype(np.float32), ranges.astype(np.float32)


def minmax_neg1_1(values: np.ndarray, minima: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    waveforms = np.asarray(values, dtype=np.float32)
    offsets = np.asarray(minima, dtype=np.float32)
    scales = np.asarray(ranges, dtype=np.float32)
    if waveforms.ndim != 3 or offsets.shape != waveforms.shape[::2] or scales.shape != offsets.shape:
        raise ValueError("waveforms and min-max coefficients do not align")
    if np.any(scales <= 0):
        raise ValueError("min-max ranges must be positive")
    return (2.0 * (waveforms - offsets[:, None, :]) / scales[:, None, :] - 1.0).astype(np.float32)


def record_joint12_minmax_statistics(
    values: np.ndarray,
    model_window_samples: int,
    minimum_record_range: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one min/range per record over time and all 12 leads."""

    waveforms = np.asarray(values, dtype=np.float32)
    if waveforms.ndim != 3 or waveforms.shape[2] != len(LEAD_ORDER):
        raise ValueError("values must have shape (records, samples, 12)")
    if model_window_samples <= 0 or model_window_samples > waveforms.shape[1]:
        raise ValueError("model window is outside the stored waveform duration")
    model_view = waveforms[:, :model_window_samples]
    if not np.all(np.isfinite(model_view)):
        raise ValueError("model waveform window contains NaN or Inf")
    minima = model_view.min(axis=(1, 2))
    ranges = model_view.max(axis=(1, 2)) - minima
    if np.any(ranges < minimum_record_range) or not np.all(np.isfinite(ranges)):
        bad = np.flatnonzero(ranges < minimum_record_range)
        raise ValueError(f"constant or near-constant PTB-XL records: {bad[:10].tolist()}")
    return minima.astype(np.float32), ranges.astype(np.float32)


def joint12_minmax_neg1_1(
    values: np.ndarray, minima: np.ndarray, ranges: np.ndarray
) -> np.ndarray:
    """Apply one record-wise affine transform shared by all leads."""

    waveforms = np.asarray(values, dtype=np.float32)
    offsets = np.asarray(minima, dtype=np.float32)
    scales = np.asarray(ranges, dtype=np.float32)
    if waveforms.ndim != 3 or waveforms.shape[2] != len(LEAD_ORDER):
        raise ValueError("values must have shape (records, samples, 12)")
    if offsets.shape != (len(waveforms),) or scales.shape != offsets.shape:
        raise ValueError("joint 12-lead coefficients must have one value per record")
    if not np.all(np.isfinite(waveforms)) or not np.all(np.isfinite(offsets)):
        raise ValueError("joint 12-lead min-max inputs must be finite")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("joint 12-lead ranges must be finite and positive")
    return (
        2.0 * (waveforms - offsets[:, None, None]) / scales[:, None, None] - 1.0
    ).astype(np.float32)


def _load_and_resample_record(
    source_root: Path,
    relative_record: str,
    source_rate_hz: int,
    output_rate_hz: int,
    duration_seconds: int,
) -> np.ndarray:
    signal, fields = wfdb.rdsamp(str(source_root / relative_record))
    if int(fields.get("fs", -1)) != source_rate_hz:
        raise ValueError(f"{relative_record} has sampling rate {fields.get('fs')}")
    signal_names = [str(name).upper() for name in fields.get("sig_name", [])]
    if signal_names != LEAD_ORDER:
        raise ValueError(f"{relative_record} has unexpected lead order {signal_names}")
    units = fields.get("units", [])
    if list(units) != ["mV"] * len(LEAD_ORDER):
        raise ValueError(f"{relative_record} does not encode all leads in mV")
    return resample_waveform(signal, source_rate_hz, output_rate_hz, duration_seconds)


def _split_hash(metadata: pd.DataFrame, splits: Mapping[str, list[int] | np.ndarray]) -> str:
    lines = []
    for split, indices in splits.items():
        for index in indices:
            row = metadata.iloc[int(index)]
            lines.append(
                f"{int(row.ecg_id)}\t{int(row.patient_id)}\t{split}\t{int(row.strat_fold)}"
            )
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()


def run(args: argparse.Namespace) -> Path:
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    repository_root = Path(__file__).resolve().parents[1]
    if output_dir == repository_root or repository_root in output_dir.parents:
        raise ValueError("preprocessed datasets must be written outside the Git repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(source_root)
    splits = official_split_indices(metadata)
    source_inventory_hash = validate_waveform_inventory(source_root, metadata)
    output_samples = args.output_rate * args.duration_seconds
    model_window_samples = args.output_rate * args.model_window_seconds
    temporary_arrays: dict[str, Path] = {}
    output_hashes: dict[str, str] = {}
    split_summaries: dict[str, object] = {}
    effective_splits: dict[str, list[int]] = {}
    excluded_records: list[dict[str, object]] = []
    joint12 = getattr(args, "normalization_scope", "per_record_per_lead") == "per_record_joint_12lead"

    try:
        for split, indices in splits.items():
            output_path = output_dir / f"X_{split}_resampled.npy"
            temporary_path = output_dir / f".{output_path.name}.building"
            temporary_arrays[split] = temporary_path
            waveforms = np.lib.format.open_memmap(
                temporary_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(indices), output_samples, len(LEAD_ORDER)),
            )
            coefficient_shape = (len(indices),) if joint12 else (len(indices), len(LEAD_ORDER))
            minima = np.empty(coefficient_shape, dtype=np.float32)
            ranges = np.empty_like(minima)
            effective_indices: list[int] = []
            output_index = 0
            for scanned_index, source_index in enumerate(indices):
                row = metadata.iloc[int(source_index)]
                waveform = _load_and_resample_record(
                    source_root,
                    str(row.filename_hr),
                    args.source_rate,
                    args.output_rate,
                    args.duration_seconds,
                )
                model_view = waveform[:model_window_samples]
                if joint12:
                    record_minima = np.float32(model_view.min())
                    record_ranges = np.float32(model_view.max() - record_minima)
                    invalid = bool(record_ranges < args.minimum_lead_range)
                    invalid_leads = np.empty(0, dtype=np.int64)
                else:
                    record_minima = model_view.min(axis=0)
                    record_ranges = model_view.max(axis=0) - record_minima
                    invalid_leads = np.flatnonzero(record_ranges < args.minimum_lead_range)
                    invalid = bool(len(invalid_leads))
                if invalid:
                    detail = (
                        {
                            "reason": "joint_12lead_range_below_threshold_in_fixed_model_window",
                            "record_range_mV": float(record_ranges),
                        }
                        if joint12
                        else {
                            "reason": "required_lead_range_below_threshold_in_fixed_model_window",
                            "lead_indices": invalid_leads.astype(int).tolist(),
                            "lead_names": [LEAD_ORDER[index] for index in invalid_leads],
                            "lead_ranges_mV": [float(record_ranges[index]) for index in invalid_leads],
                        }
                    )
                    excluded_records.append(
                        {
                            "ecg_id": int(row.ecg_id),
                            "patient_id": int(row.patient_id),
                            "split": split,
                            "strat_fold": int(row.strat_fold),
                            **detail,
                        }
                    )
                else:
                    waveforms[output_index] = waveform
                    minima[output_index] = record_minima
                    ranges[output_index] = record_ranges
                    effective_indices.append(int(source_index))
                    output_index += 1
                if (scanned_index + 1) % 500 == 0:
                    print(f"{split}: processed {scanned_index + 1}/{len(indices)}", flush=True)
            waveforms.flush()
            del waveforms
            minima = minima[:output_index]
            ranges = ranges[:output_index]
            if output_index == len(indices):
                os.replace(temporary_path, output_path)
            else:
                compact_path = output_dir / f".{output_path.name}.compacting"
                compact = np.lib.format.open_memmap(
                    compact_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(output_index, output_samples, len(LEAD_ORDER)),
                )
                source = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
                for start in range(0, output_index, 512):
                    end = min(start + 512, output_index)
                    compact[start:end] = source[start:end]
                compact.flush()
                del compact
                del source
                temporary_path.unlink()
                os.replace(compact_path, output_path)

            effective_splits[split] = effective_indices
            rows = metadata.iloc[effective_indices]
            coefficient_prefix = "record_joint" if joint12 else "record"
            sidecars = {
                f"record_ids_{split}.npy": rows.ecg_id.to_numpy(dtype=np.int32),
                f"patient_ids_{split}.npy": rows.patient_id.to_numpy(dtype=np.int32),
                f"strat_folds_{split}.npy": rows.strat_fold.to_numpy(dtype=np.uint8),
                f"{coefficient_prefix}_minima_{split}.npy": minima,
                f"{coefficient_prefix}_ranges_{split}.npy": ranges,
            }
            for filename, array in sidecars.items():
                _atomic_save_npy(output_dir / filename, array)
            for path in [output_path, *(output_dir / filename for filename in sidecars)]:
                output_hashes[path.name] = _sha256(path)
            split_summaries[split] = {
                "records": output_index,
                "patients": int(rows.patient_id.nunique()),
                "folds": sorted(int(value) for value in rows.strat_fold.unique()),
                "excluded_records": len(indices) - output_index,
            }
    except BaseException:
        for temporary_path in temporary_arrays.values():
            temporary_path.unlink(missing_ok=True)
        raise

    split_hash = _split_hash(metadata, effective_splits)
    clinical_policy = {
        "hrv": {
            "status": "disabled_by_author_protocol",
            "reason": "ten_second_records_are_too_short_for_stable_hrv",
            "window_concatenation_prohibited": True,
        },
        "interval_metrics": {
            "status": "eligible_with_independent_delineation_or_aligned_ptbxl_plus_fiducials",
        },
        "physical_amplitude_metrics": {
            "status": "eligible_for_real_mV_waveforms",
            "generated_inverse_policy": "ground_truth_target_scaler_is_oracle_only",
        },
        "ptbxl_plus": {
            "status": "pending_complete_download_and_separate_alignment",
            "must_not_change_ptbxl_waveform_split": True,
        },
    }
    dataset_version = (
        "ptbxl-1.0.1-official-folds-record-joint12-minmax-neg1-1-v1"
        if joint12
        else "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1"
    )
    normalization = (
        {
            "model_input": "per_record_joint_12lead_minmax_neg1_1",
            "normalization_id": "record_joint12_minmax_neg1_1_v1",
            "formula": "x_scaled=2*(x-record_joint_min)/record_joint_range-1",
            "inverse_formula": "x=(x_scaled+1)*record_joint_range/2+record_joint_min",
            "coefficient_scope": "fixed_first_model_window_all_time_and_all_12_leads",
            "coefficient_files": "record_joint_minima_{split}.npy and record_joint_ranges_{split}.npy",
            "coefficient_shape": ["records"],
            "preserves_interlead_relative_amplitudes_and_offsets": True,
            "heldout_target_statistics_used": True,
            "deployment_boundary": (
                "paired-benchmark normalization only: held-out coefficients use Lead II and the "
                "11 target leads and are unavailable from Lead II alone"
            ),
            "stored_X_arrays_remain_mV_for_legacy_RCFM_loader_compatibility": True,
        }
        if joint12
        else {
            "model_input": "per_record_per_lead_minmax_neg1_1",
            "normalization_id": "record_minmax_neg1_1_v1",
            "formula": "x_scaled=2*(x-record_min)/record_range-1",
            "inverse_formula": "x=(x_scaled+1)*record_range/2+record_min",
            "coefficient_scope": "fixed_first_model_window",
            "coefficient_files": "record_minima_{split}.npy and record_ranges_{split}.npy",
            "coefficient_shape": ["records", 12],
            "preserves_interlead_relative_amplitudes_and_offsets": False,
            "stored_X_arrays_remain_mV_for_legacy_RCFM_loader_compatibility": True,
        }
    )
    manifest = {
        "schema_version": 1,
        "dataset": "PTB-XL",
        "dataset_version": dataset_version,
        "source_records": len(metadata),
        "eligible_records": len(metadata) - len(excluded_records),
        "excluded_records": excluded_records,
        "source_sampling_rate_hz": args.source_rate,
        "source_duration_seconds": args.duration_seconds,
        "source_physical_unit": "mV",
        "source_lead_order": LEAD_ORDER,
        "source_metadata_sha256": _sha256(source_root / "ptbxl_database.csv"),
        "source_scp_statements_sha256": _sha256(source_root / "scp_statements.csv"),
        "source_path_size_inventory_sha256": source_inventory_hash,
        "split_method": "official_ptbxl_strat_fold_1_8_train_9_val_10_test",
        "split_hash": split_hash,
        "patient_disjoint_verified": True,
        "splits": split_summaries,
        "stored_waveforms": {
            "format": "X_{train,val,test}_resampled.npy",
            "layout": ["records", "samples", "leads"],
            "dtype": "float32",
            "sampling_rate_hz": args.output_rate,
            "samples_per_record": output_samples,
            "physical_unit": "mV",
        },
        "model_view": {
            "window_policy": "first_contiguous_window_per_record",
            "window_seconds": args.model_window_seconds,
            "condition_lead": "II",
            "condition_lead_index": 1,
            "target_leads": [lead for index, lead in enumerate(LEAD_ORDER) if index != 1],
            "target_lead_indices": [index for index in range(len(LEAD_ORDER)) if index != 1],
        },
        "normalization": normalization,
        "signal_qc": {
            "timing": "after_official_fold_assignment_before_output_publication",
            "scope": (
                "fixed_first_model_window_joint_12lead_range"
                if joint12
                else "fixed_first_model_window_all_12_leads"
            ),
            "minimum_lead_range_mV": args.minimum_lead_range,
            "excluded_record_count": len(excluded_records),
            "policy": "exclude_entire_paired_record_without_reassigning_folds",
        },
        "output_sha256": output_hashes,
        "clinical_policy": clinical_policy,
        "command": shlex.join(sys.argv),
    }
    _atomic_json(output_dir / "dataset_manifest.json", manifest)
    _atomic_json(output_dir / "clinical_policy.json", clinical_policy)
    print(f"PTB-XL preprocessing complete: {output_dir}")
    print(f"split_hash={split_hash}")
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_rate", type=int, default=500)
    parser.add_argument("--output_rate", type=int, default=128)
    parser.add_argument("--duration_seconds", type=int, default=10)
    parser.add_argument("--model_window_seconds", type=int, default=4)
    parser.add_argument("--minimum_lead_range", type=float, default=1e-6)
    parser.add_argument(
        "--normalization_scope",
        choices=["per_record_per_lead", "per_record_joint_12lead"],
        default="per_record_per_lead",
    )
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
