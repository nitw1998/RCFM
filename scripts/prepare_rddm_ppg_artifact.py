"""Create an RDDM-compatible paired artifact with all-zero PPG windows removed."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np


UPSTREAM_COMMIT = "7d5348843c3985c211a23ae5105a2d9497d5156a"
WINDOW_SAMPLES = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def all_zero_ppg_mask(ppg: np.ndarray) -> np.ndarray:
    """Match upstream nan_to_num(float32), then identify strictly all-zero rows."""

    values = np.asarray(ppg)
    if values.ndim != 2 or values.shape[1] != WINDOW_SAMPLES:
        raise ValueError(f"PPG arrays must have shape (windows, {WINDOW_SAMPLES})")
    upstream_values = np.nan_to_num(values.astype(np.float32, copy=False))
    return np.all(upstream_values == 0.0, axis=1)


def filter_paired_windows(
    ecg: np.ndarray,
    ppg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove all-zero PPG rows and the ECG rows paired with them."""

    ecg = np.asarray(ecg)
    ppg = np.asarray(ppg)
    if ecg.ndim != 2 or ecg.shape[1] != WINDOW_SAMPLES:
        raise ValueError(f"ECG arrays must have shape (windows, {WINDOW_SAMPLES})")
    if len(ecg) != len(ppg):
        raise ValueError("ECG and PPG arrays must contain the same number of windows")
    rejected = all_zero_ppg_mask(ppg)
    kept_indices = np.flatnonzero(~rejected).astype(np.int64)
    rejected_indices = np.flatnonzero(rejected).astype(np.int64)
    return ecg[kept_indices], ppg[kept_indices], kept_indices, rejected_indices


def prepare(source_root: Path, output_dir: Path) -> Path:
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    if source_root == output_dir:
        raise ValueError("output_dir must differ from the read-only source_root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    split_summaries: dict[str, object] = {}
    membership_lines: list[str] = []
    for split in ("train", "test"):
        ecg_path = source_root / f"ecg_{split}_4sec.npy"
        ppg_path = source_root / f"ppg_{split}_4sec.npy"
        if not ecg_path.is_file() or not ppg_path.is_file():
            raise FileNotFoundError(f"missing RDDM-compatible paired arrays for split={split}")
        ecg = np.load(ecg_path, allow_pickle=False)
        ppg = np.load(ppg_path, allow_pickle=False)
        filtered_ecg, filtered_ppg, kept_indices, rejected_indices = filter_paired_windows(
            ecg, ppg
        )

        source_hashes[ecg_path.name] = _sha256(ecg_path)
        source_hashes[ppg_path.name] = _sha256(ppg_path)
        outputs = {
            f"ecg_{split}_4sec.npy": filtered_ecg,
            f"ppg_{split}_4sec.npy": filtered_ppg,
            f"kept_indices_{split}.npy": kept_indices,
            f"filtered_all_zero_ppg_indices_{split}.npy": rejected_indices,
        }
        for name, values in outputs.items():
            path = output_dir / name
            _atomic_save_npy(path, values)
            output_hashes[name] = _sha256(path)
        membership_lines.extend(f"{split}\t{int(index)}" for index in kept_indices)
        split_summaries[split] = {
            "source_windows": int(len(ecg)),
            "retained_windows": int(len(filtered_ecg)),
            "filtered_all_zero_ppg_windows": int(len(rejected_indices)),
            "source_ecg_dtype": str(ecg.dtype),
            "source_ppg_dtype": str(ppg.dtype),
            "output_ecg_dtype": str(filtered_ecg.dtype),
            "output_ppg_dtype": str(filtered_ppg.dtype),
        }

    split_membership_hash = hashlib.sha256(
        ("\n".join(membership_lines) + "\n").encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset": "MIMIC-AFib",
        "dataset_version": "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1",
        "upstream_repository": "https://github.com/DebadityaQU/RDDM",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_license": "MIT",
        "source_arrays_read_only": True,
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": 4,
        "sampling_rate_hz": 128,
        "filter": {
            "field": "condition_ppg",
            "timing": "after_upstream_nan_to_num_float32_semantics_before_minmax_and_cleaning",
            "predicate": "all_512_samples_equal_zero",
            "paired_ecg_row_removed": True,
            "nonzero_constant_windows_removed": False,
        },
        "upstream_loader_after_qc": {
            "normalization": "independent_per_window_ecg_and_ppg_minmax_neg1_1",
            "ppg_clean": "neurokit2.ppg_clean_sampling_rate_128",
            "ecg_clean": "neurokit2.ecg_clean_pantompkins1985_sampling_rate_128",
            "training_mask": "target_ecg_r_peaks_roi_size_32",
        },
        "subject_disjoint_verified": False,
        "subject_metadata_available": False,
        "split_membership_hash": split_membership_hash,
        "splits": split_summaries,
        "source_sha256": source_hashes,
        "output_sha256": output_hashes,
        "command": shlex.join(sys.argv),
    }
    _atomic_json(output_dir / "dataset_manifest.json", manifest)
    print(f"RDDM-compatible PPG QC complete: {output_dir}")
    print(json.dumps(split_summaries, indent=2, sort_keys=True))
    print(f"split_membership_hash={split_membership_hash}")
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_argparser().parse_args()
    prepare(arguments.source_root, arguments.output_dir)
