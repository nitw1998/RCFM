"""Build a pre-cleaned MIMIC-AFib + WESAD artifact for joint RDDM training.

The transformation order follows the released RDDM loader: float32/nan_to_num,
independent per-window min-max scaling to [-1, 1], NeuroKit PPG cleaning,
Pan--Tompkins ECG cleaning, and a 32-sample R-peak ROI on training targets.
Held-out masks are deliberately not generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import neurokit2 as nk
import numpy as np
import sklearn
from sklearn.preprocessing import minmax_scale


SAMPLE_RATE = 128
WINDOW_SAMPLES = 512
ROI_SIZE = 32
UPSTREAM_COMMIT = "7d5348843c3985c211a23ae5105a2d9497d5156a"
DATASETS = ("MIMIC-AFib", "WESAD")
SPLITS = ("train", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _window_minmax(values: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(values, dtype=np.float32))
    if array.ndim != 2 or array.shape[1] != WINDOW_SAMPLES:
        raise ValueError(f"expected (windows,{WINDOW_SAMPLES}) array, got {array.shape}")
    if np.any(np.ptp(array, axis=1) <= 0):
        raise ValueError("constant windows must be removed before RDDM cleaning")
    return np.asarray(minmax_scale(array, (-1, 1), axis=1), dtype=np.float32)


def _clean_pair(pair: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ecg, ppg = pair
    clean_ppg = nk.ppg_clean(ppg, sampling_rate=SAMPLE_RATE)
    clean_ecg = nk.ecg_clean(
        ecg, sampling_rate=SAMPLE_RATE, method="pantompkins1985"
    )
    return np.asarray(clean_ecg, dtype=np.float32), np.asarray(clean_ppg, dtype=np.float32)


def _clean_arrays(
    ecg: np.ndarray, ppg: np.ndarray, workers: int, chunksize: int
) -> tuple[np.ndarray, np.ndarray]:
    scaled_ecg, scaled_ppg = _window_minmax(ecg), _window_minmax(ppg)
    pairs = zip(scaled_ecg, scaled_ppg)
    if workers == 1:
        rows = list(map(_clean_pair, pairs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_clean_pair, pairs, chunksize=chunksize))
    clean_ecg = np.stack([row[0] for row in rows]).astype(np.float32)
    clean_ppg = np.stack([row[1] for row in rows]).astype(np.float32)
    if not np.all(np.isfinite(clean_ecg)) or not np.all(np.isfinite(clean_ppg)):
        raise FloatingPointError("NeuroKit cleaning produced NaN or Inf")
    return clean_ecg, clean_ppg


def _region_mask(ecg: np.ndarray) -> np.ndarray:
    mask = np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
    try:
        _, info = nk.ecg_peaks(
            ecg,
            sampling_rate=SAMPLE_RATE,
            method="pantompkins1985",
            correct_artifacts=True,
            show=False,
        )
        peaks = info.get("ECG_R_Peaks", [])
    except Exception:
        peaks = []
    for peak in peaks:
        start = max(0, int(peak) - ROI_SIZE // 2)
        end = min(start + ROI_SIZE, WINDOW_SAMPLES)
        mask[0, start:end] = 1.0
    return mask


def _masks(ecg: np.ndarray, workers: int, chunksize: int) -> np.ndarray:
    if workers == 1:
        return np.stack(list(map(_region_mask, ecg))).astype(np.float32)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        values = executor.map(_region_mask, ecg, chunksize=chunksize)
        return np.stack(list(values)).astype(np.float32)


def _source_root(dataset: str, mimic_root: Path, wesad_root: Path) -> Path:
    root = mimic_root if dataset == "MIMIC-AFib" else wesad_root
    candidate = root / dataset
    return candidate if candidate.is_dir() else root


def prepare(args: argparse.Namespace) -> Path:
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty directory: {output_root}")
    if args.workers <= 0 or args.chunksize <= 0:
        raise ValueError("workers and chunksize must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, object] = {}
    combined_membership: list[str] = []

    for dataset in DATASETS:
        source = _source_root(dataset, args.mimic_root.resolve(), args.wesad_root.resolve())
        destination = output_root / dataset
        destination.mkdir()
        split_records: dict[str, object] = {}
        for split in SPLITS:
            ecg_path = source / f"ecg_{split}_4sec.npy"
            ppg_path = source / f"ppg_{split}_4sec.npy"
            if not ecg_path.is_file() or not ppg_path.is_file():
                raise FileNotFoundError(f"missing paired {dataset} {split} arrays under {source}")
            ecg = np.load(ecg_path, allow_pickle=False)
            ppg = np.load(ppg_path, allow_pickle=False)
            if ecg.shape != ppg.shape or ecg.ndim != 2 or ecg.shape[1] != WINDOW_SAMPLES:
                raise ValueError(f"invalid paired shape for {dataset}/{split}: {ecg.shape}/{ppg.shape}")
            if args.max_windows is not None:
                ecg, ppg = ecg[: args.max_windows], ppg[: args.max_windows]
            clean_ecg, clean_ppg = _clean_arrays(ecg, ppg, args.workers, args.chunksize)
            outputs = {
                f"ecg_{split}_4sec.npy": clean_ecg,
                f"ppg_{split}_4sec.npy": clean_ppg,
            }
            if split == "train":
                outputs["region_masks_train.npy"] = _masks(
                    clean_ecg, args.workers, args.chunksize
                )
            hashes: dict[str, str] = {}
            for name, values in outputs.items():
                path = destination / name
                _atomic_save_npy(path, values)
                hashes[name] = _sha256(path)
            combined_membership.extend(f"{dataset}\t{split}\t{i}" for i in range(len(ecg)))
            split_records[split] = {
                "windows": int(len(ecg)),
                "ecg_source_sha256": _sha256(ecg_path),
                "ppg_source_sha256": _sha256(ppg_path),
                "outputs": hashes,
                "region_mask_generated": split == "train",
            }
            print(f"prepared {dataset}/{split}: {len(ecg)} windows", flush=True)
        datasets[dataset] = {"source_root": str(source), "splits": split_records}

    membership_hash = hashlib.sha256(
        ("\n".join(combined_membership) + "\n").encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "dataset_version": "rddm-mimic-wesad-joint-official-loader-clean-v1",
        "datasets": datasets,
        "joint_training_datasets": list(DATASETS),
        "combined_membership_hash": membership_hash,
        "sample_rate_hz": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "preprocessing": {
            "order": [
                "nan_to_num_float32",
                "independent_per_window_per_modality_minmax_neg1_1",
                "neurokit2_ppg_clean_default_sampling_rate_128",
                "neurokit2_ecg_clean_pantompkins1985_sampling_rate_128",
                "training_target_r_peak_roi_width_32",
            ],
            "test_mask_generated": False,
            "inverse_transform": "unavailable_after_per_window_scaling_and_cleaning",
        },
        "provenance": {
            "upstream_repository": "https://github.com/DebadityaQU/RDDM",
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_license": "MIT",
            "implementation_status": "released-loader-faithful; joint-two-dataset adaptation",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "neurokit2": nk.__version__,
        },
        "command": shlex.join(sys.argv),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output_root / "dataset_manifest.json", manifest)
    print(f"completed artifact: {output_root}", flush=True)
    return output_root


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic_root", type=Path, required=True)
    parser.add_argument("--wesad_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument("--max_windows", type=int, default=None, help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    prepare(build_argparser().parse_args())
