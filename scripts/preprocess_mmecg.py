"""Build a subject-isolated mmECG RCG-to-ECG window dataset.

The source release contains repeated recordings for 11 subjects. This script
splits subjects before producing overlapping windows, preventing adjacent
windows from the same recording from crossing the train/test boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.model_selection import GroupShuffleSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--window_samples", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def _scalar(value: np.ndarray) -> object:
    squeezed = np.asarray(value).squeeze()
    if squeezed.size != 1:
        raise ValueError(f"expected scalar metadata, got shape {squeezed.shape}")
    return squeezed.item()


def _fuse_rcg(mmwave: np.ndarray) -> np.ndarray:
    if mmwave.ndim != 2 or mmwave.shape[1] != 50:
        raise ValueError(f"expected time x 50 mmWave channels, got {mmwave.shape}")
    energy = np.sum(np.square(mmwave, dtype=np.float64), axis=0)
    total = float(energy.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("mmWave channel energy must be finite and positive")
    return np.asarray(mmwave @ (energy / total), dtype=np.float32)


def _windows(signal: np.ndarray, size: int, overlap: float) -> np.ndarray:
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("window size must be positive and overlap must be in [0, 1)")
    step = int(size * (1.0 - overlap))
    if step <= 0 or len(signal) < size:
        raise ValueError("invalid window step or signal shorter than one window")
    starts = range(0, len(signal) - size + 1, step)
    return np.stack([signal[start : start + size] for start in starts]).astype(np.float32)


def _load_records(source_root: Path) -> list[dict[str, object]]:
    paths = sorted(source_root.glob("*.mat"), key=lambda path: int(path.stem))
    if not paths:
        raise FileNotFoundError(f"no numeric .mat files found under {source_root}")
    records = []
    for path in paths:
        data = sio.loadmat(path)["data"]
        if data.shape != (1, 1):
            raise ValueError(f"unexpected data struct shape in {path}: {data.shape}")
        fields = list(data[0, 0])
        mmwave = np.asarray(fields[0], dtype=np.float64)
        ecg = np.asarray(fields[1], dtype=np.float32).reshape(-1)
        if len(ecg) != len(mmwave) or not np.all(np.isfinite(ecg)):
            raise ValueError(f"invalid paired waveforms in {path}")
        records.append(
            {
                "source_file": path.name,
                "subject_id": str(_scalar(fields[3])),
                "label": str(_scalar(fields[6])),
                "rcg": _fuse_rcg(mmwave),
                "ecg": ecg,
            }
        )
    return records


def _split_records(
    records: list[dict[str, object]], test_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    groups = np.asarray([record["subject_id"] for record in records])
    indices = np.arange(len(records))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train, test = next(splitter.split(indices, groups=groups))
    if set(groups[train]) & set(groups[test]):
        raise AssertionError("subject leakage detected")
    return train.tolist(), test.tolist()


def _build_split(
    records: list[dict[str, object]], indices: list[int], size: int, overlap: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ecg_windows, rcg_windows, subject_ids, source_files = [], [], [], []
    for index in indices:
        record = records[index]
        ecg = _windows(record["ecg"], size, overlap)
        rcg = _windows(record["rcg"], size, overlap)
        if ecg.shape != rcg.shape:
            raise ValueError(f"window mismatch for {record['source_file']}")
        ecg_windows.append(ecg)
        rcg_windows.append(rcg)
        subject_ids.extend([record["subject_id"]] * len(ecg))
        source_files.extend([record["source_file"]] * len(ecg))
    return (
        np.concatenate(ecg_windows),
        np.concatenate(rcg_windows),
        np.asarray(subject_ids),
        np.asarray(source_files),
    )


def main() -> None:
    args = parse_args()
    records = _load_records(args.source_root)
    train_indices, test_indices = _split_records(records, args.test_fraction, args.seed)
    subject_map = {
        subject: f"S{index:03d}"
        for index, subject in enumerate(
            sorted({record["subject_id"] for record in records}, key=int), start=1
        )
    }
    for record in records:
        record["subject_id"] = subject_map[record["subject_id"]]
    train = _build_split(records, train_indices, args.window_samples, args.overlap)
    test = _build_split(records, test_indices, args.window_samples, args.overlap)

    for split, arrays in (("train", train), ("test", test)):
        ecg, rcg, subjects, files = arrays
        if not np.all(np.isfinite(ecg)) or not np.all(np.isfinite(rcg)):
            raise ValueError(f"{split} contains non-finite samples")
        if np.any(np.ptp(ecg, axis=1) <= 0) or np.any(np.ptp(rcg, axis=1) <= 0):
            raise ValueError(f"{split} contains constant ECG or RCG windows")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.output_dir / f"ecg_{split}_4sec.npy", ecg)
        np.save(args.output_dir / f"ppg_{split}_4sec.npy", rcg)
        np.save(args.output_dir / f"subject_ids_{split}.npy", subjects)
        np.save(args.output_dir / f"source_files_{split}.npy", files)

    assignment = {
        "train_subject_ids": sorted(set(train[2].tolist())),
        "test_subject_ids": sorted(set(test[2].tolist())),
    }
    split_hash = hashlib.sha256(
        json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    manifest = {
        "dataset": "mmECG",
        "dataset_version": "mmecg-public-20221108-subject-split-window-minmax-v1",
        "source_root": str(args.source_root),
        "source_records": len(records),
        "subject_count": len({record["subject_id"] for record in records}),
        "record_count_by_subject": dict(
            sorted(Counter(record["subject_id"] for record in records).items())
        ),
        "split_method": "GroupShuffleSplit_on_subject_before_windowing",
        "split_seed": args.seed,
        "test_fraction": args.test_fraction,
        "split_hash": split_hash,
        **assignment,
        "train_records": len(train_indices),
        "test_records": len(test_indices),
        "train_windows": len(train[0]),
        "test_windows": len(test[0]),
        "sampling_rate_hz": 128,
        "window_samples": args.window_samples,
        "window_seconds": args.window_samples / 128,
        "overlap": args.overlap,
        "condition": "energy-weighted 50-channel RCG stored under historical ppg_* filenames",
        "target": "single-channel ECG",
        "normalization": "deferred_to_loader_window_minmax_neg1_1_v1",
        "signal_cleaning": "none",
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
