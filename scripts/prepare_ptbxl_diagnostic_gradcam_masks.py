"""Build PTB-XL training masks from each record's positive all-statements labels."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.interpretability.gradcam import (
    covering_crop_starts,
    gradcam_native_multi_target_1d,
    normalize_soft_mask,
    project_cam_to_sample_grid,
    resample_mask_to_sample_grid,
    stitch_temporal_cams,
)
from src.rcfm.interpretability.ptbxl_benchmark_compat import load_xresnet1d101


METHOD = "xresnet1d101_all_positive_statements_gradcam_l5_sample_center_v1"
DATASET_VERSION = "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1"
SPLIT_HASH = "7784c98a2c8daccc23fc7cb0d47dc1933eeee77f120c05f8a24ac149cd8474f7"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_positive_labels(
    metadata_path: Path,
    record_ids_path: Path,
    mlb_path: Path,
    benchmark_y_train_path: Path,
) -> tuple[list[np.ndarray], list[str], str]:
    with mlb_path.open("rb") as handle:
        classes = [str(value) for value in pickle.load(handle).classes_]
    if len(classes) != 71 or len(set(classes)) != len(classes):
        raise ValueError("all-statements metadata must expose 71 unique classes")
    class_to_index = {name: index for index, name in enumerate(classes)}
    metadata = pd.read_csv(metadata_path, usecols=["ecg_id", "strat_fold", "scp_codes"])
    if metadata.ecg_id.duplicated().any():
        raise ValueError("PTB-XL metadata contains duplicate ECG IDs")
    metadata = metadata.set_index("ecg_id", drop=False)
    record_ids = np.load(record_ids_path, allow_pickle=False).astype(np.int64)
    unknown = sorted(set(record_ids.tolist()) - set(metadata.index.astype(int).tolist()))
    if unknown:
        raise ValueError(f"training record IDs are absent from PTB-XL metadata: {unknown[:10]}")
    labels: list[np.ndarray] = []
    reconstructed = np.zeros((len(record_ids), len(classes)), dtype=np.int64)
    for row_index, record_id in enumerate(record_ids):
        row = metadata.loc[int(record_id)]
        if int(row.strat_fold) not in range(1, 9):
            raise ValueError("a training record does not belong to official folds 1-8")
        codes = ast.literal_eval(str(row.scp_codes))
        if not isinstance(codes, dict):
            raise ValueError("scp_codes must decode to a dictionary")
        indices = sorted(class_to_index[code] for code in codes if code in class_to_index)
        if not indices:
            raise ValueError("a training record has no positive all-statements class")
        reconstructed[row_index, indices] = 1
        labels.append(np.asarray(indices, dtype=np.int64))

    benchmark = np.load(benchmark_y_train_path, allow_pickle=True)
    benchmark_train_ids = metadata.loc[metadata.strat_fold.astype(int) <= 8, "ecg_id"].to_numpy()
    if benchmark.shape != (len(benchmark_train_ids), len(classes)):
        raise ValueError("benchmark y_train shape does not match official folds 1-8")
    benchmark_row = {int(record_id): index for index, record_id in enumerate(benchmark_train_ids)}
    aligned = np.stack([benchmark[benchmark_row[int(record_id)]] for record_id in record_ids])
    if not np.array_equal(reconstructed, aligned):
        mismatch = int(np.sum(reconstructed != aligned))
        raise ValueError(f"reconstructed all-statements labels disagree with benchmark ({mismatch} cells)")
    classes_hash = hashlib.sha256(("\n".join(classes) + "\n").encode("utf-8")).hexdigest()
    return labels, classes, classes_hash


def _record_mask(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    waveform_128hz: np.ndarray,
    positive_indices: np.ndarray,
    scaler_mean: float,
    scaler_scale: float,
    device: torch.device,
) -> tuple[np.ndarray, bool]:
    values = np.asarray(waveform_128hz, dtype=np.float32)
    if values.shape != (512, 12) or not np.all(np.isfinite(values)):
        raise ValueError("PTB-XL model waveform must have shape (512, 12) and be finite")
    waveform = resample_poly(values, 100, 128, axis=0, padtype="line").astype(np.float32)[:400]
    standardized = (waveform - scaler_mean) / scaler_scale
    starts = covering_crop_starts(400, 250, 125)
    crop_masks: list[np.ndarray] = []
    for start in starts:
        inputs = torch.from_numpy(standardized[start : start + 250].T.copy())
        inputs = inputs.unsqueeze(0).to(device=device, dtype=torch.float32)
        native, _logits = gradcam_native_multi_target_1d(
            model,
            {"xres_l5": target_layer},
            inputs,
            positive_indices.tolist(),
            reduction="mean",
        )
        crop_masks.append(project_cam_to_sample_grid(native["xres_l5"], 250, 8))
    stitched = stitch_temporal_cams(crop_masks, starts, 400)
    projected = resample_mask_to_sample_grid(stitched, 100, 128, output_length=512)
    return normalize_soft_mask(projected)


def _plot_examples(
    waveforms: np.ndarray,
    masks: np.ndarray,
    labels: list[np.ndarray],
    classes: list[str],
    output_dir: Path,
) -> list[str]:
    indices = np.linspace(0, len(waveforms) - 1, min(4, len(waveforms)), dtype=np.int64)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(len(indices), 1, figsize=(7.16, 1.4 * len(indices)), squeeze=False)
    time = np.arange(512) / 128.0
    for axis, index in zip(axes[:, 0], indices):
        signal = np.asarray(waveforms[int(index), :512, 1], dtype=np.float32)
        span = max(float(signal.max() - signal.min()), np.finfo(np.float32).eps)
        signal = 2.0 * (signal - float(signal.min())) / span - 1.0
        mask = np.asarray(masks[int(index), 0])
        names = [classes[value] for value in labels[int(index)]]
        shown = ", ".join(names[:5]) + (" ..." if len(names) > 5 else "")
        axis.plot(time, signal, color="#222222", linewidth=0.65, label="Lead II")
        axis.fill_between(time, -1.0, 2.0 * mask - 1.0, color="#D55E00", alpha=0.2)
        axis.plot(time, 2.0 * mask - 1.0, color="#0072B2", linewidth=0.65, label="DiagMask")
        axis.set(xlim=(0, 4), ylim=(-1.05, 1.05), ylabel=f"Row {int(index)}")
        axis.set_title(f"Positive statements: {shown}", loc="left", fontsize=7)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1, 0].set_xlabel("Time (s)")
    figure.tight_layout(pad=0.6)
    names = ["diagnostic_gradcam_examples.png", "diagnostic_gradcam_examples.pdf"]
    figure.savefig(output_dir / names[0], dpi=600)
    figure.savefig(output_dir / names[1])
    plt.close(figure)
    return names


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark_code_root", type=Path, required=True)
    parser.add_argument("--benchmark_y_train", type=Path, required=True)
    parser.add_argument("--mlb", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected_records", type=int, default=17440)
    parser.add_argument("--max_records", type=int, default=None, help="Smoke-test cap only.")
    parser.add_argument("--progress_every", type=int, default=25)
    return parser


def run(args: argparse.Namespace) -> Path:
    dataset_root = args.data_root.resolve() / "PTBXL"
    source_path = dataset_root / "X_train_resampled.npy"
    record_ids_path = dataset_root / "record_ids_train.npy"
    required = [
        source_path, record_ids_path, args.metadata, args.checkpoint,
        args.benchmark_y_train, args.mlb, args.scaler,
    ]
    if any(not Path(path).is_file() for path in required):
        raise FileNotFoundError("a required PTB-XL waveform/label/checkpoint file is missing")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    waveforms = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if waveforms.shape != (args.expected_records, 1280, 12):
        raise ValueError("PTB-XL training waveform shape does not match the frozen contract")
    labels, classes, classes_hash = _load_positive_labels(
        args.metadata.resolve(), record_ids_path, args.mlb.resolve(), args.benchmark_y_train.resolve()
    )
    records = len(waveforms) if args.max_records is None else min(args.max_records, len(waveforms))
    labels = labels[:records]
    dependencies = {
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "metadata_sha256": _sha256(args.metadata.resolve()),
        "record_ids_sha256": _sha256(record_ids_path),
        "benchmark_y_train_sha256": _sha256(args.benchmark_y_train.resolve()),
        "mlb_sha256": _sha256(args.mlb.resolve()),
        "scaler_sha256": _sha256(args.scaler.resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "gradcam_module_sha256": _sha256(REPO_ROOT / "src/rcfm/interpretability/gradcam.py"),
    }
    fingerprint_payload = {
        "method": METHOD,
        "source_sha256": _sha256(source_path),
        "records": records,
        "classes_sha256": classes_hash,
        "dependencies": dependencies,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    final_path = output_dir / "region_masks_train.npy"
    partial_path = output_dir / "region_masks_train.partial.npy"
    progress_path = output_dir / "progress.json"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "completed"
            and manifest.get("build_fingerprint") == fingerprint
            and final_path.is_file()
            and manifest.get("mask", {}).get("sha256") == _sha256(final_path)
        ):
            print(final_path)
            return final_path
        raise ValueError("existing completed diagnostic-mask artifact does not match this request")

    completed = 0
    if partial_path.is_file() or progress_path.is_file():
        if not partial_path.is_file() or not progress_path.is_file():
            raise ValueError("incomplete cache is missing either its array or progress record")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("build_fingerprint") != fingerprint:
            raise ValueError("incomplete cache provenance does not match this request")
        completed = int(progress.get("completed_records", 0))
        masks = np.lib.format.open_memmap(partial_path, mode="r+", dtype=np.float32)
        if masks.shape != (records, 1, 512):
            raise ValueError("incomplete cache has an invalid shape")
    else:
        masks = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float32, shape=(records, 1, 512)
        )
        _atomic_json(progress_path, {"status": "running", "build_fingerprint": fingerprint, "completed_records": 0})

    with args.scaler.open("rb") as handle:
        scaler = pickle.load(handle)
    scaler_mean = float(np.asarray(scaler.mean_).reshape(-1)[0])
    scaler_scale = float(np.asarray(scaler.scale_).reshape(-1)[0])
    if not np.isfinite([scaler_mean, scaler_scale]).all() or scaler_scale <= 0:
        raise ValueError("benchmark scaler must provide one finite positive scale")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = load_xresnet1d101(
        args.benchmark_code_root.resolve(), args.checkpoint.resolve(), map_location="cpu"
    ).eval().to(device)
    target_layer = model[5][-1]
    for index in range(completed, records):
        mask, _degenerate = _record_mask(
            model, target_layer, waveforms[index, :512], labels[index],
            scaler_mean, scaler_scale, device,
        )
        masks[index, 0] = mask
        done = index + 1
        if done % args.progress_every == 0 or done == records:
            masks.flush()
            _atomic_json(progress_path, {"status": "running", "build_fingerprint": fingerprint, "completed_records": done})
            print(f"Diagnostic Grad-CAM masks: {done}/{records}", flush=True)
    masks.flush()
    del masks, model
    os.replace(partial_path, final_path)
    complete_masks = np.load(final_path, mmap_mode="r", allow_pickle=False)
    if not np.isfinite(complete_masks).all() or complete_masks.min() < 0 or complete_masks.max() > 1:
        raise ValueError("completed diagnostic mask cache failed finite/range validation")
    figures = _plot_examples(waveforms[:records], complete_masks, labels, classes, output_dir)
    flat = np.asarray(complete_masks).reshape(-1)
    label_counts = np.asarray([len(values) for values in labels], dtype=np.int64)
    degenerate = int(np.sum(np.ptp(np.asarray(complete_masks), axis=-1).reshape(-1) <= np.finfo(np.float32).eps))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_fingerprint": fingerprint,
        "dataset": {
            "name": "PTBXL",
            "dataset_version": DATASET_VERSION,
            "split_hash": SPLIT_HASH,
            "train_records": records,
            "source_ecg": {"file_name": source_path.name, "sha256": _sha256(source_path)},
            "record_ids_sha256": _sha256(record_ids_path),
            "sampling_rate_hz": 128,
            "folds": list(range(1, 9)),
        },
        "mask": {
            "method": METHOD,
            "file_name": final_path.name,
            "shape": list(complete_masks.shape),
            "dtype": str(complete_masks.dtype),
            "sha256": _sha256(final_path),
            "soft_occupancy_mean": float(flat.mean()),
            "quantiles": [float(value) for value in np.quantile(flat, [0, 0.25, 0.5, 0.75, 1])],
            "degenerate_records": degenerate,
            "broadcast_policy": "one temporal mask broadcast to all 11 generated target leads",
        },
        "gradcam_protocol": {
            "architecture": "fastai_xresnet1d101",
            "training_task": "PTB-XL all statements",
            "classes": 71,
            "classes_sha256": classes_hash,
            "target_score": "mean pre-sigmoid logit over record ground-truth positive statements",
            "positive_labels_per_record_min_mean_max": [
                int(label_counts.min()), float(label_counts.mean()), int(label_counts.max())
            ],
            "target_layer": "xres_l5",
            "native_resolution_ms": 80,
            "projection": "explicit_sample_center",
            "crop_length_stride_at_100hz": [250, 125],
            "crop_stitching": "overlap_mean",
            "resampling": "128 Hz to 100 Hz and sample-center interpolation back to 128 Hz",
            "normalization": "benchmark training scalar StandardScaler",
            "soft_mask_normalization": "per-record minmax [0,1], no threshold",
        },
        "dependencies": dependencies,
        "figures": {name: _sha256(output_dir / name) for name in figures},
        "test_mask_generated": False,
        "mask_usage": "training_loss_weight_only",
        "claim_boundary": (
            "Diagnosis-conditioned training mask; not an anatomical segmentation, not used at "
            "inference, and downstream benefit requires a separate matched ablation."
        ),
    }
    _atomic_json(manifest_path, manifest)
    progress_path.unlink()
    print(final_path)
    return final_path


if __name__ == "__main__":
    run(build_argparser().parse_args())
