"""Build a training-only MIMIC-AFib Grad-CAM soft-mask cache."""

from __future__ import annotations

import argparse
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
from scipy.signal import resample_poly
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.interpretability.gradcam import (
    covering_crop_starts,
    gradcam_native_multi_1d,
    normalize_soft_mask,
    project_cam_to_sample_grid,
    resample_mask_to_sample_grid,
    stitch_temporal_cams,
)
from src.rcfm.interpretability.ptbxl_benchmark_compat import load_xresnet1d101


METHOD = "xresnet1d101_afib_gradcam_l5_sample_center_v1"
DATASET_VERSION = "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1"
SPLIT_HASH = "a7e388293adaa7b48d3493efc505dd8750520730cae9fd7649157866efa86a51"
SOURCE_RATE_HZ = 128
MODEL_RATE_HZ = 100
MODEL_LENGTH = 400
CROP_LENGTH = 250
CROP_STRIDE = 125
FEATURE_STRIDE = 8
AFIB_INDEX = 4


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


def _mask_for_record(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    raw_signal: np.ndarray,
    scaler_mean: float,
    scaler_scale: float,
    device: torch.device,
) -> tuple[np.ndarray, bool]:
    waveform = resample_poly(
        np.asarray(raw_signal, dtype=np.float32), MODEL_RATE_HZ, SOURCE_RATE_HZ, padtype="line"
    ).astype(np.float32)[:MODEL_LENGTH]
    if len(waveform) != MODEL_LENGTH:
        raise ValueError("MIMIC input did not resample to the expected four-second model grid")
    twelve_lead = np.repeat(waveform[:, None], 12, axis=1)
    standardized = (twelve_lead - scaler_mean) / scaler_scale
    starts = covering_crop_starts(MODEL_LENGTH, CROP_LENGTH, CROP_STRIDE)
    crop_masks: list[np.ndarray] = []
    for start in starts:
        crop = torch.from_numpy(standardized[start : start + CROP_LENGTH].T.copy())
        inputs = crop.unsqueeze(0).to(device=device, dtype=torch.float32)
        native, _logits = gradcam_native_multi_1d(
            model, {"xres_l5": target_layer}, inputs, AFIB_INDEX
        )
        crop_masks.append(
            project_cam_to_sample_grid(native["xres_l5"], CROP_LENGTH, FEATURE_STRIDE)
        )
    stitched = stitch_temporal_cams(crop_masks, starts, MODEL_LENGTH)
    projected = resample_mask_to_sample_grid(
        stitched, MODEL_RATE_HZ, SOURCE_RATE_HZ, output_length=512
    )
    return normalize_soft_mask(projected)


def _plot_examples(raw: np.ndarray, masks: np.ndarray, output_dir: Path) -> list[str]:
    indices = np.linspace(0, len(raw) - 1, min(4, len(raw)), dtype=np.int64)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(len(indices), 1, figsize=(7.16, 1.35 * len(indices)), squeeze=False)
    time = np.arange(512) / SOURCE_RATE_HZ
    for axis, index in zip(axes[:, 0], indices):
        signal = np.asarray(raw[int(index)], dtype=np.float32)
        span = max(float(np.max(signal) - np.min(signal)), np.finfo(np.float32).eps)
        display = 2.0 * (signal - float(np.min(signal))) / span - 1.0
        axis.plot(time, display, color="black", linewidth=0.6, label="ECG")
        mask = np.asarray(masks[int(index), 0])
        axis.fill_between(time, -1.0, 2.0 * mask - 1.0, color="#D55E00", alpha=0.22,
                          linewidth=0)
        axis.plot(time, 2.0 * mask - 1.0, color="#0072B2", linewidth=0.6,
                  label="Grad-CAM mask")
        axis.set_xlim(0, 4)
        axis.set_ylim(-1.05, 1.05)
        axis.set_ylabel(f"Window {int(index)}")
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1, 0].set_xlabel("Time (s)")
    figure.tight_layout(pad=0.5)
    names = ["gradcam_mask_examples.png", "gradcam_mask_examples.pdf"]
    figure.savefig(output_dir / names[0], dpi=600)
    figure.savefig(output_dir / names[1])
    plt.close(figure)
    return names


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark_code_root", type=Path, required=True)
    parser.add_argument("--mlb", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_layer", choices=["xres_l5"], default="xres_l5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected_records", type=int, default=8400)
    parser.add_argument("--max_records", type=int, default=None, help="Smoke-test cap only.")
    parser.add_argument("--progress_every", type=int, default=25)
    return parser


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.data_root.resolve() / "MIMIC-AFib" / "ecg_train_4sec.npy"
    required = [source_path, args.checkpoint, args.mlb, args.scaler]
    if any(not Path(path).is_file() for path in required):
        raise FileNotFoundError("a required waveform/checkpoint/metadata file is missing")
    if not (args.benchmark_code_root / "models" / "xresnet1d.py").is_file():
        raise FileNotFoundError("benchmark_code_root does not contain models/xresnet1d.py")
    if args.expected_records < 1 or args.progress_every < 1:
        raise ValueError("record and progress counts must be positive")

    raw = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if raw.ndim != 2 or raw.shape[1] != 512:
        raise ValueError("MIMIC training ECG must have shape (records, 512)")
    if len(raw) != args.expected_records:
        raise ValueError(f"expected {args.expected_records} training records, found {len(raw)}")
    records = len(raw) if args.max_records is None else min(args.max_records, len(raw))
    source_hash = _sha256(source_path)
    dependencies = {
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "mlb_sha256": _sha256(args.mlb.resolve()),
        "scaler_sha256": _sha256(args.scaler.resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "gradcam_module_sha256": _sha256(
            REPO_ROOT / "src" / "rcfm" / "interpretability" / "gradcam.py"
        ),
        "compat_module_sha256": _sha256(
            REPO_ROOT / "src" / "rcfm" / "interpretability" / "ptbxl_benchmark_compat.py"
        ),
    }
    fingerprint_payload = {
        "method": METHOD,
        "source_ecg_sha256": source_hash,
        "records": records,
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
        raise ValueError("existing completed Grad-CAM artifact does not match this request")

    completed = 0
    if partial_path.is_file() or progress_path.is_file():
        if not partial_path.is_file() or not progress_path.is_file():
            raise ValueError("incomplete cache is missing either its array or progress record")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("build_fingerprint") != fingerprint:
            raise ValueError("incomplete cache provenance does not match this request")
        completed = int(progress.get("completed_records", 0))
        masks = np.lib.format.open_memmap(partial_path, mode="r+", dtype=np.float32)
        if masks.shape != (records, 1, 512) or completed < 0 or completed > records:
            raise ValueError("incomplete cache shape/progress is invalid")
    else:
        masks = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float32, shape=(records, 1, 512)
        )
        _atomic_json(
            progress_path,
            {"status": "running", "build_fingerprint": fingerprint, "completed_records": 0},
        )

    with args.mlb.open("rb") as handle:
        classes = np.asarray(pickle.load(handle).classes_, dtype=str)
    afib_matches = np.flatnonzero(classes == "AFIB")
    if len(classes) != 71 or afib_matches.tolist() != [AFIB_INDEX]:
        raise ValueError("checkpoint metadata must expose AFIB at fixed class index 4 of 71")
    with args.scaler.open("rb") as handle:
        scaler = pickle.load(handle)
    scaler_mean = float(np.asarray(scaler.mean_).reshape(-1)[0])
    scaler_scale = float(np.asarray(scaler.scale_).reshape(-1)[0])
    if not np.isfinite([scaler_mean, scaler_scale]).all() or scaler_scale <= 0:
        raise ValueError("PTB-XL scaler must provide one finite positive scale")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = load_xresnet1d101(
        args.benchmark_code_root.resolve(), args.checkpoint.resolve(), map_location="cpu"
    ).eval().to(device)
    target_layer = model[5][-1]
    for index in range(completed, records):
        mask, is_degenerate = _mask_for_record(
            model, target_layer, raw[index], scaler_mean, scaler_scale, device
        )
        masks[index, 0] = mask
        done = index + 1
        if done % args.progress_every == 0 or done == records:
            masks.flush()
            _atomic_json(
                progress_path,
                {
                    "status": "running",
                    "build_fingerprint": fingerprint,
                    "completed_records": done,
                },
            )
            print(f"Grad-CAM masks: {done}/{records}", flush=True)
    del model
    masks.flush()
    del masks
    os.replace(partial_path, final_path)
    complete_masks = np.load(final_path, mmap_mode="r", allow_pickle=False)
    if not np.isfinite(complete_masks).all() or complete_masks.min() < 0 or complete_masks.max() > 1:
        raise ValueError("completed mask cache failed finite/range validation")
    plot_names = _plot_examples(raw[:records], complete_masks, output_dir)
    values = np.asarray(complete_masks).reshape(-1)
    quantiles = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    mask_hash = _sha256(final_path)
    degenerate_total = int(
        np.sum(np.ptp(np.asarray(complete_masks), axis=-1).reshape(-1) <= np.finfo(np.float32).eps)
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_fingerprint": fingerprint,
        "dataset": {
            "name": "MIMIC-AFib",
            "dataset_version": DATASET_VERSION,
            "split_hash": SPLIT_HASH,
            "train_records": records,
            "source_ecg": {"file_name": source_path.name, "sha256": source_hash},
            "sampling_rate_hz": SOURCE_RATE_HZ,
            "normalization_used_by_downstream_training": "rddm_window_minmax_neg1_1_v1",
        },
        "mask": {
            "method": METHOD,
            "file_name": final_path.name,
            "shape": list(complete_masks.shape),
            "dtype": str(complete_masks.dtype),
            "sha256": mask_hash,
            "soft_occupancy_mean": float(values.mean()),
            "quantile_levels": [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0],
            "quantiles": [float(value) for value in quantiles],
            "degenerate_records": degenerate_total,
        },
        "gradcam_protocol": {
            "architecture": "fastai_xresnet1d101",
            "training_labels": "PTB-XL all statements",
            "target": "AFIB pre-sigmoid logit",
            "target_index": AFIB_INDEX,
            "target_layer": args.target_layer,
            "native_feature_stride_samples_at_100_hz": FEATURE_STRIDE,
            "native_feature_resolution_ms": 80,
            "single_lead_adapter": "replicate MIMIC ECG to 12 channels",
            "resampling": "128 Hz to 100 Hz; sample-center interpolation back to 128 Hz",
            "crop_length_samples_at_100_hz": CROP_LENGTH,
            "crop_stride_samples_at_100_hz": CROP_STRIDE,
            "crop_stitching": "overlap_mean",
            "projection": "explicit_sample_center",
            "per_window_normalization": "soft_minmax_0_1_no_threshold",
            "scaler_mean": scaler_mean,
            "scaler_scale": scaler_scale,
        },
        "dependencies": dependencies,
        "figures": {name: _sha256(output_dir / name) for name in plot_names},
        "test_mask_generated": False,
        "mask_usage": "training_loss_and_diagnostics_only",
        "claim_boundary": (
            "Legacy cross-domain ablation only: single-lead replication, MIMIC lead/unit uncertainty, "
            "and failed prior P/QRS/T alignment checks prohibit physiological-transfer claims."
        ),
    }
    _atomic_json(manifest_path, manifest)
    progress_path.unlink()
    print(final_path)
    return final_path


if __name__ == "__main__":
    run(build_argparser().parse_args())
