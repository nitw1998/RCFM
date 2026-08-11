"""Audit MIMIC-AFib region-mask phase and temporal resolution without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shlex
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from scipy.signal import resample_poly
from scipy.special import expit
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.interpretability.ecgmambaformer_compat import (
    diagnostic_class_names,
    load_ecgmambaformer_fca_mgda,
    record_global_zscore,
)
from src.rcfm.interpretability.gradcam import (
    covering_crop_starts,
    gradcam_1d,
    gradcam_native_multi_1d,
    normalize_soft_mask,
    project_cam_to_sample_grid,
    resample_mask_to_sample_grid,
    stitch_temporal_cams,
)
from src.rcfm.interpretability.ptbxl_benchmark_compat import load_xresnet1d101
from src.rcfm.metrics.clinical import ECGFiducials, delineate_ecg


XRESNET_LAYERS = {"xres_l4": (4, 4), "xres_l5": (5, 8), "xres_l6": (6, 16), "xres_l7": (7, 32)}
REFERENCE_NAMES = ("p", "qrs", "t", "qt", "morphology")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _rddm_target_view(raw: np.ndarray, sampling_rate: int) -> np.ndarray:
    signal = np.nan_to_num(np.asarray(raw, dtype=np.float32))
    span = float(signal.max() - signal.min())
    if span <= np.finfo(np.float32).eps:
        raise ValueError("constant ECG window")
    scaled = 2.0 * (signal - float(signal.min())) / span - 1.0
    return np.asarray(
        nk.ecg_clean(scaled, sampling_rate=sampling_rate, method="pantompkins1985"),
        dtype=np.float32,
    )


def _refine_peaks_to_local_extrema(
    signal: np.ndarray,
    peaks: np.ndarray,
    sampling_rate: float,
    search_ms: float = 100.0,
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    candidates = np.asarray(peaks, dtype=np.int64).reshape(-1)
    radius = max(1, int(round(search_ms * sampling_rate / 1000.0)))
    refined: list[int] = []
    for peak in candidates:
        if peak < 0 or peak >= len(values):
            continue
        start, end = max(0, peak - radius), min(len(values), peak + radius + 1)
        baseline_start, baseline_end = max(0, peak - 4 * radius), min(len(values), peak + 4 * radius + 1)
        baseline = float(np.median(values[baseline_start:baseline_end]))
        local = int(start + np.argmax(np.abs(values[start:end] - baseline)))
        refined.append(local)
    return np.asarray(sorted(set(refined)), dtype=np.int64)


def _peak_window_mask(peaks: np.ndarray, length: int, width: int = 32) -> np.ndarray:
    mask = np.zeros(length, dtype=np.float32)
    for peak in np.asarray(peaks, dtype=np.int64):
        start = max(0, int(peak) - width // 2)
        end = min(start + width, length)
        mask[start:end] = 1.0
    return mask


def _interval_mask(starts: np.ndarray, ends: np.ndarray, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=np.float32)
    for start, end in zip(np.asarray(starts), np.asarray(ends)):
        if 0 <= int(start) < int(end) <= length:
            mask[int(start) : int(end) + 1] = 1.0
    return mask


def _fiducial_masks(fiducials: ECGFiducials, length: int) -> dict[str, np.ndarray]:
    t_onsets = 2 * fiducials.t_peaks - fiducials.t_offsets
    masks = {
        "p": _interval_mask(fiducials.p_onsets, fiducials.p_offsets, length),
        "qrs": _interval_mask(fiducials.qrs_onsets, fiducials.qrs_offsets, length),
        "t": _interval_mask(t_onsets, fiducials.t_offsets, length),
        "qt": _interval_mask(fiducials.qrs_onsets, fiducials.t_offsets, length),
    }
    masks["morphology"] = np.maximum.reduce([masks["p"], masks["qrs"], masks["t"]])
    return masks


def _centered_delineation(signal: np.ndarray, sampling_rate: int) -> ECGFiducials | None:
    result = delineate_ecg(signal, sampling_rate=sampling_rate, method="dwt")
    return result.fiducials if result.success else None


def _shift_with_zeros(values: np.ndarray, lag: int) -> np.ndarray:
    output = np.zeros_like(values)
    if lag == 0:
        output[:] = values
    elif lag > 0:
        output[lag:] = values[:-lag]
    else:
        output[:lag] = values[-lag:]
    return output


def _soft_alignment_metrics(mask: np.ndarray, reference: np.ndarray, max_lag: int = 32) -> dict[str, float]:
    values, degenerate = normalize_soft_mask(np.asarray(mask, dtype=np.float32))
    truth = np.asarray(reference, dtype=np.float32)
    if truth.ndim != 1 or truth.shape != values.shape or not np.any(truth > 0):
        raise ValueError("reference must be a nonempty one-dimensional mask matching mask")
    truth = (truth > 0).astype(np.float32)
    mass = float(values.sum())
    mass_in = float((values * truth).sum() / mass) if mass > 0 else 0.0
    inside_mean = float(values[truth > 0].mean())
    outside_mean = float(values[truth == 0].mean()) if np.any(truth == 0) else inside_mean
    threshold = float(np.quantile(values, 0.8))
    top = np.zeros_like(values, dtype=bool) if degenerate else values >= threshold
    intersection = float(np.sum(top & (truth > 0)))
    dice = 2.0 * intersection / max(float(np.sum(top) + np.sum(truth > 0)), 1.0)
    best_lag, best_correlation = 0, 0.0
    if not degenerate:
        best_correlation = -1.0
        for lag in range(-max_lag, max_lag + 1):
            shifted = _shift_with_zeros(values, lag)
            correlation = float(np.corrcoef(shifted, truth)[0, 1])
            if correlation > best_correlation:
                best_lag, best_correlation = lag, correlation
    zero_lag = float(np.corrcoef(values, truth)[0, 1]) if np.std(values) > 1e-8 else 0.0
    return {
        "mass_in_reference": mass_in,
        "inside_minus_outside": inside_mean - outside_mean,
        "top20_dice": dice,
        "zero_lag_correlation": zero_lag,
        "best_lag_samples": float(best_lag),
        "best_lag_correlation": float(best_correlation),
        "mask_degenerate": float(degenerate),
    }


def _legacy_interpolate(native: np.ndarray, length: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(native, dtype=np.float32)).reshape(1, 1, -1)
    return F.interpolate(tensor, size=length, mode="linear", align_corners=False)[0, 0].numpy()


def _xresnet_masks(
    model: torch.nn.Module,
    raw_signal: np.ndarray,
    scaler_mean: float,
    scaler_scale: float,
    target_index: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], float]:
    waveform = resample_poly(raw_signal, 100, 128, padtype="line").astype(np.float32)
    waveform = waveform[:400]
    twelve_lead = np.repeat(waveform[:, None], 12, axis=1)
    standardized = (twelve_lead - scaler_mean) / scaler_scale
    starts = covering_crop_starts(len(standardized), 250, 125)
    layer_modules = {name: model[index][-1] for name, (index, _stride) in XRESNET_LAYERS.items()}
    projected: dict[str, list[np.ndarray]] = {name: [] for name in XRESNET_LAYERS}
    legacy_l7: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for start in starts:
        crop = torch.from_numpy(standardized[start : start + 250].T.copy()).unsqueeze(0)
        crop = crop.to(device=device, dtype=torch.float32)
        native, crop_logits = gradcam_native_multi_1d(model, layer_modules, crop, target_index)
        logits.append(crop_logits)
        for name, (_index, stride) in XRESNET_LAYERS.items():
            projected[name].append(project_cam_to_sample_grid(native[name], 250, stride))
        legacy_l7.append(_legacy_interpolate(native["xres_l7"], 250))
    masks: dict[str, np.ndarray] = {}
    for name, crop_masks in projected.items():
        stitched = stitch_temporal_cams(crop_masks, starts, len(waveform))
        masks[name], _ = normalize_soft_mask(
            resample_mask_to_sample_grid(stitched, 100, 128, output_length=512)
        )
    legacy = stitch_temporal_cams(legacy_l7, starts, len(waveform))
    masks["xres_l7_legacy_interpolation"], _ = normalize_soft_mask(
        resample_mask_to_sample_grid(legacy, 100, 128, output_length=512)
    )
    probability = float(np.max(expit(np.stack(logits)[:, target_index])))
    return masks, probability


def _ecgmamba_masks(
    model: torch.nn.Module,
    raw_signal: np.ndarray,
    target_index: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], float]:
    waveform = resample_poly(raw_signal, 500, 128, padtype="line").astype(np.float32)[:2000]
    twelve_lead = np.repeat(waveform[:, None], 12, axis=1)
    normalized, _mean, _scale = record_global_zscore(twelve_lead)
    inputs = torch.from_numpy(normalized.T.copy()).unsqueeze(0).to(device=device, dtype=torch.float32)
    cam, logits = gradcam_1d(model, model.encoder, inputs, target_index)
    with torch.no_grad():
        semantic = model.semantic_probabilities(inputs)[0].detach().cpu().numpy()
    output: dict[str, np.ndarray] = {}
    output["mamba_norm_gradcam"], _ = normalize_soft_mask(
        resample_mask_to_sample_grid(cam, 500, 128, output_length=512)
    )
    for class_index, name in enumerate(("background", "p", "qrs", "t")):
        output[f"mamba_semantic_{name}"] = resample_mask_to_sample_grid(
            semantic[class_index], 500, 128, output_length=512
        )
    output["mamba_semantic_morphology"] = 1.0 - output["mamba_semantic_background"]
    return output, float(expit(logits[target_index]))


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), str(row["reference"])), []).append(row)
    output: list[dict[str, object]] = []
    metric_names = [
        "mass_in_reference", "inside_minus_outside", "top20_dice",
        "zero_lag_correlation", "best_lag_samples", "best_lag_correlation",
        "mask_degenerate",
    ]
    for (method, reference), items in sorted(grouped.items()):
        row: dict[str, object] = {"method": method, "reference": reference, "records": len(items)}
        for metric in metric_names:
            values = np.asarray([float(item[metric]) for item in items])
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"median_{metric}"] = float(np.median(values))
        output.append(row)
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_examples(examples: list[dict[str, object]], output_dir: Path, sampling_rate: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    count = len(examples)
    figure, axes = plt.subplots(count, 4, figsize=(7.16, 1.55 * count), squeeze=False)
    xres_names = ["xres_l7_legacy_interpolation", "xres_l4", "xres_l5", "xres_l6", "xres_l7"]
    mamba_names = [
        "mamba_norm_gradcam", "mamba_semantic_p", "mamba_semantic_qrs",
        "mamba_semantic_t", "mamba_semantic_morphology",
    ]
    for row_index, example in enumerate(examples):
        time = np.arange(512) / sampling_rate
        signal = np.asarray(example["signal"])
        references = example["references"]
        axis = axes[row_index, 0]
        axis.plot(time, signal, color="black", linewidth=0.55)
        colors = {"p": "#4c78a8", "qrs": "#d1495b", "t": "#59a14f"}
        y0, y1 = np.quantile(signal, [0.01, 0.99])
        for name, color in colors.items():
            mask = np.asarray(references[name]) > 0
            axis.fill_between(time, y0, y1, where=mask, color=color, alpha=0.16, linewidth=0)
        axis.set_title(f"Window {example['row_index']} | DWT reference")
        axis.set_ylabel("Normalized ECG")

        axis = axes[row_index, 1]
        axis.imshow(
            np.stack([example["masks"]["pan_original"], example["masks"]["pan_refined"]]),
            extent=(0, 4, 1.5, -0.5), aspect="auto", cmap="Reds", vmin=0, vmax=1,
            interpolation="nearest",
        )
        axis.set_yticks([0, 1], ["Pan", "refined"])
        axis.set_title("R-peak masks")

        axis = axes[row_index, 2]
        axis.imshow(
            np.stack([example["masks"][name] for name in xres_names]),
            extent=(0, 4, 4.5, -0.5), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
            interpolation="nearest",
        )
        axis.set_yticks(range(5), ["L7 old", "L4 40ms", "L5 80ms", "L6 160ms", "L7 320ms"])
        axis.set_title("XResNet AFIB Grad-CAM")

        axis = axes[row_index, 3]
        axis.imshow(
            np.stack([example["masks"][name] for name in mamba_names]),
            extent=(0, 4, 4.5, -0.5), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
            interpolation="nearest",
        )
        axis.set_yticks(range(5), ["NORM CAM", "P", "QRS", "T", "foreground"])
        axis.set_title("ECGMamba outputs")
        for column in range(4):
            axes[row_index, column].set_xlim(0, 4)
            if row_index == count - 1:
                axes[row_index, column].set_xlabel("Time (s)")
    figure.tight_layout(pad=0.7)
    figure.savefig(output_dir / "mask_alignment_examples.png", dpi=600)
    figure.savefig(output_dir / "mask_alignment_examples.pdf")
    plt.close(figure)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--xresnet_checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark_code_root", type=Path, required=True)
    parser.add_argument("--mlb", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--ecgmamba_checkpoint", type=Path, required=True)
    parser.add_argument("--ecgmamba_root", type=Path, required=True)
    parser.add_argument("--scp_statements", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--plot_samples", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    if args.samples < 1 or args.plot_samples < 1:
        raise ValueError("sample counts must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    data_path = args.data_root.resolve() / "MIMIC-AFib" / "ecg_train_4sec.npy"
    raw_array = np.load(data_path, mmap_mode="r", allow_pickle=False)
    indices = np.linspace(0, len(raw_array) - 1, args.samples, dtype=np.int64)

    with args.mlb.open("rb") as handle:
        mlb = pickle.load(handle)
    with args.scaler.open("rb") as handle:
        scaler = pickle.load(handle)
    classes = np.asarray(mlb.classes_, dtype=str)
    afib_matches = np.flatnonzero(classes == "AFIB")
    if len(afib_matches) != 1:
        raise ValueError("XResNet metadata must expose one AFIB class")
    afib_index = int(afib_matches[0])
    scaler_mean = float(np.asarray(scaler.mean_).reshape(-1)[0])
    scaler_scale = float(np.asarray(scaler.scale_).reshape(-1)[0])

    mamba_classes = diagnostic_class_names(args.scp_statements.resolve())
    if "NORM" not in mamba_classes or "AFIB" in mamba_classes:
        raise ValueError("ECGMamba diagnostic head must contain NORM and exclude AFIB")
    norm_index = mamba_classes.index("NORM")
    xresnet = load_xresnet1d101(
        args.benchmark_code_root.resolve(), args.xresnet_checkpoint.resolve(), map_location="cpu"
    ).eval().to(device)
    mamba = load_ecgmambaformer_fca_mgda(
        args.ecgmamba_root.resolve(), args.ecgmamba_checkpoint.resolve(), len(mamba_classes), map_location="cpu"
    ).eval().to(device)

    metric_rows: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    for row_index in indices:
        raw = np.asarray(raw_array[int(row_index)], dtype=np.float32)
        signal = _rddm_target_view(raw, 128)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fiducials = _centered_delineation(signal, 128)
        if fiducials is None:
            record_rows.append({"row_index": int(row_index), "status": "delineation_failed"})
            continue
        references = _fiducial_masks(fiducials, 512)
        _, peak_info = nk.ecg_peaks(
            signal, sampling_rate=128, method="pantompkins1985", correct_artifacts=True
        )
        pan_peaks = np.asarray(peak_info.get("ECG_R_Peaks", []), dtype=np.int64)
        refined_peaks = _refine_peaks_to_local_extrema(signal, pan_peaks, 128)
        masks = {
            "pan_original": _peak_window_mask(pan_peaks, 512),
            "pan_refined": _peak_window_mask(refined_peaks, 512),
        }
        xres_masks, afib_probability = _xresnet_masks(
            xresnet, raw, scaler_mean, scaler_scale, afib_index, device
        )
        mamba_masks, norm_probability = _ecgmamba_masks(mamba, raw, norm_index, device)
        masks.update(xres_masks)
        masks.update(mamba_masks)

        comparisons = {
            "pan_original": ("qrs",),
            "pan_refined": ("qrs",),
            "xres_l7_legacy_interpolation": REFERENCE_NAMES,
            "xres_l4": REFERENCE_NAMES,
            "xres_l5": REFERENCE_NAMES,
            "xres_l6": REFERENCE_NAMES,
            "xres_l7": REFERENCE_NAMES,
            "mamba_norm_gradcam": REFERENCE_NAMES,
            "mamba_semantic_p": ("p",),
            "mamba_semantic_qrs": ("qrs",),
            "mamba_semantic_t": ("t",),
            "mamba_semantic_morphology": ("morphology",),
        }
        for method, reference_names in comparisons.items():
            for reference_name in reference_names:
                if not np.any(references[reference_name]):
                    continue
                metrics = _soft_alignment_metrics(masks[method], references[reference_name])
                metric_rows.append(
                    {"row_index": int(row_index), "method": method, "reference": reference_name, **metrics}
                )
        record_rows.append(
            {
                "row_index": int(row_index),
                "status": "completed",
                "pan_peaks": len(pan_peaks),
                "refined_peaks": len(refined_peaks),
                "dwt_r_peaks": len(fiducials.r_peaks),
                "xresnet_afib_probability": afib_probability,
                "ecgmamba_norm_probability": norm_probability,
            }
        )
        if len(examples) < args.plot_samples:
            examples.append(
                {"row_index": int(row_index), "signal": signal, "references": references, "masks": masks}
            )

    if not metric_rows or not examples:
        raise RuntimeError("no MIMIC windows completed mask auditing")
    aggregate = _aggregate(metric_rows)
    _write_csv(output_dir / "per_window_alignment.csv", metric_rows)
    _write_csv(output_dir / "aggregate_alignment.csv", aggregate)
    _write_csv(output_dir / "window_qc.csv", record_rows)
    _plot_examples(examples, output_dir, 128)

    aggregate_lookup = {(row["method"], row["reference"]): row for row in aggregate}
    candidates = {}
    for method, reference in (
        ("pan_original", "qrs"), ("pan_refined", "qrs"),
        ("xres_l4", "morphology"), ("xres_l5", "morphology"),
        ("mamba_semantic_p", "p"), ("mamba_semantic_qrs", "qrs"),
        ("mamba_semantic_t", "t"),
    ):
        row = aggregate_lookup.get((method, reference))
        if row is not None:
            candidates[f"{method}:{reference}"] = {
                "median_best_lag_samples": row["median_best_lag_samples"],
                "mean_mass_in_reference": row["mean_mass_in_reference"],
                "mean_top20_dice": row["mean_top20_dice"],
            }
    artifacts = [
        "per_window_alignment.csv", "aggregate_alignment.csv", "window_qc.csv",
        "mask_alignment_examples.png", "mask_alignment_examples.pdf",
    ]
    summary = {
        "schema_version": 1,
        "status": "completed_exploratory_mask_audit",
        "training_integration": False,
        "dataset": "MIMIC-AFib",
        "dataset_split": "training_calibration_rows_only",
        "row_selection": indices.tolist(),
        "completed_windows": sum(row["status"] == "completed" for row in record_rows),
        "sampling_rate_hz": 128,
        "xresnet": {
            "target": "AFIB pre-sigmoid logit",
            "input_rate_hz": 100,
            "crop_samples": 250,
            "crop_starts_for_four_seconds": covering_crop_starts(400, 250, 125),
            "native_resolution_ms": {name: stride * 10.0 for name, (_index, stride) in XRESNET_LAYERS.items()},
            "projection": "explicit nominal feature centers at offset 0 and architecture stride",
            "legacy_last_layer_half_bin_offset_ms": 151.25,
            "input_adaptation": "single MIMIC ECG lead replicated to 12 channels",
            "input_unit_status": "source physical lead/unit not verified",
            "checkpoint_sha256": _sha256(args.xresnet_checkpoint.resolve()),
        },
        "ecgmamba": {
            "diagnostic_target": "NORM because AFIB is absent from the 44-class head",
            "semantic_outputs": ["background", "P", "QRS", "T"],
            "input_rate_hz": 500,
            "output_rate_hz": 500,
            "back_projection": "sample-center linear interpolation to 128 Hz",
            "input_adaptation": "single MIMIC ECG lead replicated to 12 channels",
            "checkpoint_sha256": _sha256(args.ecgmamba_checkpoint.resolve()),
        },
        "reference": {
            "algorithm": "NeuroKit2 DWT delineation on the exact RDDM-normalized/cleaned training target",
            "t_wave_interval": "symmetric onset estimate 2*T_peak-T_offset through T_offset",
            "role": "algorithmic temporal reference, not certified clinical ground truth",
        },
        "candidate_summary": candidates,
        "claim_boundary": (
            "This audit diagnoses temporal projection and cross-domain behavior only. Single-lead "
            "replication, unverified source units/lead identity, four-second windows, and algorithmic "
            "fiducials prevent treating either learned model as a validated MIMIC region-mask source."
        ),
        "mask_gate": {
            "xresnet": "rejected_for_training: coarse/heterogeneous residual timing plus 12-lead and unit mismatch",
            "ecgmamba": "not_yet_accepted: P/QRS timing promising but T timing fails and input is replicated single-lead",
            "pan_refined": "candidate_for_expanded_calibration: deterministic local-extremum correction only",
            "dwt_fiducial": "candidate_reference_mask: algorithmic P/QRS/QT intervals with explicit failure handling",
        },
        "source_sha256": {
            "audit_script": _sha256(Path(__file__).resolve()),
            "gradcam_module": _sha256(REPO_ROOT / "src/rcfm/interpretability/gradcam.py"),
        },
        "execution": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "requested_device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "rcfm_git_commit": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
            "rcfm_git_dirty": bool(_git_value(REPO_ROOT, "status", "--porcelain")),
        },
        "artifact_sha256": {name: _sha256(output_dir / name) for name in artifacts},
        "outputs": [*artifacts, "summary.json"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output_dir": str(output_dir), "completed_windows": summary["completed_windows"]}))
    return output_dir


if __name__ == "__main__":
    run(build_argparser().parse_args())
