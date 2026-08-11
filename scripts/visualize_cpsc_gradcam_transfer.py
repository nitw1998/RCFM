"""Visualize a PTB-XL all-statements AFIB Grad-CAM on CPSC2018 records."""

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
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from scipy.special import expit


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocess_cpsc2018 import discover_record_paths, load_reference
from src.rcfm.interpretability.gradcam import (
    gradcam_1d,
    normalize_soft_mask,
    stitch_temporal_cams,
    validation_crop_starts,
)
from src.rcfm.interpretability.ptbxl_benchmark_compat import load_xresnet1d101


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_cpsc_record(
    path: Path,
    source_rate_hz: int,
    output_rate_hz: int,
    window_seconds: int,
    minimum_lead_std: float,
) -> tuple[np.ndarray, int]:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=["ECG"])
    if "ECG" not in payload or not hasattr(payload["ECG"], "data"):
        raise ValueError("missing ECG.data")
    values = np.asarray(payload["ECG"].data, dtype=np.float64)
    required = source_rate_hz * window_seconds
    if values.ndim != 2 or values.shape[0] != 12 or values.shape[1] < required:
        raise ValueError("record is not a complete 12-lead 10-second window")
    window = values[:, :required]
    if not np.all(np.isfinite(window)):
        raise ValueError("record contains NaN or Inf")
    output = resample_poly(window, output_rate_hz, source_rate_hz, axis=1, padtype="line")
    expected = output_rate_hz * window_seconds
    if output.shape != (12, expected) or not np.all(np.isfinite(output)):
        raise ValueError("resampled record has an invalid shape or value")
    if np.any(np.std(output, axis=1, dtype=np.float64) < minimum_lead_std):
        raise ValueError("record contains a constant or near-constant lead")
    return output.T.astype(np.float32), int(values.shape[1])


def _select_records(
    cpsc_root: Path,
    seed: int,
    samples_per_group: int,
    source_rate_hz: int,
    output_rate_hz: int,
    window_seconds: int,
    minimum_lead_std: float,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows = load_reference(cpsc_root / "REFERENCE.csv")
    paths = discover_record_paths(cpsc_root)
    groups = {
        "AF": [row for row in rows if 2 in row["labels"]],
        "non-AF": [row for row in rows if row["labels"] == (1,)],
    }
    selected: list[dict[str, object]] = []
    rejected: dict[str, int] = {}
    for group_index, (group, candidates) in enumerate(groups.items()):
        rng = np.random.default_rng(seed + group_index)
        order = rng.permutation(len(candidates))
        accepted = 0
        rejected[group] = 0
        for candidate_index in order:
            row = candidates[int(candidate_index)]
            record_id = str(row["record_id"])
            try:
                waveform, source_samples = _load_cpsc_record(
                    paths[record_id],
                    source_rate_hz,
                    output_rate_hz,
                    window_seconds,
                    minimum_lead_std,
                )
            except ValueError:
                rejected[group] += 1
                continue
            selected.append(
                {
                    "group": group,
                    "record_id": record_id,
                    "labels": tuple(int(value) for value in row["labels"]),
                    "waveform": waveform,
                    "source_samples": source_samples,
                }
            )
            accepted += 1
            if accepted == samples_per_group:
                break
        if accepted != samples_per_group:
            raise RuntimeError(f"only {accepted} QC-valid records found for {group}")
    return selected, rejected


def _infer_record(
    model: torch.nn.Module,
    waveform: np.ndarray,
    scaler_mean: float,
    scaler_scale: float,
    target_index: int,
    crop_length: int,
    crop_stride: int,
    device: torch.device,
) -> dict[str, object]:
    standardized = (waveform - scaler_mean) / scaler_scale
    starts = validation_crop_starts(len(standardized), crop_length, crop_stride)
    crop_cams: list[np.ndarray] = []
    crop_logits: list[np.ndarray] = []
    target_layer = model[7][-1]
    for start in starts:
        crop = standardized[start : start + crop_length].T.copy()
        inputs = torch.from_numpy(crop).unsqueeze(0).to(device=device, dtype=torch.float32)
        cam, logits = gradcam_1d(model, target_layer, inputs, target_index)
        crop_cams.append(cam)
        crop_logits.append(logits)
    stitched = stitch_temporal_cams(crop_cams, starts, len(standardized))
    mask, degenerate = normalize_soft_mask(stitched)
    probabilities = expit(np.stack(crop_logits, axis=0))
    aggregate_probabilities = np.max(probabilities, axis=0)
    return {
        "mask": mask,
        "raw_cam": stitched,
        "crop_starts": np.asarray(starts, dtype=np.int64),
        "crop_logits": np.stack(crop_logits).astype(np.float32),
        "probabilities": aggregate_probabilities.astype(np.float32),
        "degenerate": bool(degenerate),
    }


def _write_figure(records: list[dict[str, object]], output_dir: Path, plot_lead: int) -> None:
    groups = ["AF", "non-AF"]
    grouped = {group: [record for record in records if record["group"] == group] for group in groups}
    rows = max(len(grouped[group]) for group in groups)
    figure, axes = plt.subplots(rows, 2, figsize=(14, 2.7 * rows), squeeze=False)
    image = None
    for column, group in enumerate(groups):
        for row_index, record in enumerate(grouped[group]):
            axis = axes[row_index, column]
            waveform = np.asarray(record["waveform"])[:, plot_lead]
            mask = np.asarray(record["mask"])
            time = np.arange(len(waveform), dtype=np.float32) / 100.0
            lower, upper = np.quantile(waveform, [0.01, 0.99])
            margin = max(float(upper - lower) * 0.18, 0.05)
            y_min, y_max = float(lower - margin), float(upper + margin)
            image = axis.imshow(
                mask[np.newaxis, :],
                extent=(0.0, 10.0, y_min, y_max),
                aspect="auto",
                origin="lower",
                cmap="YlOrRd",
                vmin=0.0,
                vmax=1.0,
                alpha=0.62,
                interpolation="bilinear",
            )
            axis.plot(time, waveform, color="#17202a", linewidth=0.85)
            axis.set_xlim(0.0, 10.0)
            axis.set_ylim(y_min, y_max)
            axis.grid(axis="x", color="white", alpha=0.45, linewidth=0.6)
            title_group = "AF" if group == "AF" else "Non-AF (normal)"
            axis.set_title(
                f"{title_group} {row_index + 1} | AFIB p={record['afib_probability']:.3f} | "
                f"top={record['top_class']}"
            )
            axis.set_ylabel(f"Lead {LEAD_NAMES[plot_lead]} (source units)")
            if row_index == rows - 1:
                axis.set_xlabel("Time (s)")
    figure.suptitle(
        "Cross-domain AFIB Grad-CAM: PTB-XL all-statements XResNet1D-101 -> CPSC2018",
        fontsize=14,
        y=0.995,
    )
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.015)
        colorbar.set_label("Per-record normalized soft Grad-CAM")
    figure.text(
        0.5,
        0.008,
        "Exploratory classifier substitution; fixed AFIB logit, 2.5 s crops, overlap-mean stitching, no threshold.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.07, right=0.91, top=0.92, bottom=0.08, hspace=0.48, wspace=0.20)
    figure.savefig(output_dir / "gradcam_af_vs_non_af.png", dpi=240)
    figure.savefig(output_dir / "gradcam_af_vs_non_af.pdf")
    plt.close(figure)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark_code_root", type=Path, required=True)
    parser.add_argument("--mlb", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--cpsc_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--samples_per_group", type=int, default=3)
    parser.add_argument("--source_rate_hz", type=int, default=500)
    parser.add_argument("--classifier_rate_hz", type=int, default=100)
    parser.add_argument("--window_seconds", type=int, default=10)
    parser.add_argument("--crop_samples", type=int, default=250)
    parser.add_argument("--crop_stride", type=int, default=125)
    parser.add_argument("--minimum_lead_std", type=float, default=1e-6)
    parser.add_argument("--plot_lead", choices=LEAD_NAMES, default="II")
    return parser


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.classifier_rate_hz * args.window_seconds != 1000:
        raise ValueError("this frozen visualization requires a 10-second, 100 Hz classifier input")
    if args.crop_samples != 250 or args.crop_stride != 125:
        raise ValueError("this frozen visualization requires 250-sample crops with stride 125")
    if args.samples_per_group < 1:
        raise ValueError("samples_per_group must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    mlb = _load_pickle(args.mlb.resolve())
    scaler = _load_pickle(args.scaler.resolve())
    classes = np.asarray(mlb.classes_, dtype=str)
    matches = np.flatnonzero(classes == "AFIB")
    if len(classes) != 71 or len(matches) != 1:
        raise ValueError("label binarizer must contain one AFIB class among 71 all-statement classes")
    target_index = int(matches[0])
    scaler_mean = float(np.asarray(scaler.mean_).reshape(-1)[0])
    scaler_scale = float(np.asarray(scaler.scale_).reshape(-1)[0])
    if not np.isfinite(scaler_mean) or not np.isfinite(scaler_scale) or scaler_scale <= 0:
        raise ValueError("standard scaler parameters are invalid")

    model = load_xresnet1d101(args.benchmark_code_root, args.checkpoint, map_location="cpu")
    model.eval().to(device)
    selected, rejected = _select_records(
        args.cpsc_root.resolve(),
        args.seed,
        args.samples_per_group,
        args.source_rate_hz,
        args.classifier_rate_hz,
        args.window_seconds,
        args.minimum_lead_std,
    )
    for record in selected:
        result = _infer_record(
            model,
            np.asarray(record["waveform"]),
            scaler_mean,
            scaler_scale,
            target_index,
            args.crop_samples,
            args.crop_stride,
            device,
        )
        record.update(result)
        record["afib_probability"] = float(result["probabilities"][target_index])
        top_index = int(np.argmax(result["probabilities"]))
        record["top_class_index"] = top_index
        record["top_class"] = str(classes[top_index])

    plot_lead = LEAD_NAMES.index(args.plot_lead)
    _write_figure(selected, output_dir, plot_lead)

    with (output_dir / "selected_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "sample_index",
                "record_id",
                "cpsc_labels",
                "source_samples",
                "afib_probability",
                "top_class_index",
                "top_class",
                "cam_degenerate",
            ],
        )
        writer.writeheader()
        group_counts = {"AF": 0, "non-AF": 0}
        for record in selected:
            group = str(record["group"])
            group_counts[group] += 1
            writer.writerow(
                {
                    "group": group,
                    "sample_index": group_counts[group],
                    "record_id": record["record_id"],
                    "cpsc_labels": ";".join(str(value) for value in record["labels"]),
                    "source_samples": record["source_samples"],
                    "afib_probability": f"{record['afib_probability']:.9g}",
                    "top_class_index": record["top_class_index"],
                    "top_class": record["top_class"],
                    "cam_degenerate": record["degenerate"],
                }
            )

    np.savez_compressed(
        output_dir / "gradcam_results.npz",
        record_ids=np.asarray([record["record_id"] for record in selected], dtype="U16"),
        groups=np.asarray([record["group"] for record in selected], dtype="U8"),
        cpsc_labels=np.asarray([",".join(map(str, record["labels"])) for record in selected], dtype="U16"),
        waveforms=np.stack([record["waveform"] for record in selected]).astype(np.float32),
        masks=np.stack([record["mask"] for record in selected]).astype(np.float32),
        raw_cams=np.stack([record["raw_cam"] for record in selected]).astype(np.float32),
        crop_starts=np.stack([record["crop_starts"] for record in selected]).astype(np.int64),
        crop_logits=np.stack([record["crop_logits"] for record in selected]).astype(np.float32),
        aggregate_probabilities=np.stack([record["probabilities"] for record in selected]).astype(np.float32),
        class_names=classes.astype("U32"),
    )

    benchmark_root = args.benchmark_code_root.resolve().parent
    artifact_names = [
        "gradcam_af_vs_non_af.png",
        "gradcam_af_vs_non_af.pdf",
        "gradcam_results.npz",
        "selected_records.csv",
    ]
    protocol = {
        "status": "exploratory_classifier_substitution",
        "reviewer_scope": "Reviewer 2 Comment 4 cross-domain Grad-CAM robustness",
        "claim_boundary": (
            "XResNet1D-101 visualization only; it does not reproduce or replace the unavailable "
            "manuscript ResNet-50 protocol and is not quantitative mask validation."
        ),
        "classifier": {
            "architecture": "fastai_xresnet1d101",
            "training_dataset": "PTB-XL",
            "training_task": "all statements",
            "target_class": "AFIB",
            "target_index": target_index,
            "target_layer": "model[7][-1] (last residual block)",
            "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
            "mlb_sha256": _sha256(args.mlb.resolve()),
            "scaler_sha256": _sha256(args.scaler.resolve()),
            "benchmark_commit": _git_value(benchmark_root, "rev-parse", "HEAD"),
            "benchmark_license": "GPL-3.0",
            "strict_checkpoint_load": True,
        },
        "input": {
            "dataset": "CPSC2018",
            "lead_order": LEAD_NAMES,
            "source_rate_hz": args.source_rate_hz,
            "classifier_rate_hz": args.classifier_rate_hz,
            "window_seconds": args.window_seconds,
            "window_rule": "first 10 seconds; no target- or CAM-based selection",
            "resampling": "scipy.signal.resample_poly with padtype=line",
            "standardization": {
                "rule": "(x - PTB-XL training scalar mean) / scalar scale",
                "mean": scaler_mean,
                "scale": scaler_scale,
            },
            "qc": "finite, >=10 seconds, all 12 resampled lead std >= configured threshold",
            "minimum_lead_std": args.minimum_lead_std,
        },
        "selection": {
            "seed": args.seed,
            "samples_per_group": args.samples_per_group,
            "AF": "CPSC label 2 appears in the record labels",
            "non_AF": "CPSC labels exactly (1,), i.e. normal without AF",
            "ordering": "seeded permutation within metadata-defined group, then input QC only",
            "rejected_before_quota": rejected,
            "mask_or_prediction_used_for_selection": False,
        },
        "gradcam": {
            "score": "pre-sigmoid AFIB logit for both groups",
            "channel_weights": "temporal mean of activation gradients",
            "positive_evidence": "ReLU",
            "crop_samples": args.crop_samples,
            "crop_stride": args.crop_stride,
            "crop_prediction_aggregation": "per-class maximum probability",
            "crop_cam_interpolation": "linear, align_corners=False",
            "crop_cam_stitching": "mean in overlapping temporal regions before normalization",
            "normalization": "independent per-record min-max to [0,1]",
            "threshold": None,
        },
        "figure": {
            "plot_lead": args.plot_lead,
            "classifier_input_leads": 12,
            "record_ids_shown": False,
        },
        "artifact_sha256": {
            name: _sha256(output_dir / name) for name in artifact_names
        },
        "execution": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "requested_device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "gradient_determinism": (
                "Sample selection and crop coordinates are deterministic. Under PyTorch 2.0 CUDA, "
                "adaptive max-pool backward has no deterministic implementation, so Grad-CAM values "
                "are not guaranteed to be bitwise identical across executions."
            ),
            "rcfm_git_commit": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
            "rcfm_git_dirty": bool(_git_value(REPO_ROOT, "status", "--porcelain")),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "outputs": [*artifact_names, "protocol.json"],
    }
    with (output_dir / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output_dir": str(output_dir), "records": len(selected)}, sort_keys=True))
    return output_dir


if __name__ == "__main__":
    run(build_argparser().parse_args())
