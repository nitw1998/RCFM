"""Evaluate a frozen PTB-XL+ delineation ResUNet on official fold 10."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.rcfm.experiment import atomic_json
from src.rcfm.interpretability.delineation_dataset import (
    EVENT_NAMES,
    WAVE_NAMES,
    PTBXLPlusDelineationDataset,
)
from src.rcfm.interpretability.delineation_unet import DelineationResUNet1D, delineation_loss
from scripts.train_delineation_unet import match_event_positions


COLORS = {"p": "#2A9D8F", "qrs": "#D55E00", "t": "#3572B0"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _classification_metrics(tp: int, fp: int, fn: int, valid: int) -> dict[str, float]:
    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "valid_samples": int(valid),
    }


def _select_visualization_indices(dataset: PTBXLPlusDelineationDataset, count: int) -> list[int]:
    if count <= 0:
        return []
    candidates = []
    for index, (record_index, lead_index, _crop_start) in enumerate(dataset.samples.tolist()):
        if bool(np.asarray(dataset.wave_valid[record_index, lead_index]).all()):
            candidates.append(index)
    if not candidates:
        raise ValueError("test split contains no windows with P/QRS/T supervision")
    positions = np.linspace(0, len(candidates) - 1, min(count, len(candidates)), dtype=int)
    return [candidates[int(position)] for position in positions]


def _merge_training_metrics(paths: Iterable[Path]) -> dict[str, dict[int, float]]:
    merged: dict[str, dict[int, float]] = defaultdict(dict)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                merged[row["metric"]][int(row["epoch"])] = float(row["value"])
    return dict(merged)


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_training_curves(metrics: Mapping[str, Mapping[int, float]], output_dir: Path) -> None:
    required = ("train/loss", "val/region_macro_dice", "val/fiducial_mae_ms")
    if not all(name in metrics for name in required):
        raise ValueError("training metrics do not contain the required curve fields")
    _set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.25), constrained_layout=True)

    epochs = sorted(metrics["train/loss"])
    axes[0].plot(epochs, [metrics["train/loss"][epoch] for epoch in epochs], color="#222222")
    if "val/loss" in metrics:
        val_epochs = sorted(metrics["val/loss"])
        axes[0].plot(
            val_epochs,
            [metrics["val/loss"][epoch] for epoch in val_epochs],
            color="#777777",
            linestyle="--",
        )
    axes[0].set(title="Optimization", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(["Train", "Validation"], frameon=False)

    for wave in (*WAVE_NAMES, "macro"):
        metric = f"val/region_dice_{wave}" if wave != "macro" else "val/region_macro_dice"
        curve_epochs = sorted(metrics[metric])
        axes[1].plot(
            curve_epochs,
            [metrics[metric][epoch] for epoch in curve_epochs],
            color=COLORS.get(wave, "#222222"),
            linestyle="--" if wave == "macro" else "-",
            label=wave.upper(),
        )
    axes[1].set(title="Fold-9 region agreement", xlabel="Epoch", ylabel="Dice", ylim=(0.75, 0.95))
    axes[1].legend(frameon=False, ncol=2)

    fid_epochs = sorted(metrics["val/fiducial_mae_ms"])
    axes[2].plot(
        fid_epochs,
        [metrics["val/fiducial_mae_ms"][epoch] for epoch in fid_epochs],
        color="#6A3D9A",
        label="MAE",
    )
    axes[2].set(title="Fold-9 fiducials", xlabel="Epoch", ylabel="MAE (ms)")
    best_epoch = max(metrics["val/model_selection_score"], key=metrics["val/model_selection_score"].get)
    for axis in axes:
        axis.axvline(best_epoch, color="#C43C39", linestyle=":", linewidth=1.0)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[2].text(best_epoch + 0.8, axes[2].get_ylim()[1], f"selected: {best_epoch}", va="top")
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"training_validation_curves.{suffix}", dpi=300)
    plt.close(fig)


def _region_spans(binary: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(binary, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _plot_examples(examples: list[dict[str, object]], sample_rate_hz: int, output_dir: Path) -> None:
    _set_plot_style()
    fig, axes = plt.subplots(
        len(examples), 2, figsize=(7.16, 1.55 * len(examples)), squeeze=False
    )
    for row, example in enumerate(examples):
        signal = np.asarray(example["signal"])
        reference = np.asarray(example["reference"])
        probabilities = np.asarray(example["probabilities"])
        time_axis = np.arange(signal.size) / sample_rate_hz
        left, right = axes[row]
        left.plot(time_axis, signal, color="#222222", linewidth=0.8)
        ymin, ymax = float(signal.min()), float(signal.max())
        for wave_index, wave in enumerate(WAVE_NAMES):
            for start, stop in _region_spans(reference[wave_index] >= 0.5):
                left.axvspan(start / sample_rate_hz, stop / sample_rate_hz, color=COLORS[wave], alpha=0.18)
        left.set(ylabel=f"Example {chr(65 + row)}\nnormalized ECG", xlim=(0, signal.size / sample_rate_hz), ylim=(ymin, ymax))
        left.spines[["top", "right"]].set_visible(False)
        left.grid(axis="x", color="#EEEEEE", linewidth=0.4)

        for wave_index, wave in enumerate(WAVE_NAMES):
            right.plot(time_axis, probabilities[wave_index], color=COLORS[wave], label=f"{wave.upper()} prob.")
            right.step(
                time_axis,
                reference[wave_index],
                where="mid",
                color=COLORS[wave],
                linestyle=":",
                linewidth=0.8,
            )
        right.axhline(0.5, color="#777777", linewidth=0.6, linestyle="--")
        right.set(ylabel="Mask probability", xlim=(0, signal.size / sample_rate_hz), ylim=(-0.02, 1.02))
        right.spines[["top", "right"]].set_visible(False)
        right.grid(axis="y", color="#EEEEEE", linewidth=0.4)
        if row == len(examples) - 1:
            left.set_xlabel("Time (s)")
            right.set_xlabel("Time (s)")
    handles = [Patch(facecolor=COLORS[name], alpha=0.3, label=f"{name.upper()} reference") for name in WAVE_NAMES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.08, top=0.93, hspace=0.42, wspace=0.22)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"fold10_mask_examples.{suffix}", dpi=300)
    plt.close(fig)


def _loss_arguments(config: Mapping[str, object]) -> dict[str, float]:
    return {
        name: float(config[name])
        for name in (
            "focal_gamma",
            "region_bce_weight",
            "region_dice_weight",
            "fiducial_weight",
            "heatmap_positive_weight",
        )
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Path:
    checkpoint_path = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    waveform_root = Path(args.waveform_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("kind") != "ptbxl_plus_delineation_resunet1d":
        raise ValueError("checkpoint is not a PTB-XL+ delineation ResUNet")
    config = checkpoint["config"]
    if int(checkpoint["epoch"]) != int(args.expected_epoch):
        raise ValueError("checkpoint epoch does not match the predeclared selected epoch")
    manifest_path = data_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset_version"] != config["dataset_version"]:
        raise ValueError("checkpoint and sidecar dataset versions differ")
    if manifest["waveform_split_hash"] != config["waveform_split_hash"]:
        raise ValueError("checkpoint and waveform split hashes differ")
    if manifest["split_and_eligibility_hash"] != checkpoint["provenance"]["split_and_eligibility_hash"]:
        raise ValueError("checkpoint and sidecar eligibility hashes differ")
    if _sha256(manifest_path) != checkpoint["provenance"]["dataset_manifest_sha256"]:
        raise ValueError("sidecar manifest checksum differs from checkpoint provenance")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset = PTBXLPlusDelineationDataset(
        data_root,
        waveform_root,
        "test",
        window_samples=int(config["window_samples"]),
        crop_starts=config["crop_starts"],
        heatmap_sigma_samples=float(config["heatmap_sigma_samples"]),
        heatmap_edge_ignore_samples=int(config["heatmap_edge_ignore_samples"]),
        normalization_clip_z=float(config["normalization_clip_z"]),
    )
    if args.expected_test_windows is not None and len(dataset) != args.expected_test_windows:
        raise ValueError(f"expected {args.expected_test_windows} test windows, found {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    model = DelineationResUNet1D(int(config["base_channels"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    region_threshold = float(config["region_threshold"])
    fiducial_threshold = float(config["fiducial_threshold"])
    tolerance = int(config["fiducial_match_tolerance_samples"])
    tp = np.zeros(len(WAVE_NAMES), dtype=np.int64)
    fp = np.zeros(len(WAVE_NAMES), dtype=np.int64)
    fn = np.zeros(len(WAVE_NAMES), dtype=np.int64)
    valid_samples = np.zeros(len(WAVE_NAMES), dtype=np.int64)
    target_positive = np.zeros(len(WAVE_NAMES), dtype=np.int64)
    predicted_positive = np.zeros(len(WAVE_NAMES), dtype=np.int64)
    event_errors: list[list[int]] = [[] for _ in EVENT_NAMES]
    event_targets = np.zeros(len(EVENT_NAMES), dtype=np.int64)
    event_misses = np.zeros(len(EVENT_NAMES), dtype=np.int64)
    event_extras = np.zeros(len(EVENT_NAMES), dtype=np.int64)
    per_window_dice: list[float] = []
    weighted_loss_sum = 0.0
    evaluated_windows = 0
    example_indices = _select_visualization_indices(dataset, args.examples)
    examples: dict[int, dict[str, object]] = {}

    for batch_index, cpu_batch in enumerate(tqdm(loader, desc="fold-10 delineation")):
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in cpu_batch.items()
        }
        outputs = model(batch["signal"])
        losses = delineation_loss(outputs, batch, **_loss_arguments(config))
        batch_size = int(batch["signal"].shape[0])
        weighted_loss_sum += float(losses["loss"].cpu()) * batch_size
        probabilities = torch.sigmoid(outputs["region_logits"])
        predicted = probabilities >= region_threshold
        target = batch["regions"] >= 0.5
        mask = batch["region_mask"] >= 0.5
        for wave_index in range(len(WAVE_NAMES)):
            wave_predicted = predicted[:, wave_index] & mask[:, wave_index]
            wave_target = target[:, wave_index] & mask[:, wave_index]
            tp[wave_index] += int((wave_predicted & wave_target).sum().cpu())
            fp[wave_index] += int((wave_predicted & ~wave_target & mask[:, wave_index]).sum().cpu())
            fn[wave_index] += int((~wave_predicted & wave_target).sum().cpu())
            valid_samples[wave_index] += int(mask[:, wave_index].sum().cpu())
            target_positive[wave_index] += int(wave_target.sum().cpu())
            predicted_positive[wave_index] += int(wave_predicted.sum().cpu())

        sample_tp = (predicted & target & mask).sum(dim=2).float()
        sample_denominator = ((predicted & mask).sum(dim=2) + (target & mask).sum(dim=2)).float()
        class_valid = mask.any(dim=2)
        sample_dice = torch.where(
            sample_denominator > 0,
            2.0 * sample_tp / sample_denominator.clamp_min(1.0),
            torch.ones_like(sample_denominator),
        )
        macro = (sample_dice * class_valid).sum(dim=1) / class_valid.sum(dim=1).clamp_min(1)
        per_window_dice.extend(float(value) for value in macro.cpu().tolist())

        predicted_heatmaps = torch.sigmoid(outputs["fiducial_logits"]).cpu()
        target_heatmaps = cpu_batch["heatmaps"]
        heatmap_masks = cpu_batch["heatmap_mask"] > 0.5
        predicted_pool = F.max_pool1d(predicted_heatmaps, 15, stride=1, padding=7)
        target_pool = F.max_pool1d(target_heatmaps, 15, stride=1, padding=7)
        predicted_flags = (
            (predicted_heatmaps >= predicted_pool)
            & (predicted_heatmaps >= fiducial_threshold)
            & heatmap_masks
        )
        target_flags = (target_heatmaps >= target_pool) & (target_heatmaps >= 0.9) & heatmap_masks
        for sample_index in range(batch_size):
            for event_index in range(len(EVENT_NAMES)):
                if not bool(heatmap_masks[sample_index, event_index].any()):
                    continue
                predicted_events = torch.nonzero(
                    predicted_flags[sample_index, event_index], as_tuple=False
                ).flatten().tolist()
                reference_events = torch.nonzero(
                    target_flags[sample_index, event_index], as_tuple=False
                ).flatten().tolist()
                errors, misses, extras = match_event_positions(
                    predicted_events, reference_events, tolerance
                )
                event_errors[event_index].extend(errors)
                event_targets[event_index] += len(reference_events)
                event_misses[event_index] += misses
                event_extras[event_index] += extras

        start = evaluated_windows
        stop = start + batch_size
        for global_index in example_indices:
            if start <= global_index < stop:
                local_index = global_index - start
                examples[global_index] = {
                    "signal": cpu_batch["signal"][local_index, 0].numpy(),
                    "reference": cpu_batch["regions"][local_index].numpy(),
                    "probabilities": probabilities[local_index].cpu().numpy(),
                }
        evaluated_windows += batch_size
        if args.max_batches is not None and batch_index + 1 >= args.max_batches:
            break

    if not evaluated_windows:
        raise ValueError("test loader produced no windows")
    complete = evaluated_windows == len(dataset)
    if not complete and args.max_batches is None:
        raise RuntimeError("evaluation ended before the complete fold-10 split")

    region_rows = []
    for index, name in enumerate(WAVE_NAMES):
        row = {"wave": name}
        row.update(_classification_metrics(int(tp[index]), int(fp[index]), int(fn[index]), int(valid_samples[index])))
        row.update(
            {
                "reference_occupancy": float(target_positive[index] / valid_samples[index]),
                "predicted_occupancy": float(predicted_positive[index] / valid_samples[index]),
                "threshold": region_threshold,
            }
        )
        region_rows.append(row)

    fiducial_rows = []
    for index, name in enumerate(EVENT_NAMES):
        matched = len(event_errors[index])
        predictions = matched + int(event_extras[index])
        mae_ms = (
            float(np.mean(event_errors[index]) * 1000.0 / config["sample_rate_hz"])
            if event_errors[index]
            else float("nan")
        )
        fiducial_rows.append(
            {
                "event": name,
                "reference_events": int(event_targets[index]),
                "matched_events": matched,
                "mae_ms": mae_ms,
                "miss_rate": float(event_misses[index] / event_targets[index]),
                "extra_rate": float(event_extras[index] / predictions) if predictions else float("nan"),
                "threshold": fiducial_threshold,
                "match_tolerance_ms": float(tolerance * 1000.0 / config["sample_rate_hz"]),
            }
        )

    _write_csv(output_dir / "region_metrics.csv", region_rows)
    _write_csv(output_dir / "fiducial_metrics.csv", fiducial_rows)
    dice_array = np.asarray(per_window_dice, dtype=np.float64)
    summary = {
        "status": "completed" if complete else "capped_smoke_only",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "selection_best_epoch": int(checkpoint["best_epoch"]),
        "selection_best_score": float(checkpoint["best_score"]),
        "evaluated_windows": evaluated_windows,
        "available_test_windows": len(dataset),
        "test_loss": weighted_loss_sum / evaluated_windows,
        "region_macro_dice": float(np.mean([row["dice"] for row in region_rows])),
        "region_macro_iou": float(np.mean([row["iou"] for row in region_rows])),
        "per_window_macro_dice_median": float(np.median(dice_array)),
        "per_window_macro_dice_q25": float(np.quantile(dice_array, 0.25)),
        "per_window_macro_dice_q75": float(np.quantile(dice_array, 0.75)),
        "fiducial_macro_mae_ms": float(np.mean([row["mae_ms"] for row in fiducial_rows])),
        "fiducial_micro_mae_ms": float(
            np.mean([error for errors in event_errors for error in errors])
            * 1000.0
            / config["sample_rate_hz"]
        ),
        "fiducial_micro_miss_rate": float(event_misses.sum() / event_targets.sum()),
        "fiducial_micro_extra_rate": float(
            event_extras.sum() / (sum(len(errors) for errors in event_errors) + event_extras.sum())
        ),
        "annotation_claim_boundary": "agreement with algorithm-generated PTB-XL+ ECGdeli fiducials",
    }
    atomic_json(output_dir / "summary.json", summary)

    training_metrics = _merge_training_metrics(Path(path) for path in args.training_metrics)
    if training_metrics:
        _plot_training_curves(training_metrics, output_dir)
    complete_examples = [examples[index] for index in example_indices if index in examples]
    if complete_examples:
        _plot_examples(complete_examples, int(config["sample_rate_hz"]), output_dir)

    protocol = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "PTB-XL+ ECGdeli limb-lead delineation",
        "split": "official_fold_10",
        "selection_split": "official_fold_9",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection": "maximum fold-9 model_selection_score",
        "checkpoint_best_score": float(checkpoint["best_score"]),
        "training_git_commit": checkpoint["provenance"]["git_commit"],
        "evaluation_script_sha256": _sha256(Path(__file__).resolve()),
        "training_metrics_sha256": {
            Path(path).name + f"#{index}": _sha256(Path(path).resolve())
            for index, path in enumerate(args.training_metrics)
        },
        "dataset_manifest_sha256": _sha256(manifest_path),
        "split_and_eligibility_hash": manifest["split_and_eligibility_hash"],
        "waveform_split_hash": manifest["waveform_split_hash"],
        "test_windows": len(dataset),
        "evaluated_windows": evaluated_windows,
        "region_threshold": region_threshold,
        "fiducial_threshold": fiducial_threshold,
        "fiducial_match_tolerance_samples": tolerance,
        "sample_rate_hz": int(config["sample_rate_hz"]),
        "window_samples": int(config["window_samples"]),
        "visualization_selection": "equally spaced among P/QRS/T-supervised fold-10 windows before prediction",
        "visualization_count": len(complete_examples),
        "identifiers_saved": False,
        "raw_arrays_saved": False,
        "mimic_used": False,
        "claim_boundary": "algorithm-generated annotation agreement; no manual clinical ground truth",
        "device": str(device),
        "max_batches": args.max_batches,
    }
    atomic_json(output_dir / "protocol.json", protocol)
    artifact_files = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json"
    )
    atomic_json(
        output_dir / "artifact_manifest.json",
        {
            "schema_version": 1,
            "files": {path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size} for path in artifact_files},
        },
    )
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--waveform_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--training_metrics", nargs="*", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--expected_epoch", type=int, default=13)
    parser.add_argument("--expected_test_windows", type=int, default=39555)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--max_batches", type=int)
    return parser


if __name__ == "__main__":
    evaluate(build_argparser().parse_args())
