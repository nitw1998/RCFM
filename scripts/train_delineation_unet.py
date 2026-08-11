"""Train the standalone PTB-XL+ dual-head ECG delineation ResUNet1D."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.rcfm.experiment import WandbLogger, atomic_json
from src.rcfm.interpretability.delineation_dataset import (
    EVENT_NAMES,
    WAVE_NAMES,
    PTBXLPlusDelineationDataset,
)
from src.rcfm.interpretability.delineation_unet import (
    DelineationResUNet1D,
    delineation_loss,
)
from src.rcfm.runtime import exception_summary, gradients_are_finite


METRIC_HEADERS = ("epoch", "global_step", "split", "metric", "value")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _repository_state(root: Path) -> tuple[str, bool, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    return commit, bool(status.strip()), status


def _append_metrics(
    path: Path, epoch: int, global_step: int, split: str, metrics: Mapping[str, float]
) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_HEADERS)
        if write_header:
            writer.writeheader()
        for metric, value in sorted(metrics.items()):
            writer.writerow(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "split": split,
                    "metric": metric,
                    "value": repr(float(value)),
                }
            )


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _loss_arguments(args: argparse.Namespace) -> dict[str, float]:
    return {
        "focal_gamma": args.focal_gamma,
        "region_bce_weight": args.region_bce_weight,
        "region_dice_weight": args.region_dice_weight,
        "fiducial_weight": args.fiducial_weight,
        "heatmap_positive_weight": args.heatmap_positive_weight,
    }


def _local_maxima(values: torch.Tensor, threshold: float, kernel_size: int = 15) -> list[int]:
    pooled = F.max_pool1d(
        values[None, None, :], kernel_size=kernel_size, stride=1, padding=kernel_size // 2
    )[0, 0]
    indices = torch.nonzero((values >= pooled) & (values >= threshold), as_tuple=False).flatten()
    return [int(value) for value in indices.cpu().tolist()]


def match_event_positions(
    predicted: list[int], reference: list[int], tolerance_samples: int
) -> tuple[list[int], int, int]:
    """Match sorted predicted/reference events without reusing a detection."""

    available = set(range(len(predicted)))
    errors: list[int] = []
    misses = 0
    for target in reference:
        candidates = sorted(available, key=lambda index: abs(predicted[index] - target))
        if not candidates or abs(predicted[candidates[0]] - target) > tolerance_samples:
            misses += 1
            continue
        chosen = candidates[0]
        errors.append(abs(predicted[chosen] - target))
        available.remove(chosen)
    return errors, misses, len(available)


@torch.no_grad()
def validate(
    model: DelineationResUNet1D,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    loss_values: list[float] = []
    intersections = torch.zeros(len(WAVE_NAMES), dtype=torch.float64)
    denominators = torch.zeros(len(WAVE_NAMES), dtype=torch.float64)
    event_errors: list[list[int]] = [[] for _ in EVENT_NAMES]
    event_targets = np.zeros(len(EVENT_NAMES), dtype=np.int64)
    event_misses = np.zeros(len(EVENT_NAMES), dtype=np.int64)
    event_extras = np.zeros(len(EVENT_NAMES), dtype=np.int64)
    for batch_index, cpu_batch in enumerate(loader):
        batch = _move_batch(cpu_batch, device)
        outputs = model(batch["signal"])
        losses = delineation_loss(outputs, batch, **_loss_arguments(args))
        loss_values.append(float(losses["loss"].cpu()))
        predicted_regions = torch.sigmoid(outputs["region_logits"]) >= args.region_threshold
        target_regions = batch["regions"] >= 0.5
        region_mask = batch["region_mask"] >= 0.5
        intersections += (
            (predicted_regions & target_regions & region_mask).sum(dim=(0, 2)).double().cpu()
        )
        denominators += (
            ((predicted_regions & region_mask).sum(dim=(0, 2)) + (target_regions & region_mask).sum(dim=(0, 2)))
            .double()
            .cpu()
        )
        predicted_heatmaps = torch.sigmoid(outputs["fiducial_logits"]).cpu()
        target_heatmaps = batch["heatmaps"].cpu()
        heatmap_masks = batch["heatmap_mask"].cpu() > 0.5
        predicted_pool = F.max_pool1d(predicted_heatmaps, 15, stride=1, padding=7)
        target_pool = F.max_pool1d(target_heatmaps, 15, stride=1, padding=7)
        predicted_peak_flags = (
            (predicted_heatmaps >= predicted_pool)
            & (predicted_heatmaps >= args.fiducial_threshold)
            & heatmap_masks
        )
        target_peak_flags = (
            (target_heatmaps >= target_pool) & (target_heatmaps >= 0.9) & heatmap_masks
        )
        for sample_index in range(predicted_heatmaps.shape[0]):
            for event_index in range(len(EVENT_NAMES)):
                valid = heatmap_masks[sample_index, event_index]
                if not bool(valid.any()):
                    continue
                predicted = torch.nonzero(
                    predicted_peak_flags[sample_index, event_index], as_tuple=False
                ).flatten().tolist()
                reference = torch.nonzero(
                    target_peak_flags[sample_index, event_index], as_tuple=False
                ).flatten().tolist()
                errors, misses, extras = match_event_positions(
                    predicted, reference, args.fiducial_match_tolerance_samples
                )
                event_errors[event_index].extend(errors)
                event_targets[event_index] += len(reference)
                event_misses[event_index] += misses
                event_extras[event_index] += extras
        if args.validation_max_batches is not None and batch_index + 1 >= args.validation_max_batches:
            break
    if not loss_values:
        raise ValueError("validation loader produced no batches")
    metrics: dict[str, float] = {"val/loss": float(np.mean(loss_values))}
    dice_values = []
    for index, name in enumerate(WAVE_NAMES):
        dice = float((2.0 * intersections[index] + 1.0) / (denominators[index] + 1.0))
        metrics[f"val/region_dice_{name}"] = dice
        dice_values.append(dice)
    metrics["val/region_macro_dice"] = float(np.mean(dice_values))
    all_errors: list[int] = []
    for index, name in enumerate(EVENT_NAMES):
        errors = event_errors[index]
        all_errors.extend(errors)
        metrics[f"val/{name}_mae_ms"] = (
            float(np.mean(errors) * 1000.0 / args.sample_rate_hz) if errors else float("nan")
        )
        metrics[f"val/{name}_miss_rate"] = (
            float(event_misses[index] / event_targets[index]) if event_targets[index] else float("nan")
        )
    fiducial_mae_ms = (
        float(np.mean(all_errors) * 1000.0 / args.sample_rate_hz) if all_errors else float("inf")
    )
    total_targets = int(event_targets.sum())
    metrics["val/fiducial_mae_ms"] = fiducial_mae_ms
    metrics["val/fiducial_miss_rate"] = (
        float(event_misses.sum() / total_targets) if total_targets else 1.0
    )
    total_predictions = total_targets - int(event_misses.sum()) + int(event_extras.sum())
    metrics["val/fiducial_extra_rate"] = (
        float(event_extras.sum() / total_predictions) if total_predictions > 0 else 0.0
    )
    metrics["val/model_selection_score"] = (
        metrics["val/region_macro_dice"]
        - args.selection_mae_weight_per_ms * fiducial_mae_ms
        - args.selection_miss_weight * metrics["val/fiducial_miss_rate"]
    )
    return metrics


def run(args: argparse.Namespace) -> Path:
    if not args.data_root or not args.waveform_root or not args.output_dir:
        raise ValueError("data_root, waveform_root, and output_dir are required")
    if args.sample_rate_hz != 128 or args.window_samples != 512:
        raise ValueError("the frozen RCFM transfer grid is 128 Hz and 512 samples")
    _set_deterministic(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    data_root = Path(args.data_root).resolve()
    waveform_root = Path(args.waveform_root).resolve()
    output_root = Path(args.output_dir).resolve()
    dataset_manifest_path = data_root / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest["dataset_version"] != args.dataset_version:
        raise ValueError("configured dataset version does not match delineation sidecar")
    if dataset_manifest["waveform_split_hash"] != args.waveform_split_hash:
        raise ValueError("configured waveform split hash does not match sidecar")
    if dataset_manifest["selected_leads"] != list(args.leads):
        raise ValueError("configured leads do not match sidecar")
    if dataset_manifest["crop_starts"] != list(args.crop_starts):
        raise ValueError("configured crop starts do not match sidecar")

    dataset_kwargs = {
        "sidecar_root": data_root,
        "waveform_root": waveform_root,
        "window_samples": args.window_samples,
        "crop_starts": args.crop_starts,
        "heatmap_sigma_samples": args.heatmap_sigma_samples,
        "heatmap_edge_ignore_samples": args.heatmap_edge_ignore_samples,
        "normalization_clip_z": args.normalization_clip_z,
    }
    train_set = PTBXLPlusDelineationDataset(
        split="train", max_records=args.max_train_records, **dataset_kwargs
    )
    validation_set = PTBXLPlusDelineationDataset(
        split="val", max_records=args.max_validation_records, **dataset_kwargs
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.validation_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    model = DelineationResUNet1D(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def learning_rate_multiplier(epoch_index: int) -> float:
        if args.warmup_epochs > 0 and epoch_index < args.warmup_epochs:
            return float(epoch_index + 1) / args.warmup_epochs
        progress = (epoch_index - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = LambdaLR(optimizer, learning_rate_multiplier)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / "ptbxl_plus_delineation" / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    repository_root = Path(__file__).resolve().parents[1]
    git_commit, git_dirty, git_status = _repository_state(repository_root)
    resolved = vars(args).copy()
    for private_key in ("data_root", "waveform_root", "output_dir", "config"):
        resolved.pop(private_key, None)
    resolved.update(
        {
            "run_id": run_id,
            "model": "DelineationResUNet1D",
            "output_stride": 1,
            "region_classes": list(WAVE_NAMES),
            "fiducial_events": list(EVENT_NAMES),
            "train_windows": len(train_set),
            "validation_windows": len(validation_set),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "dataset_manifest_sha256": _sha256(dataset_manifest_path),
            "split_and_eligibility_hash": dataset_manifest["split_and_eligibility_hash"],
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        }
    )
    atomic_json(run_dir / "resolved_config.json", resolved)
    atomic_json(
        run_dir / "run_metadata.json",
        {
            "schema_version": 1,
            "status": "running",
            "run_id": run_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "annotation_claim_boundary": "agreement with algorithm-generated PTB-XL+ ECGdeli fiducials",
            "mimic_test_used_for_training_or_model_selection": False,
        },
    )
    atomic_json(run_dir / "checkpoint_manifest.json", {"schema_version": 1, "checkpoints": {}})
    (run_dir / "git_state.txt").write_text(
        f"commit={git_commit}\ndirty={git_dirty}\n{git_status}", encoding="utf-8"
    )
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "unavailable"
    (run_dir / "environment.txt").write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                f"torch={torch.__version__}",
                f"cuda={torch.version.cuda}",
                f"device={device}",
                f"gpu={gpu_name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logger = WandbLogger(
        mode=args.wandb_mode,
        run_dir=run_dir,
        config={**resolved, "hostname": socket.gethostname(), "gpu_model": gpu_name},
        project=args.wandb_project,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        run_name=args.wandb_run_name or run_id,
    )
    metrics_path = run_dir / "metrics.csv"
    global_step = 0
    best_score = -float("inf")
    best_epoch = None
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_score = float(checkpoint["best_score"])
        best_epoch = checkpoint.get("best_epoch")

    def save_checkpoint(label: str, epoch: int) -> None:
        checkpoint_path = run_dir / f"checkpoint_{label}.pt"
        temporary_path = run_dir / f".{checkpoint_path.name}.tmp"
        torch.save(
            {
                "schema_version": 1,
                "kind": "ptbxl_plus_delineation_resunet1d",
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "config": resolved,
                "output_spec": {
                    "sample_rate_hz": args.sample_rate_hz,
                    "window_samples": args.window_samples,
                    "region_classes": list(WAVE_NAMES),
                    "fiducial_events": list(EVENT_NAMES),
                },
                "provenance": {
                    "dataset_manifest_sha256": resolved["dataset_manifest_sha256"],
                    "split_and_eligibility_hash": resolved["split_and_eligibility_hash"],
                    "git_commit": git_commit,
                    "command": shlex.join(sys.argv),
                },
            },
            temporary_path,
        )
        os.replace(temporary_path, checkpoint_path)
        manifest_path = run_dir / "checkpoint_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["checkpoints"][label] = {
            "file": checkpoint_path.name,
            "epoch": epoch,
            "global_step": global_step,
        }
        atomic_json(manifest_path, manifest)

    final_validation: dict[str, float] = {}
    started = time.monotonic()
    try:
        for epoch_index in range(start_epoch, args.epochs):
            model.train()
            accumulated: dict[str, list[float]] = {}
            progress = tqdm(train_loader, desc=f"epoch {epoch_index + 1}/{args.epochs}")
            for batch_index, cpu_batch in enumerate(progress):
                batch = _move_batch(cpu_batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    outputs = model(batch["signal"])
                    losses = delineation_loss(outputs, batch, **_loss_arguments(args))
                if not bool(torch.isfinite(losses["loss"])):
                    raise FloatingPointError("delineation forward loss contains NaN or Inf")
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                finite_gradients = gradients_are_finite(model.parameters())
                amp_overflow = bool(scaler.is_enabled() and not finite_gradients)
                if not finite_gradients and not amp_overflow:
                    raise FloatingPointError("delineation gradients contain NaN or Inf without AMP")
                gradient_norm = (
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    if finite_gradients
                    else None
                )
                scaler.step(optimizer)
                scaler.update()
                step_metrics = {
                    f"train/{key}": float(value.detach().cpu())
                    for key, value in losses.items()
                }
                if gradient_norm is not None:
                    step_metrics["train/gradient_norm"] = float(gradient_norm.detach().cpu())
                step_metrics["train/amp_overflow"] = float(amp_overflow)
                step_metrics["train/learning_rate"] = float(optimizer.param_groups[0]["lr"])
                if not all(np.isfinite(value) for value in step_metrics.values()):
                    raise FloatingPointError("training metrics contain NaN or Inf")
                for key, value in step_metrics.items():
                    accumulated.setdefault(key, []).append(value)
                if global_step % args.log_interval_steps == 0:
                    logger.log(step_metrics, step=global_step)
                global_step += 1
                progress.set_postfix(loss=f"{step_metrics['train/loss']:.4f}")
                if args.max_batches is not None and batch_index + 1 >= args.max_batches:
                    break
            scheduler.step()
            train_metrics = {key: float(np.mean(values)) for key, values in accumulated.items()}
            _append_metrics(metrics_path, epoch_index + 1, global_step, "train", train_metrics)
            logger.log(train_metrics, step=global_step)
            if (epoch_index + 1) % args.validation_interval_epochs == 0 or epoch_index + 1 == args.epochs:
                final_validation = validate(model, validation_loader, device, args)
                _append_metrics(metrics_path, epoch_index + 1, global_step, "val", final_validation)
                logger.log(final_validation, step=global_step)
                score = final_validation["val/model_selection_score"]
                if score > best_score:
                    best_score = score
                    best_epoch = epoch_index + 1
                    save_checkpoint("best", epoch_index + 1)
            save_checkpoint("latest", epoch_index + 1)
            if (epoch_index + 1) % args.save_every == 0:
                save_checkpoint(f"epoch_{epoch_index + 1}", epoch_index + 1)
            print(
                f"epoch={epoch_index + 1} train_loss={train_metrics['train/loss']:.6f} "
                f"val_macro_dice={final_validation.get('val/region_macro_dice', float('nan')):.5f} "
                f"val_fiducial_mae_ms={final_validation.get('val/fiducial_mae_ms', float('nan')):.3f}",
                flush=True,
            )
        summary = {
            "status": "completed",
            "best_epoch": best_epoch,
            "best_model_selection_score": best_score,
            "final_validation_macro_dice": final_validation.get("val/region_macro_dice"),
            "final_validation_fiducial_mae_ms": final_validation.get("val/fiducial_mae_ms"),
            "training_duration_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = run_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(summary)
        atomic_json(metadata_path, metadata)
        logger.finish(summary)
    except BaseException as error:
        failure = {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "exception_type": type(error).__name__,
            "exception_message": exception_summary(error),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = run_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(failure)
        atomic_json(metadata_path, metadata)
        logger.finish(failure, exit_code=1)
        raise
    return run_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--data_root", default=os.environ.get("PTBXL_PLUS_DELINEATION_ROOT"))
    parser.add_argument("--waveform_root", default=os.environ.get("PTBXL_WAVEFORM_ROOT"))
    parser.add_argument("--output_dir", default=os.environ.get("RCFM_RUNS_ROOT"))
    parser.add_argument("--run_id")
    parser.add_argument("--resume")
    parser.add_argument("--dataset_version")
    parser.add_argument("--waveform_split_hash")
    parser.add_argument("--leads", nargs="+", default=list(("I", "II", "III", "aVR", "aVL", "aVF")))
    parser.add_argument("--sample_rate_hz", type=int, default=128)
    parser.add_argument("--window_samples", type=int, default=512)
    parser.add_argument("--crop_starts", nargs="+", type=int, default=[0, 384, 768])
    parser.add_argument("--normalization_clip_z", type=float, default=5.0)
    parser.add_argument("--heatmap_sigma_samples", type=float, default=2.0)
    parser.add_argument("--heatmap_edge_ignore_samples", type=int, default=8)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--validation_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--region_bce_weight", type=float, default=1.0)
    parser.add_argument("--region_dice_weight", type=float, default=1.0)
    parser.add_argument("--fiducial_weight", type=float, default=0.5)
    parser.add_argument("--heatmap_positive_weight", type=float, default=4.0)
    parser.add_argument("--region_threshold", type=float, default=0.5)
    parser.add_argument("--fiducial_threshold", type=float, default=0.3)
    parser.add_argument("--fiducial_match_tolerance_samples", type=int, default=16)
    parser.add_argument("--selection_mae_weight_per_ms", type=float, default=0.001)
    parser.add_argument("--selection_miss_weight", type=float, default=0.1)
    parser.add_argument("--validation_interval_epochs", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--log_interval_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb_project", default="RCFM")
    parser.add_argument("--wandb_group", default="ptbxl-plus-delineation")
    parser.add_argument("--wandb_job_type", default="mask-backbone-train")
    parser.add_argument("--wandb_run_name")
    parser.add_argument("--max_batches", type=int)
    parser.add_argument("--validation_max_batches", type=int)
    parser.add_argument("--max_train_records", type=int)
    parser.add_argument("--max_validation_records", type=int)
    return parser


def parse_args_with_config(argv: list[str] | None = None) -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config")
    known, _ = preliminary.parse_known_args(argv)
    parser = build_argparser()
    if known.config:
        defaults = json.loads(Path(known.config).read_text(encoding="utf-8"))
        if not isinstance(defaults, dict):
            raise ValueError("config must contain a JSON object")
        known_keys = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - known_keys)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    missing = [name for name in ("dataset_version", "waveform_split_hash") if not getattr(args, name)]
    if missing:
        parser.error("missing required experiment metadata: " + ", ".join(missing))
    return args


if __name__ == "__main__":
    run(parse_args_with_config())
