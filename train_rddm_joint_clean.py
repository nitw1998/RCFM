"""Train RDDM jointly on pre-cleaned MIMIC-AFib and WESAD windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from diffusion import RDDM
from model import ConditionNet, DiffusionUNetCrossAttention
from src.rcfm.checkpoint import capture_rng_states
from src.rcfm.experiment import RunArtifacts, WandbLogger
from train_rcfm import set_deterministic


DATASETS = ("MIMIC-AFib", "WESAD")
UPSTREAM_COMMIT = "7d5348843c3985c211a23ae5105a2d9497d5156a"
DATASET_VERSION = "rddm-mimic-wesad-joint-official-loader-clean-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CleanRDDMDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.ecg = np.load(root / "ecg_train_4sec.npy", mmap_mode="r", allow_pickle=False)
        self.ppg = np.load(root / "ppg_train_4sec.npy", mmap_mode="r", allow_pickle=False)
        self.masks = np.load(root / "region_masks_train.npy", mmap_mode="r", allow_pickle=False)
        if self.ecg.shape != self.ppg.shape or self.ecg.ndim != 2 or self.ecg.shape[1] != 512:
            raise ValueError(f"invalid cleaned paired arrays under {root}")
        if self.masks.shape != (len(self.ecg), 1, 512):
            raise ValueError(f"invalid cleaned ROI masks under {root}")
        if not all(array.dtype == np.float32 for array in (self.ecg, self.ppg, self.masks)):
            raise ValueError("cleaned RDDM arrays must be float32")

    def __getitem__(self, index: int):
        return (
            np.asarray(self.ecg[index]).reshape(1, 512).copy(),
            np.asarray(self.ppg[index]).reshape(1, 512).copy(),
            np.asarray(self.masks[index]).copy(),
        )

    def __len__(self) -> int:
        return len(self.ecg)


def _validate_artifact(root: Path) -> tuple[dict[str, object], dict[str, int]]:
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing cleaned artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "completed"
        or manifest.get("dataset_version") != DATASET_VERSION
        or manifest.get("joint_training_datasets") != list(DATASETS)
        or manifest.get("provenance", {}).get("upstream_commit") != UPSTREAM_COMMIT
        or manifest.get("preprocessing", {}).get("test_mask_generated") is not False
    ):
        raise ValueError("cleaned artifact violates the joint RDDM contract")
    counts: dict[str, int] = {}
    for dataset in DATASETS:
        record = manifest.get("datasets", {}).get(dataset, {})
        splits = record.get("splits", {})
        counts[dataset] = int(splits.get("train", {}).get("windows", -1))
        for split_name, names in {
            "train": ("ecg_train_4sec.npy", "ppg_train_4sec.npy", "region_masks_train.npy"),
            "test": ("ecg_test_4sec.npy", "ppg_test_4sec.npy"),
        }.items():
            outputs = splits.get(split_name, {}).get("outputs", {})
            for name in names:
                path = root / dataset / name
                if not path.is_file() or outputs.get(name) != _sha256(path):
                    raise ValueError(f"cleaned artifact hash mismatch: {dataset}/{name}")
        if (root / dataset / "region_masks_test.npy").exists():
            raise ValueError(f"held-out target mask must not exist: {dataset}")
    return manifest, counts


def _lr_factor(epoch_index: int, epochs: int, warmup_epochs: int) -> float:
    """Paper-inferred linear warm-up followed by cosine decay."""
    if epochs <= 0 or warmup_epochs <= 0 or warmup_epochs >= epochs:
        raise ValueError("require 0 < warmup_epochs < epochs")
    if epoch_index < warmup_epochs:
        return float(epoch_index + 1) / float(warmup_epochs)
    progress = (epoch_index + 1 - warmup_epochs) / float(epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", default=os.environ.get("RDDM_JOINT_DATA_ROOT"))
    parser.add_argument("--output_dir", default=os.environ.get("RCFM_RUNS_ROOT"))
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=80)
    parser.add_argument("--nT", type=int, default=10)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.2)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--condition_drop_probability", type=float, default=0.0)
    parser.add_argument("--ddpm_loss_weight", type=float, default=100.0)
    parser.add_argument("--region_loss_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_interval_steps", type=int, default=10)
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb_project", default="RCFM")
    parser.add_argument("--wandb_group", default="rddm-mimic-wesad-joint-clean-v1")
    parser.add_argument("--upstream_commit", default=UPSTREAM_COMMIT)
    parser.add_argument("--dataset_version", default=DATASET_VERSION)
    parser.add_argument("--expected_mimic_train_windows", type=int, default=8400)
    parser.add_argument("--expected_wesad_train_windows", type=int, default=17494)
    parser.add_argument("--max_batches", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def parse_args_with_config(argv: list[str] | None = None) -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", required=True)
    known, _ = preliminary.parse_known_args(argv)
    parser = build_argparser()
    defaults = json.loads(Path(known.config).read_text(encoding="utf-8"))
    valid = {action.dest for action in parser._actions}
    if unknown := sorted(set(defaults) - valid):
        raise ValueError(f"unknown joint RDDM config keys: {unknown}")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    validate_config(args)
    return args


def validate_config(args: argparse.Namespace) -> None:
    expected = {
        "epochs": 1000,
        "batch_size": 512,
        "warmup_epochs": 20,
        "nT": 10,
        "beta_start": 1e-4,
        "beta_end": 0.2,
        "condition_drop_probability": 0.0,
        "ddpm_loss_weight": 100.0,
        "region_loss_weight": 1.0,
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset_version": DATASET_VERSION,
    }
    bad = [key for key, value in expected.items() if getattr(args, key) != value]
    if bad:
        raise ValueError("joint RDDM config violates the frozen upstream-derived contract: " + ", ".join(bad))
    if args.save_every <= 0 or args.num_workers < 0 or args.learning_rate <= 0:
        raise ValueError("save_every/learning_rate must be positive and num_workers nonnegative")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max_batches must be positive")


def train(args: argparse.Namespace) -> Path:
    validate_config(args)
    if not args.data_root or not args.output_dir:
        raise ValueError("RDDM_JOINT_DATA_ROOT/data_root and RCFM_RUNS_ROOT/output_dir are required")
    data_root = Path(args.data_root).resolve()
    manifest, counts = _validate_artifact(data_root)
    expected_counts = {
        "MIMIC-AFib": args.expected_mimic_train_windows,
        "WESAD": args.expected_wesad_train_windows,
    }
    if counts != expected_counts and args.max_batches is None:
        raise ValueError(f"joint training counts changed: {counts} != {expected_counts}")
    dataset = ConcatDataset([CleanRDDMDataset(data_root / name) for name in DATASETS])
    set_deterministic(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    rddm = RDDM(
        eps_model=DiffusionUNetCrossAttention(512, 1, str(device), num_heads=args.attention_heads),
        region_model=DiffusionUNetCrossAttention(512, 1, str(device), num_heads=args.attention_heads),
        betas=(args.beta_start, args.beta_end),
        n_T=args.nT,
    ).to(device)
    condition_1, condition_2 = ConditionNet().to(device), ConditionNet().to(device)
    if args.data_parallel:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError("data_parallel=true requires at least two visible CUDA devices")
        rddm = torch.nn.DataParallel(rddm)
        condition_1 = torch.nn.DataParallel(condition_1)
        condition_2 = torch.nn.DataParallel(condition_2)
    parameters = [*rddm.parameters(), *condition_1.parameters(), *condition_2.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _lr_factor(epoch, args.epochs, args.warmup_epochs),
    )

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir).resolve() / "ppg2ecg" / "MIMIC-AFib__WESAD" / run_id
    resolved = vars(args).copy()
    resolved.update(
        {
            "run_id": run_id,
            "datasets": list(DATASETS),
            "train_windows_by_dataset": counts,
            "total_train_windows": len(dataset),
            "model_family": "RDDM",
            "comparison_role": "two_dataset_joint_upstream_loader_reproduction",
            "scheduler": "paper_inferred_linear_warmup_cosine_v1",
            "scheduler_fidelity": "adaptation-required; upstream lr_scheduler.py is absent",
            "artifact_manifest_sha256": _sha256(data_root / "dataset_manifest.json"),
            "model_parameter_count": sum(p.numel() for p in parameters if p.requires_grad),
        }
    )
    artifacts = RunArtifacts(
        run_dir,
        resolved,
        {
            "schema_version": 1,
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "heldout_evaluated_during_training": False,
        },
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                f"torch={torch.__version__}",
                f"cuda={torch.version.cuda}",
                f"device={device}",
                f"visible_cuda_devices={torch.cuda.device_count()}",
            ]
        ),
        f"upstream_commit={UPSTREAM_COMMIT}\nhost={socket.gethostname()}",
    )
    logger = WandbLogger(
        mode=args.wandb_mode,
        run_dir=run_dir,
        config=resolved,
        project=args.wandb_project,
        group=args.wandb_group,
        job_type="joint-train",
        run_name=run_id,
    )
    global_step, started = 0, time.monotonic()
    try:
        for epoch in range(1, args.epochs + 1):
            rddm.train(); condition_1.train(); condition_2.train()
            epoch_values: list[tuple[float, float, float]] = []
            progress = tqdm(loader, desc=f"joint RDDM epoch {epoch}/{args.epochs}")
            for batch_index, (target, condition, mask) in enumerate(progress):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                target = target.float().to(device); condition = condition.float().to(device); mask = mask.float().to(device)
                optimizer.zero_grad(set_to_none=True)
                encoded_1 = condition_1(condition, drop_prob=args.condition_drop_probability)
                encoded_2 = condition_2(condition, drop_prob=args.condition_drop_probability)
                ddpm_loss, region_loss = rddm(
                    x=target, cond1=encoded_1, cond2=encoded_2, patch_labels=mask
                )
                # DataParallel gathers one scalar loss per device; reduce those
                # device-local means before backward, matching upstream's
                # ``loss.mean().backward()`` behavior.
                weighted_ddpm = (args.ddpm_loss_weight * ddpm_loss).mean()
                weighted_region = (args.region_loss_weight * region_loss).mean()
                loss = weighted_ddpm + weighted_region
                loss.backward(); optimizer.step()
                values = (float(weighted_ddpm.detach()), float(weighted_region.detach()), float(loss.detach()))
                if not np.all(np.isfinite(values)):
                    raise FloatingPointError("joint RDDM loss contains NaN or Inf")
                epoch_values.append(values); global_step += 1
                if global_step % args.log_interval_steps == 0:
                    logger.log(
                        {"train/ddpm_loss_weighted": values[0], "train/region_loss_weighted": values[1], "train/total_loss": values[2], "train/epoch": epoch},
                        step=global_step,
                    )
                progress.set_postfix(loss=f"{values[2]:.4f}")
            scheduler.step()
            means = np.mean(np.asarray(epoch_values), axis=0)
            epoch_metrics = {
                "train/ddpm_loss_weighted": float(means[0]),
                "train/region_loss_weighted": float(means[1]),
                "train/total_loss": float(means[2]),
                "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            artifacts.append_metrics("epoch_metrics.csv", epoch, global_step, epoch_metrics)
            logger.log(epoch_metrics, step=global_step)
            if epoch % args.save_every == 0 or epoch == args.epochs:
                def state(module):
                    return module.module.state_dict() if isinstance(module, torch.nn.DataParallel) else module.state_dict()
                checkpoint = {
                    "schema_version": 1,
                    "kind": "joint_mimic_wesad_rddm_reproduction",
                    "epoch": epoch,
                    "global_step": global_step,
                    "rddm_state": state(rddm),
                    "condition_1_state": state(condition_1),
                    "condition_2_state": state(condition_2),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "rng_states": capture_rng_states(),
                    "config": resolved,
                    "dataset_manifest": manifest,
                    "provenance": {"upstream_commit": UPSTREAM_COMMIT, "command": shlex.join(sys.argv)},
                }
                path = run_dir / f"checkpoint_epoch_{epoch}.pt"
                temporary = path.with_suffix(".pt.tmp")
                torch.save(checkpoint, temporary); os.replace(temporary, path)
                artifacts.update_checkpoint_manifest("latest", {"file": path.name, "epoch": epoch, "global_step": global_step})
        summary = {
            "status": "completed",
            "final_epoch": args.epochs,
            "global_step": global_step,
            "training_duration_seconds": time.monotonic() - started,
            "heldout_evaluated_during_training": False,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        artifacts.update_run_metadata(summary); logger.finish(summary)
    except BaseException as error:
        failure = {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "exception_type": type(error).__name__,
            "exception_message": str(error).splitlines()[0],
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        artifacts.update_run_metadata(failure); logger.finish(failure, exit_code=1)
        raise
    return run_dir


if __name__ == "__main__":
    train(parse_args_with_config())
