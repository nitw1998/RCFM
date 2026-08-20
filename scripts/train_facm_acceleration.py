#!/usr/bin/env python3
"""Train an RCFM-OneStep deployment model with the FACM objective.

FACM is an external deployment-optimization method (Peng et al., 2025), not
an original RCFM contribution. The frozen multistep RCFM is used only while
training; saved student checkpoints contain no teacher parameters.
"""

from __future__ import annotations

import argparse
import json
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

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.rcfm.experiment import RunArtifacts, WandbLogger, atomic_json  # noqa: E402
from src.rcfm.one_step.facm_adapter import (  # noqa: E402
    FACM_UPSTREAM_COMMIT,
    build_facm_training_models,
)
from src.rcfm.one_step.facm_loss_1d import FACMLoss1D  # noqa: E402
from train_rcfm import build_datasets, set_deterministic  # noqa: E402


FROZEN_DATA_PROTOCOLS = {
    "PTBXL": {
        "task": "ecg2ecg", "dataset_version": "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1",
        "split_hash": "7784c98a2c8daccc23fc7cb0d47dc1933eeee77f120c05f8a24ac149cd8474f7",
        "normalization_id": "record_minmax_neg1_1_v1",
        "alignment_id": "ptbxl_official_folds_first4s_same_record_lead_II_to_other11_v1",
        "expected_train_windows": 17440, "expected_heldout_windows": 2193,
        "heldout_split": "val", "condition_lead_index": 1,
        "target_lead_indices": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    "CPSC2018": {
        "task": "ecg2ecg", "dataset_version": "cpsc2018-source-derived-all12lead-qc-v3-record-minmax-neg1-1",
        "split_hash": "35e0a796a60e4d6979b1f050048495fdf1826eedda7d40e47a56d4dcd5874223",
        "normalization_id": "record_minmax_neg1_1_v1",
        "alignment_id": "same_record_simultaneous_channels_first_4s_lead_II_to_other_11_v3",
        "expected_train_windows": 5487, "expected_heldout_windows": 686,
        "heldout_split": "val", "condition_lead_index": 1,
        "target_lead_indices": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    "MIMIC-AFib": {
        "task": "ppg2ecg", "dataset_version": "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1",
        "split_hash": "a7e388293adaa7b48d3493efc505dd8750520730cae9fd7649157866efa86a51",
        "normalization_id": "rddm_window_minmax_neg1_1_v1",
        "alignment_id": "paired_array_row_rddm_contract_zero_ppg_qc_v1",
        "expected_train_windows": 8400, "expected_heldout_windows": 1800,
        "heldout_split": "test", "condition_lead_index": None, "target_lead_indices": None,
    },
    "WESAD": {
        "task": "ppg2ecg", "dataset_version": "wesad-subject-fold1-linear-resample-window-minmax-v1",
        "split_hash": "0b90bffe7c3032243c803e534de3592618284d26792c2b612ce7af17a81a85cd",
        "normalization_id": "window_minmax_neg1_1_v1",
        "alignment_id": "native_common_start_same_window_no_delay_correction_subject_fold1_v1",
        "expected_train_windows": 17494, "expected_heldout_windows": 4213,
        "heldout_split": "test", "condition_lead_index": None, "target_lead_indices": None,
    },
    "mmECG": {
        "task": "rcg2ecg", "dataset_version": "mmecg-public-20221108-subject-split-window-minmax-v1",
        "split_hash": "e26fc81121cfd3b0e457608e37a7aa496ac7c48a0e0e7583a465bd069cc9da9f",
        "normalization_id": "window_minmax_neg1_1_v1",
        "alignment_id": "same_record_same_window_no_additional_phase_correction_subject_split_v1",
        "expected_train_windows": 9590, "expected_heldout_windows": 2877,
        "heldout_split": "test", "condition_lead_index": None, "target_lead_indices": None,
    },
}


def _repository_state() -> tuple[str, bool, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip()), status


def _mean_metrics(values: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(item)) for key, item in values.items()}


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher_checkpoint", default=os.environ.get("RCFM_FACM_TEACHER_CHECKPOINT"))
    parser.add_argument("--data_root", default=os.environ.get("RCFM_DATA_ROOT"))
    parser.add_argument("--output_dir", default=os.environ.get("RCFM_RUNS_ROOT"))
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--relaxation_power", type=float, default=0.5)
    parser.add_argument("--time_mean", type=float, default=0.8)
    parser.add_argument("--time_std", type=float, default=1.6)
    parser.add_argument("--robust_c", type=float, default=1e-3)
    parser.add_argument("--robust_power", type=float, default=0.5)
    parser.add_argument("--cm_weight", type=float, default=1.0)
    parser.add_argument("--fm_weight", type=float, default=1.0)
    parser.add_argument("--log_interval_steps", type=int, default=10)
    parser.add_argument("--max_train_records", type=int, default=None)
    parser.add_argument("--max_heldout_records", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb_project", default="RCFM")
    parser.add_argument("--wandb_group", default="mimic-afib-rcfm-facm")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--task", default="ppg2ecg")
    parser.add_argument("--datasets", default="MIMIC-AFib")
    parser.add_argument("--dataset_version", default=None)
    parser.add_argument("--split_hash", default=None)
    parser.add_argument("--normalization_id", default=None)
    parser.add_argument("--condition_unit", default=None)
    parser.add_argument("--target_unit", default=None)
    parser.add_argument("--alignment_id", default=None)
    parser.add_argument("--condition_lead", default="PPG")
    parser.add_argument("--target_lead", default="upstream_artifact_ecg_channel")
    parser.add_argument("--condition_lead_index", type=int, default=None)
    parser.add_argument("--target_lead_index", type=int, default=None)
    parser.add_argument("--target_lead_indices", default=None)
    parser.add_argument("--heldout_split", choices=["val", "test"], default="test")
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--expected_train_windows", type=int, default=8400)
    parser.add_argument("--expected_heldout_windows", type=int, default=1800)
    parser.add_argument("--expected_base_checkpoint_sha256", default=None)
    parser.add_argument("--facm_upstream_commit", default=FACM_UPSTREAM_COMMIT)
    parser.add_argument("--facm_time_strategy", default="expanded_time_interval")
    parser.add_argument("--facm_teacher_mode", default="conditional_no_cfg")
    parser.add_argument("--model_name", default="RCFM-OneStep")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config")
    known, _ = preliminary.parse_known_args(argv)
    parser = build_argparser()
    if known.config:
        defaults = json.loads(Path(known.config).read_text(encoding="utf-8"))
        if not isinstance(defaults, dict):
            raise ValueError("FACM config must contain a JSON object")
        known_fields = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - known_fields)
        if unknown:
            raise ValueError(f"FACM config has unknown keys: {unknown}")
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    required_paths = {"teacher_checkpoint": args.teacher_checkpoint, "data_root": args.data_root, "output_dir": args.output_dir}
    missing_paths = [key for key, value in required_paths.items() if not value]
    if missing_paths:
        raise ValueError("missing required FACM paths: " + ", ".join(missing_paths))
    if args.datasets not in FROZEN_DATA_PROTOCOLS:
        raise ValueError(f"unsupported FACM dataset: {args.datasets}")
    protocol = FROZEN_DATA_PROTOCOLS[args.datasets]
    expected = {
        **protocol,
        "facm_upstream_commit": FACM_UPSTREAM_COMMIT,
        "facm_time_strategy": "expanded_time_interval",
        "facm_teacher_mode": "conditional_no_cfg",
        "model_name": "RCFM-OneStep",
    }
    target_indices = args.target_lead_indices
    if isinstance(target_indices, str):
        target_indices = [int(item) for item in target_indices.split(",") if item]
    actual = vars(args) | {"target_lead_indices": target_indices}
    mismatched = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatched:
        raise ValueError("FACM config violates the frozen dataset protocol: " + ", ".join(mismatched))
    if not args.expected_base_checkpoint_sha256 or len(args.expected_base_checkpoint_sha256) != 64:
        raise ValueError("expected_base_checkpoint_sha256 is required")
    positive = (
        "epochs", "batch_size", "gradient_accumulation_steps",
        "lr", "adam_beta2", "adam_eps", "grad_clip", "save_every",
        "log_interval_steps", "window_size",
    )
    if any(float(getattr(args, key)) <= 0 for key in positive):
        raise ValueError("FACM numeric training settings must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    if not 0 < args.adam_beta2 < 1:
        raise ValueError("adam_beta2 must lie strictly between zero and one")
    if args.epochs % args.save_every != 0:
        raise ValueError("epochs must be divisible by save_every")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max_batches must be positive")


def train(args: argparse.Namespace) -> Path:
    validate_args(args)
    set_deterministic(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    teacher_path = Path(args.teacher_checkpoint).resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError("frozen base RCFM checkpoint does not exist")

    train_set, heldout_set = build_datasets(
        task=args.task,
        datasets=[args.datasets],
        data_root=args.data_root,
        window_size=args.window_size,
        normalization_id=args.normalization_id,
        condition_lead_index=args.condition_lead_index,
        target_lead_index=args.target_lead_index,
        target_lead_indices=args.target_lead_indices,
        heldout_split=args.heldout_split,
        max_train_records=args.max_train_records,
        max_heldout_records=args.max_heldout_records,
        return_region_mask_train=True,
    )
    expected_train = min(args.expected_train_windows, args.max_train_records) if args.max_train_records else args.expected_train_windows
    expected_heldout = min(args.expected_heldout_windows, args.max_heldout_records) if args.max_heldout_records else args.expected_heldout_windows
    if len(train_set) != expected_train or len(heldout_set) != expected_heldout:
        raise ValueError(f"{args.datasets} FACM data sizes disagree with the frozen split")
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    models = build_facm_training_models(
        teacher_path,
        device=device,
        expected_sha256=args.expected_base_checkpoint_sha256,
        expected_protocol={**FROZEN_DATA_PROTOCOLS[args.datasets], "datasets": args.datasets},
    )
    parameters = models.trainable_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, args.adam_beta2),
        eps=args.adam_eps,
    )
    objective = FACMLoss1D(
        relaxation_power=args.relaxation_power,
        robust_c=args.robust_c,
        robust_power=args.robust_power,
        cm_weight=args.cm_weight,
        fm_weight=args.fm_weight,
        time_mean=args.time_mean,
        time_std=args.time_std,
    )

    dataset_slug = args.datasets.lower().replace("-", "_")
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        f"%Y%m%dT%H%M%SZ_rcfm_onestep_{dataset_slug}_s{args.seed}"
    )
    run_dir = Path(args.output_dir) / "one_step" / args.task / args.datasets / run_id
    if run_dir.exists():
        raise FileExistsError("FACM run directory already exists")
    commit, dirty, status = _repository_state()
    resolved = vars(args).copy()
    for private_key in ("teacher_checkpoint", "data_root", "output_dir"):
        resolved[private_key] = f"<{private_key}>"
    resolved.update(
        {
            "run_id": run_id,
            "device": str(device),
            "git_commit": commit,
            "git_dirty": dirty,
            "base_checkpoint_sha256": models.base_checkpoint_sha256,
            "base_checkpoint_epoch": models.base_checkpoint["epoch"],
            "base_checkpoint_inference_nfe": models.base_checkpoint["config"]["inference_steps"],
            "student_initialization": "exact_base_rcfm_weights",
            "teacher_online_at_inference": False,
            "region_mask_used_by_facm": False,
            "ot_used_by_facm": False,
            "target_used_at_inference": False,
            "inference_nfe": 1,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "student_parameter_count": sum(parameter.numel() for parameter in parameters),
        }
    )
    environment = "\n".join(
        [
            f"python={platform.python_version()}",
            f"numpy={np.__version__}",
            f"torch={torch.__version__}",
            f"cuda={torch.version.cuda}",
            f"device={device}",
            f"gpu={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'unavailable'}",
        ]
    )
    artifacts = RunArtifacts(
        run_dir,
        resolved,
        {
            "schema_version": 1,
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "method_role": "external_facm_deployment_optimization",
        },
        environment,
        f"commit={commit}\ndirty={dirty}\n{status}",
    )
    logger = WandbLogger(
        mode=args.wandb_mode,
        run_dir=run_dir,
        config={**resolved, "hostname": socket.gethostname()},
        project=args.wandb_project,
        group=args.wandb_group,
        job_type="facm_distillation",
        run_name=args.wandb_run_name or run_id,
    )

    global_step = 0
    optimizer_step = 0
    start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def checkpoint_payload(epoch: int) -> dict:
        return {
            "schema_version": 1,
            "kind": "rcfm_facm_one_step",
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "student_flow_state": models.student_flow.state_dict(),
            "student_condition_state": models.student_condition.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": resolved,
            "normalization": models.base_checkpoint["normalization"],
            "output_spec": models.base_checkpoint["output_spec"],
            "facm_contract": {
                "upstream_commit": FACM_UPSTREAM_COMMIT,
                "teacher_checkpoint_sha256": models.base_checkpoint_sha256,
                "teacher_parameters_saved": False,
                "time_strategy": "expanded_time_interval",
                "cm_condition": "t",
                "fm_condition": "2-t",
                "teacher_velocity_condition": "t",
                "jvp_tangents": ["teacher_velocity", "one"],
                "loss": "cm_norm_l2_plus_fm_mse_plus_full_signal_cosine",
                "inference": "x1=z+student(z,t=0,E(source))",
                "nfe": 1,
            },
            "provenance": {
                "git_commit": commit,
                "git_dirty": dirty,
                "command": shlex.join(sys.argv),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        }

    def save(label: str, epoch: int) -> None:
        path = run_dir / f"checkpoint_{label}.pt"
        _atomic_torch_save(checkpoint_payload(epoch), path)
        artifacts.update_checkpoint_manifest(
            label,
            {"file": path.name, "epoch": epoch, "global_step": global_step, "teacher_parameters_saved": False},
        )

    try:
        optimizer.zero_grad(set_to_none=True)
        for epoch_index in range(args.epochs):
            models.student_flow.train()
            models.student_condition.train()
            models.teacher_flow.eval()
            models.teacher_condition.eval()
            epoch_values: dict[str, list[float]] = {}
            batch_count = 0
            progress = tqdm(loader, desc=f"FACM epoch {epoch_index + 1}/{args.epochs}")
            for batch_index, (target, condition, _region_mask) in enumerate(progress):
                target = target.float().to(device)
                condition = condition.float().to(device)
                with torch.no_grad():
                    teacher_conditions = models.teacher_condition(condition)
                student_conditions = models.student_condition(condition)
                try:
                    output = objective(
                        student_flow=models.student_flow,
                        teacher_flow=models.teacher_flow,
                        target=target,
                        student_conditions=student_conditions,
                        teacher_conditions=teacher_conditions,
                    )
                except RuntimeError as error:
                    if "forward AD" in str(error) or "jvp" in str(error).lower():
                        raise RuntimeError(
                            "RCFM attention is incompatible with the required FACM torch.func.jvp path"
                        ) from error
                    raise
                loss = output["loss"] / args.gradient_accumulation_steps
                loss.backward()
                batch_count += 1
                should_step = (
                    batch_count % args.gradient_accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                    or (args.max_batches is not None and batch_index + 1 >= args.max_batches)
                )
                gradient_norm = torch.tensor(float("nan"), device=device)
                if should_step:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError("FACM gradient norm is NaN or Inf")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                step_metrics = {
                    key: float(value.detach().cpu())
                    for key, value in output.items()
                    if key != "loss" and value.numel() == 1
                }
                if should_step:
                    step_metrics["train/gradient_norm"] = float(gradient_norm.detach().cpu())
                step_metrics.update(
                    {
                        "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "train/epoch": float(epoch_index + 1),
                        "train/global_step": float(global_step),
                        "train/optimizer_step": float(optimizer_step),
                    }
                )
                if not all(np.isfinite(value) for value in step_metrics.values()):
                    raise FloatingPointError("FACM training metrics contain NaN or Inf")
                for key, value in step_metrics.items():
                    epoch_values.setdefault(key, []).append(value)
                if global_step % args.log_interval_steps == 0:
                    logger.log(step_metrics, step=global_step)
                global_step += 1
                progress.set_postfix(loss=f"{step_metrics['train/total_loss']:.4f}")
                if args.max_batches is not None and batch_index + 1 >= args.max_batches:
                    break
            aggregated = _mean_metrics(epoch_values)
            artifacts.append_metrics("epoch_metrics.csv", epoch_index + 1, global_step, aggregated)
            logger.log(aggregated, step=global_step)
            save("latest", epoch_index + 1)
            if (epoch_index + 1) % args.save_every == 0:
                save(f"epoch_{epoch_index + 1}", epoch_index + 1)

        summary = {
            "status": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_duration_seconds": time.monotonic() - start,
            "global_steps": global_step,
            "optimizer_steps": optimizer_step,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "teacher_parameters_saved": False,
            "inference_nfe": 1,
        }
        artifacts.update_run_metadata(summary)
        atomic_json(run_dir / "facm_protocol.json", {"schema_version": 1, **resolved, **summary})
        logger.finish(summary)
        return run_dir
    except BaseException as error:
        message = str(error).splitlines()[0]
        for private_path in (str(args.teacher_checkpoint), str(args.data_root), str(args.output_dir)):
            message = message.replace(private_path, "<local-path>")
        failure = {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "exception_type": type(error).__name__,
            "exception_message": message,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        artifacts.update_run_metadata(failure)
        logger.finish(failure, exit_code=1)
        raise


if __name__ == "__main__":
    train(parse_args())
