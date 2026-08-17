"""Train a lightweight deterministic 1-D CNN direct-regression baseline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.rcfm.baselines import DirectRegressionCNN
from src.rcfm.checkpoint import capture_rng_states, restore_rng_states
from src.rcfm.experiment import RunArtifacts, WandbLogger
from src.rcfm.metrics.paper_statistics import per_sample_pearson
from train_rcfm import build_datasets, parse_datasets, parse_lead_indices, set_deterministic


MIMIC_SPLIT_HASH = "a7e388293adaa7b48d3493efc505dd8750520730cae9fd7649157866efa86a51"
PTBXL_SPLIT_HASH = "7784c98a2c8daccc23fc7cb0d47dc1933eeee77f120c05f8a24ac149cd8474f7"
CPSC2018_SPLIT_HASH = "35e0a796a60e4d6979b1f050048495fdf1826eedda7d40e47a56d4dcd5874223"
WESAD_SPLIT_HASH = "0b90bffe7c3032243c803e534de3592618284d26792c2b612ce7af17a81a85cd"
MMECG_SPLIT_HASH = "e26fc81121cfd3b0e457608e37a7aa496ac7c48a0e0e7583a465bd069cc9da9f"
PTBXL_TARGETS = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", default=os.environ.get("RCFM_DATA_ROOT"))
    parser.add_argument("--output_dir", default=os.environ.get("RCFM_RUNS_ROOT"))
    parser.add_argument("--run_id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--task", choices=("ppg2ecg", "ecg2ecg", "rcg2ecg"))
    parser.add_argument("--datasets")
    parser.add_argument("--dataset_version")
    parser.add_argument("--split_hash")
    parser.add_argument("--normalization_id")
    parser.add_argument("--condition_unit")
    parser.add_argument("--target_unit")
    parser.add_argument("--alignment_id")
    parser.add_argument("--condition_lead")
    parser.add_argument("--target_lead")
    parser.add_argument("--condition_lead_index", type=int)
    parser.add_argument("--target_lead_indices")
    parser.add_argument("--heldout_split", choices=("val", "test"))
    parser.add_argument("--heldout_role", choices=("validation", "upstream_test_final_only"))
    parser.add_argument("--expected_train_windows", type=int)
    parser.add_argument("--expected_heldout_windows", type=int)
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dilations", default="1,2,4,8,16,32,64")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--validation_interval_epochs", type=int, default=25)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device")
    parser.add_argument("--log_interval_steps", type=int, default=10)
    parser.add_argument("--max_train_records", type=int)
    parser.add_argument("--max_heldout_records", type=int)
    parser.add_argument("--max_batches", type=int)
    parser.add_argument("--validation_max_batches", type=int)
    parser.add_argument("--wandb_mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--wandb_project", default="RCFM")
    parser.add_argument("--wandb_group", default="direct-cnn-regression")
    parser.add_argument("--wandb_job_type", default="train")
    parser.add_argument("--wandb_run_name")
    return parser


def parse_args_with_config(argv: list[str] | None = None) -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", required=True)
    known, _ = preliminary.parse_known_args(argv)
    parser = build_argparser()
    defaults = json.loads(Path(known.config).read_text(encoding="utf-8"))
    if not isinstance(defaults, dict):
        raise ValueError("direct CNN config must contain a JSON object")
    fields = {action.dest for action in parser._actions}
    if unknown := sorted(set(defaults) - fields):
        raise ValueError(f"direct CNN config has unknown keys: {unknown}")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    validate_config(args)
    return args


def _dilations(value: str | list[int]) -> tuple[int, ...]:
    items = value if isinstance(value, list) else str(value).split(",")
    result = tuple(int(item) for item in items)
    if not result or any(item <= 0 for item in result):
        raise ValueError("dilations must be positive")
    return result


def validate_config(args: argparse.Namespace) -> None:
    datasets = parse_datasets(args.datasets or "")
    targets = parse_lead_indices(args.target_lead_indices)
    if args.task == "ppg2ecg" and datasets == ["MIMIC-AFib"]:
        expected = {
            "dataset_version": "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1",
            "split_hash": MIMIC_SPLIT_HASH,
            "normalization_id": "rddm_window_minmax_neg1_1_v1",
            "alignment_id": "paired_array_row_rddm_contract_zero_ppg_qc_v1",
            "heldout_split": "test", "heldout_role": "upstream_test_final_only",
            "expected_train_windows": 8400, "expected_heldout_windows": 1800,
        }
        if args.condition_lead_index is not None or targets is not None:
            raise ValueError("MIMIC direct CNN must not declare ECG lead indices")
    elif args.task == "ecg2ecg" and datasets == ["PTBXL"]:
        expected = {
            "dataset_version": "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1",
            "split_hash": PTBXL_SPLIT_HASH,
            "normalization_id": "record_minmax_neg1_1_v1",
            "alignment_id": "ptbxl_official_folds_first4s_same_record_lead_II_to_other11_v1",
            "heldout_split": "val", "heldout_role": "validation",
            "expected_train_windows": 17440, "expected_heldout_windows": 2193,
        }
        if args.condition_lead_index != 1 or targets != PTBXL_TARGETS:
            raise ValueError("PTB-XL direct CNN requires lead II to the other 11 leads")
    elif args.task == "ecg2ecg" and datasets == ["CPSC2018"]:
        expected = {
            "dataset_version": "cpsc2018-source-derived-all12lead-qc-v3-record-minmax-neg1-1",
            "split_hash": CPSC2018_SPLIT_HASH,
            "normalization_id": "record_minmax_neg1_1_v1",
            "alignment_id": "same_record_simultaneous_channels_first_4s_lead_II_to_other_11_v3",
            "heldout_split": "val", "heldout_role": "validation",
            "expected_train_windows": 5487, "expected_heldout_windows": 686,
        }
        if args.condition_lead_index != 1 or targets != PTBXL_TARGETS:
            raise ValueError("CPSC2018 direct CNN requires lead II to the other 11 leads")
    elif args.task == "ppg2ecg" and datasets == ["WESAD"]:
        expected = {
            "dataset_version": "wesad-subject-fold1-linear-resample-window-minmax-v1",
            "split_hash": WESAD_SPLIT_HASH,
            "normalization_id": "window_minmax_neg1_1_v1",
            "alignment_id": "native_common_start_same_window_no_delay_correction_subject_fold1_v1",
            "heldout_split": "test", "heldout_role": "upstream_test_final_only",
            "expected_train_windows": 17494, "expected_heldout_windows": 4213,
        }
        if args.condition_lead_index is not None or targets is not None:
            raise ValueError("WESAD direct CNN must not declare ECG lead indices")
    elif args.task == "rcg2ecg" and datasets == ["mmECG"]:
        expected = {
            "dataset_version": "mmecg-public-20221108-subject-split-window-minmax-v1",
            "split_hash": MMECG_SPLIT_HASH,
            "normalization_id": "window_minmax_neg1_1_v1",
            "alignment_id": "same_record_same_window_no_additional_phase_correction_subject_split_v1",
            "heldout_split": "test", "heldout_role": "upstream_test_final_only",
            "expected_train_windows": 9590, "expected_heldout_windows": 2877,
        }
        if args.condition_lead_index is not None or targets is not None:
            raise ValueError("mmECG direct CNN must not declare ECG lead indices")
    else:
        raise ValueError("unsupported direct CNN dataset/task contract")
    bad = [name for name, value in expected.items() if getattr(args, name) != value]
    if bad:
        raise ValueError("direct CNN config violates frozen data contract: " + ", ".join(bad))
    required = ("condition_unit", "target_unit", "condition_lead", "target_lead")
    if missing := [name for name in required if not getattr(args, name)]:
        raise ValueError("direct CNN config missing metadata: " + ", ".join(missing))
    if args.window_size != 4 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("window_size must be 4 and epochs/batch_size must be positive")
    if args.width <= 0 or args.save_every <= 0 or args.validation_interval_epochs <= 0:
        raise ValueError("width and epoch intervals must be positive")
    if args.epochs % args.save_every:
        raise ValueError("epochs must be divisible by save_every")
    _dilations(args.dilations)
    for name in ("max_train_records", "max_heldout_records", "max_batches", "validation_max_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")


def _repository_state(root: Path) -> tuple[str, bool, str]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout
    return commit, bool(status.strip()), status


def _scheduler(optimizer, epochs: int, warmup: int):
    effective = min(warmup, max(epochs - 1, 0))
    if effective == 0:
        return CosineAnnealingLR(optimizer, T_max=epochs)
    return SequentialLR(
        optimizer,
        [LinearLR(optimizer, start_factor=1e-6, total_iters=effective), CosineAnnealingLR(optimizer, T_max=max(epochs - effective, 1))],
        milestones=[effective],
    )


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int | None) -> dict[str, float]:
    was_training = model.training
    model.eval()
    references, predictions = [], []
    for index, batch in enumerate(loader):
        target, condition = batch[:2]
        prediction = model(condition.float().to(device))
        references.append(target.numpy())
        predictions.append(prediction.cpu().numpy())
        if max_batches is not None and index + 1 >= max_batches:
            break
    if was_training:
        model.train()
    if not references:
        raise ValueError("validation loader produced no batches")
    reference, prediction = np.concatenate(references), np.concatenate(predictions)
    error = prediction - reference
    pearson = per_sample_pearson(reference, prediction)
    metrics = {
        "val/rmse": float(np.sqrt(np.mean(np.square(error)))),
        "val/mae": float(np.mean(np.abs(error))),
        "val/pearson_sample_median": float(np.nanmedian(pearson)),
        "val/pearson_usable": float(np.isfinite(pearson).sum()),
        "val/num_samples": float(len(reference)),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("direct CNN validation metrics must be finite")
    return metrics


def train(args: argparse.Namespace) -> Path:
    validate_config(args)
    if not args.data_root or not args.output_dir:
        raise ValueError("--data_root and --output_dir (or their environment variables) are required")
    set_deterministic(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    datasets = parse_datasets(args.datasets)
    targets = parse_lead_indices(args.target_lead_indices)
    train_set, heldout_set = build_datasets(
        args.task, datasets, args.data_root, args.window_size,
        normalization_id=args.normalization_id,
        condition_lead_index=args.condition_lead_index,
        target_lead_indices=targets,
        heldout_split=args.heldout_split,
        max_train_records=args.max_train_records,
        max_heldout_records=args.max_heldout_records,
        return_region_mask_train=False,
    )
    expected_train = min(args.expected_train_windows, args.max_train_records) if args.max_train_records else args.expected_train_windows
    expected_heldout = min(args.expected_heldout_windows, args.max_heldout_records) if args.max_heldout_records else args.expected_heldout_windows
    if len(train_set) != expected_train or len(heldout_set) != expected_heldout:
        raise ValueError("dataset sizes disagree with the frozen direct CNN protocol")
    first_target, first_condition = train_set[0][:2]
    output_channels, input_channels = int(first_target.shape[0]), int(first_condition.shape[0])
    model = DirectRegressionCNN(input_channels, output_channels, args.width, _dilations(args.dilations)).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = _scheduler(optimizer, args.epochs, args.warmup_epochs)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    heldout_loader = DataLoader(heldout_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir) / args.task / datasets[0] / run_id
    if run_dir.exists() and args.resume is None:
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    repo_root = Path(__file__).resolve().parent
    commit, dirty, status = _repository_state(repo_root)
    resolved = vars(args).copy()
    resolved.update({
        "datasets": datasets, "run_id": run_id, "model_family": "DirectCNN",
        "comparison_role": "deterministic_lightweight_direct_regression",
        "loss": "pointwise_mse", "stochastic": False, "sampling_steps": 0,
        "uses_test_target_at_inference": False, "uses_region_mask": False, "uses_ot": False,
        "sample_rate_hz": 128, "signal_length": 512,
        "input_channels": input_channels, "output_channels": output_channels,
        "dilations": list(_dilations(args.dilations)), "receptive_field_samples": model.receptive_field,
        "model_parameter_count": sum(parameter.numel() for parameter in parameters),
        "git_commit": commit, "git_dirty": dirty,
    })
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "unavailable"
    artifacts = RunArtifacts(
        run_dir, resolved,
        {"schema_version": 1, "run_id": run_id, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(), "heldout_evaluated_during_training": args.heldout_role == "validation"},
        "\n".join((f"python={platform.python_version()}", f"numpy={np.__version__}", f"torch={torch.__version__}", f"cuda={torch.version.cuda}", f"device={device}", f"gpu={gpu}")),
        f"commit={commit}\ndirty={dirty}\n{status}",
    )
    logger = WandbLogger(args.wandb_mode, run_dir, {**resolved, "hostname": socket.gethostname(), "gpu_model": gpu}, args.wandb_project, args.wandb_group, args.wandb_job_type, args.wandb_run_name or run_id)
    global_step, start_epoch = 0, 1
    best_rmse, best_epoch = float("inf"), None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("kind") != "direct_cnn_regression":
            raise ValueError("resume checkpoint is not a direct CNN baseline")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng_states(checkpoint["rng_states"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_rmse = float(checkpoint.get("best_rmse", float("inf")))
        best_epoch = checkpoint.get("best_epoch")

    def save(label: str, epoch: int) -> None:
        payload = {
            "schema_version": 1, "kind": "direct_cnn_regression", "epoch": epoch,
            "global_step": global_step, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "rng_states": capture_rng_states(), "config": resolved,
            "normalization": dict(train_set.normalization_metadata),
            "output_spec": {"channels": output_channels, "length": 512, "sampling_rate_hz": 128, "target_lead": args.target_lead},
            "best_rmse": best_rmse, "best_epoch": best_epoch,
            "provenance": {"git_commit": commit, "git_dirty": dirty, "command": shlex.join(sys.argv)},
        }
        path = run_dir / f"checkpoint_{label}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        artifacts.update_checkpoint_manifest(label, {"file": path.name, "epoch": epoch, "global_step": global_step})

    started = time.monotonic()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            values: dict[str, list[float]] = {"train/total_loss": [], "train/rmse": [], "train/mae": []}
            pbar = tqdm(loader, desc=f"DirectCNN epoch {epoch}/{args.epochs}")
            for batch_index, batch in enumerate(pbar):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                target, condition = (tensor.float().to(device) for tensor in batch[:2])
                optimizer.zero_grad(set_to_none=True)
                prediction = model(condition)
                loss = torch.mean(torch.square(prediction - target))
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                optimizer.step()
                error = prediction.detach() - target
                metrics = {
                    "train/total_loss": float(loss.detach().cpu()),
                    "train/rmse": float(torch.sqrt(torch.mean(torch.square(error))).cpu()),
                    "train/mae": float(torch.mean(torch.abs(error)).cpu()),
                    "train/gradient_norm": float(gradient.detach().cpu()),
                    "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("direct CNN training metrics contain NaN or Inf")
                for key in values:
                    values[key].append(metrics[key])
                if global_step % args.log_interval_steps == 0:
                    logger.log({**metrics, "train/epoch": float(epoch)}, global_step)
                global_step += 1
                pbar.set_postfix(loss=f"{metrics['train/total_loss']:.5f}")
            if not values["train/total_loss"]:
                raise ValueError("training loader produced no batches")
            scheduler.step()
            epoch_metrics = {key: float(np.mean(value)) for key, value in values.items()}
            epoch_metrics["train/learning_rate"] = float(optimizer.param_groups[0]["lr"])
            artifacts.append_metrics("epoch_metrics.csv", epoch, global_step, epoch_metrics)
            logger.log(epoch_metrics, global_step)
            if args.heldout_role == "validation" and (epoch % args.validation_interval_epochs == 0 or epoch == args.epochs):
                validation = validate(model, heldout_loader, device, args.validation_max_batches)
                artifacts.append_metrics("validation_metrics.csv", epoch, global_step, validation)
                logger.log(validation, global_step)
                if validation["val/rmse"] < best_rmse:
                    best_rmse, best_epoch = validation["val/rmse"], epoch
                    save("best_rmse", epoch)
            save("latest", epoch)
            if epoch % args.save_every == 0:
                save(f"epoch_{epoch}", epoch)
            print(f"epoch={epoch} loss={epoch_metrics['train/total_loss']:.6f} lr={epoch_metrics['train/learning_rate']:.3e}")
        summary = {
            "status": "completed", "final_epoch": args.epochs, "global_step": global_step,
            "best_validation_rmse": None if not np.isfinite(best_rmse) else best_rmse,
            "best_validation_epoch": best_epoch, "training_duration_seconds": time.monotonic() - started,
            "heldout_evaluated_during_training": args.heldout_role == "validation",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        artifacts.update_run_metadata(summary)
        logger.finish(summary)
    except BaseException as error:
        failure = {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "exception_type": type(error).__name__, "exception_message": str(error).splitlines()[0], "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        artifacts.update_run_metadata(failure)
        logger.finish(failure, exit_code=1)
        raise
    return run_dir


if __name__ == "__main__":
    train(parse_args_with_config())
