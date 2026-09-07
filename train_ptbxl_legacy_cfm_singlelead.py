"""Train the unchanged legacy CFM architecture for PTB-XL III-to-V5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import ConditionNet, DiffusionUNetCrossAttention
from train_cfm_basic import MinimalFlowMatching


LEGACY_COMMIT = "c366eee7781eb4c7f6079d3c34b60c73b46d4e5a"
MODEL_SOURCE_SHA256 = "fa9e3f101d782e6dc87550c65481f5531db4a00f9f34ebbfe6faa5440010433e"
CFM_SOURCE_SHA256 = "feb86d57d921b991fd60daf064a8b4dd2cd31fa0b8d99db1714d3e7977028067"
FLOW_PARAMETERS = 45_828_129
CONDITION_PARAMETERS = 26_926_016
WINDOW_SAMPLES = 512
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CONDITION_INDEX = 2
TARGET_INDEX = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _minmax_per_record(values: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(values, dtype=np.float32))
    if array.ndim != 2 or array.shape[1] != WINDOW_SAMPLES:
        raise ValueError(f"expected (records,{WINDOW_SAMPLES}) values, got {array.shape}")
    low = array.min(axis=1, keepdims=True)
    span = array.max(axis=1, keepdims=True) - low
    if np.any(span <= 0) or not np.all(np.isfinite(span)):
        raise ValueError("legacy single-lead records must have finite nonzero range")
    return np.asarray(2.0 * (array - low) / span - 1.0, dtype=np.float32)


def _prepare_split(path: Path, maximum: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    if raw.ndim != 3 or raw.shape[1] < WINDOW_SAMPLES or raw.shape[2] != len(LEADS):
        raise ValueError(f"invalid PTB-XL array shape: {raw.shape}")
    count = len(raw) if maximum is None else min(len(raw), maximum)
    condition = _minmax_per_record(np.asarray(raw[:count, :WINDOW_SAMPLES, CONDITION_INDEX]))
    target = _minmax_per_record(np.asarray(raw[:count, :WINDOW_SAMPLES, TARGET_INDEX]))
    return target[:, None, :], condition[:, None, :]


def _validate_sources(repository: Path) -> None:
    checks = {
        repository / "model.py": MODEL_SOURCE_SHA256,
        repository / "train_cfm_basic.py": CFM_SOURCE_SHA256,
    }
    mismatched = [path.name for path, expected in checks.items() if _sha256(path) != expected]
    if mismatched:
        raise RuntimeError(
            "legacy network source differs from commit c366eee: " + ", ".join(mismatched)
        )


def _set_deterministic(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset_version",
        "split_hash",
        "condition_lead",
        "target_lead",
        "condition_lead_index",
        "target_lead_index",
        "normalization_id",
        "architecture_id",
        "architecture_source_commit",
        "epochs",
        "batch_size",
        "num_workers",
        "learning_rate",
        "save_every",
        "grad_clip",
        "seed",
        "device",
        "wandb_mode",
        "wandb_project",
        "wandb_group",
    }
    if not isinstance(config, dict):
        raise ValueError("legacy CFM config must be a JSON object")
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"legacy CFM config missing fields: {missing}")
    expected = {
        "dataset_version": "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1",
        "split_hash": "7784c98a2c8daccc23fc7cb0d47dc1933eeee77f120c05f8a24ac149cd8474f7",
        "condition_lead": "III",
        "target_lead": "V5",
        "condition_lead_index": CONDITION_INDEX,
        "target_lead_index": TARGET_INDEX,
        "normalization_id": "per_record_per_selected_lead_minmax_neg1_1",
        "architecture_id": "legacy_DiffusionUNetCrossAttention_ConditionNet_MinimalFlowMatching_v1",
        "architecture_source_commit": LEGACY_COMMIT,
    }
    mismatched = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatched:
        raise ValueError(f"legacy single-lead contract mismatch: {mismatched}")
    for name in ("epochs", "batch_size", "num_workers", "save_every"):
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if float(config["learning_rate"]) <= 0 or float(config["grad_clip"]) <= 0:
        raise ValueError("learning rate and gradient clip must be positive")
    if config["wandb_mode"] not in {"disabled", "offline", "online"}:
        raise ValueError("invalid wandb_mode")
    return config


def build_models(device: torch.device) -> tuple[MinimalFlowMatching, ConditionNet]:
    flow = MinimalFlowMatching(
        DiffusionUNetCrossAttention(WINDOW_SAMPLES, 1, str(device), num_heads=4)
    ).to(device)
    condition = ConditionNet().to(device)
    if sum(parameter.numel() for parameter in flow.parameters()) != FLOW_PARAMETERS:
        raise RuntimeError("legacy flow parameter count changed")
    if sum(parameter.numel() for parameter in condition.parameters()) != CONDITION_PARAMETERS:
        raise RuntimeError("legacy condition parameter count changed")
    return flow, condition


def train(args: argparse.Namespace) -> Path:
    repository = Path(__file__).resolve().parent
    _validate_sources(repository)
    config = _load_config(args.config.resolve())
    for key in ("epochs", "batch_size", "num_workers", "learning_rate", "save_every", "grad_clip", "seed", "device", "wandb_mode"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _set_deterministic(int(config["seed"]), device)

    dataset_dir = args.data_root.resolve() / "PTBXL"
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != config["dataset_version"] or manifest.get("split_hash") != config["split_hash"]:
        raise ValueError("PTB-XL dataset manifest does not match the frozen config")
    train_path = dataset_dir / "X_train_resampled.npy"
    val_path = dataset_dir / "X_val_resampled.npy"
    if _sha256(train_path) != manifest["output_sha256"][train_path.name]:
        raise ValueError("PTB-XL training array hash mismatch")
    if _sha256(val_path) != manifest["output_sha256"][val_path.name]:
        raise ValueError("PTB-XL validation array hash mismatch")
    train_target, train_condition = _prepare_split(train_path, args.max_train_records)
    validation_target, validation_condition = _prepare_split(val_path, args.max_validation_records)
    expected_train = len(train_target) if args.max_train_records else 17_440
    expected_validation = len(validation_target) if args.max_validation_records else 2_193
    if len(train_target) != expected_train or len(validation_target) != expected_validation:
        raise ValueError("PTB-XL official split counts do not match the protocol")

    loader_generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_target), torch.from_numpy(train_condition)),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=loader_generator,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    flow, condition = build_models(device)
    optimizer = torch.optim.Adam(
        list(flow.parameters()) + list(condition.parameters()),
        lr=float(config["learning_rate"]),
    )

    run_id = args.run_id
    output = args.output_dir.resolve() / run_id
    if output.exists() and args.resume is None:
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True, exist_ok=args.resume is not None)
    start_epoch, global_step = 0, 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume.resolve(), map_location="cpu")
        if checkpoint.get("architecture_source_commit") != LEGACY_COMMIT:
            raise ValueError("resume checkpoint is not the frozen legacy architecture")
        if checkpoint.get("config", {}).get("condition_lead") != "III" or checkpoint.get("config", {}).get("target_lead") != "V5":
            raise ValueError("resume checkpoint is not the III-to-V5 task")
        flow.load_state_dict(checkpoint["flow_model"])
        condition.load_state_dict(checkpoint["condition_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        _restore_rng(checkpoint["rng_state"])
        loader_generator.set_state(checkpoint["loader_generator_state"])

    resolved = {
        **config,
        "run_id": run_id,
        "train_records": int(len(train_target)),
        "validation_records_reserved_not_used_for_training": int(len(validation_target)),
        "test_split_loaded_during_training": False,
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "minimal_cfm_source_sha256": CFM_SOURCE_SHA256,
        "flow_parameters": FLOW_PARAMETERS,
        "condition_parameters": CONDITION_PARAMETERS,
        "checkpoint_policy": "atomic_latest_plus_final_legacy_compatible_weights",
    }
    _atomic_json(output / "resolved_config.json", resolved)
    _atomic_json(
        output / "run_manifest.json",
        {
            "schema_version": 1,
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "architecture_unchanged_from_commit": LEGACY_COMMIT,
            "source_hashes_verified": True,
            "dataset_manifest_sha256": _sha256(manifest_path),
            "train_array_sha256": _sha256(train_path),
            "validation_array_sha256": _sha256(val_path),
            "test_split_loaded_during_training": False,
        },
    )

    wandb_run = None
    if config["wandb_mode"] != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=str(config["wandb_project"]),
            group=str(config["wandb_group"]),
            name=run_id,
            job_type="legacy-singlelead-train",
            mode=str(config["wandb_mode"]),
            config=resolved,
        )

    metrics_path = output / "training_metrics.csv"
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as metrics_handle:
        writer = csv.DictWriter(metrics_handle, fieldnames=["epoch", "global_step", "train_flow_mse"])
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, int(config["epochs"])):
            flow.train()
            condition.train()
            losses: list[float] = []
            progress = tqdm(loader, desc=f"legacy CFM epoch {epoch + 1}/{config['epochs']}")
            for batch_index, (target, source) in enumerate(progress):
                target = target.to(device, non_blocking=True)
                source = source.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                conditions = condition(source)
                noise = torch.randn_like(target)
                loss = flow(x=target, y=noise, cond=conditions, mode="train")
                if not torch.isfinite(loss):
                    raise FloatingPointError("legacy CFM loss is NaN or Inf")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(flow.parameters()) + list(condition.parameters()),
                    max_norm=float(config["grad_clip"]),
                )
                optimizer.step()
                global_step += 1
                losses.append(float(loss.detach().cpu()))
                progress.set_postfix(loss=f"{losses[-1]:.5f}")
                if args.max_batches is not None and batch_index + 1 >= args.max_batches:
                    break
            epoch_loss = float(np.mean(losses))
            writer.writerow({"epoch": epoch + 1, "global_step": global_step, "train_flow_mse": epoch_loss})
            metrics_handle.flush()
            if wandb_run is not None:
                wandb_run.log({"train/flow_mse": epoch_loss, "epoch": epoch + 1}, step=global_step)
            if (epoch + 1) % int(config["save_every"]) == 0 or epoch + 1 == int(config["epochs"]):
                checkpoint = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "flow_model": flow.state_dict(),
                    "condition_net": condition.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state": _rng_state(),
                    "loader_generator_state": loader_generator.get_state(),
                    "config": resolved,
                    "architecture_source_commit": LEGACY_COMMIT,
                }
                _atomic_torch_save(output / "checkpoint_latest.pt", checkpoint)
                _atomic_torch_save(output / "minimal_cfm_latest.pth", flow.state_dict())
                _atomic_torch_save(output / "condition_net_latest.pth", condition.state_dict())
                if epoch + 1 == int(config["epochs"]):
                    legacy_index = epoch
                    _atomic_torch_save(output / f"minimal_cfm_epoch_{legacy_index}.pth", flow.state_dict())
                    _atomic_torch_save(output / f"condition_net_epoch_{legacy_index}.pth", condition.state_dict())

    if wandb_run is not None:
        wandb_run.finish()
    final_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    final_manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "final_epoch": int(config["epochs"]),
            "global_step": global_step,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": str(device),
            },
        }
    )
    _atomic_json(output / "run_manifest.json", final_manifest)
    print(f"completed legacy PTB-XL III-to-V5 CFM: {output}", flush=True)
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--grad_clip", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--wandb_mode", choices=("disabled", "offline", "online"))
    parser.add_argument("--max_train_records", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max_validation_records", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max_batches", type=int, help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
