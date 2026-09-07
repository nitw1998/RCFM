"""Instrumented training orchestration for the canonical train_rcfm entry point."""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ConditionNet, DiffusionUNetCrossAttention
from rcfm import RegionAwareConditionalFlowMatching
from src.rcfm.checkpoint import (
    capture_rng_states,
    load_checkpoint,
    restore_rng_states,
    save_checkpoint,
)
from src.rcfm.experiment import RunArtifacts, WandbLogger
from src.rcfm.validation import validate_epoch


def _repository_state(root: Path) -> tuple[str, bool, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip()), status


def _means(metrics: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def _build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    if epochs <= 0 or warmup_epochs < 0:
        raise ValueError("epochs must be positive and warmup_epochs must be nonnegative")
    effective_warmup = min(warmup_epochs, max(epochs - 1, 0))
    if effective_warmup == 0:
        return CosineAnnealingLR(optimizer, T_max=epochs)
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=effective_warmup,
            ),
            CosineAnnealingLR(
                optimizer,
                T_max=max(epochs - effective_warmup, 1),
            ),
        ],
        milestones=[effective_warmup],
    )


def _build_training_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Build a shuffled loader without changing data exposure for OT runs."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


RESUME_IMMUTABLE_CONFIG_FIELDS = (
    "model_family",
    "experiment_role",
    "task",
    "datasets",
    "dataset_version",
    "split_hash",
    "normalization_id",
    "condition_unit",
    "target_unit",
    "alignment_id",
    "condition_lead",
    "target_lead",
    "condition_lead_index",
    "target_lead_index",
    "target_lead_indices",
    "window_size",
    "attention_heads",
    "flow_matcher",
    "sigma",
    "region_weight",
    "use_minibatch_ot",
    "ot_method",
    "ot_reg",
    "ot_normalize_cost",
    "ot_strict_mode",
    "ot_sampling_strategy",
    "batch_size",
    "lr",
    "weight_decay",
    "warmup_epochs",
    "grad_clip",
    "seed",
    "inference_steps",
    "validation_fixed_noise_seed",
    "mask_method",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_resume_contract(
    resolved: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    expected_kind: str,
    normalization: Mapping[str, Any],
    output_spec: Mapping[str, Any],
) -> None:
    """Reject any resume that changes the scientific or optimizer contract."""

    if int(checkpoint.get("schema_version", 0)) != 2:
        raise ValueError("resume requires a schema-2 checkpoint with RNG and global-step state")
    if checkpoint.get("kind") != expected_kind:
        raise ValueError("resume checkpoint kind disagrees with the requested model")
    source_epoch = int(checkpoint["epoch"])
    target_epochs = int(resolved["epochs"])
    if source_epoch <= 0 or target_epochs <= source_epoch:
        raise ValueError("resume target epochs must be greater than the checkpoint epoch")
    source_config = checkpoint["config"]
    mismatched = [
        field
        for field in RESUME_IMMUTABLE_CONFIG_FIELDS
        if source_config.get(field) != resolved.get(field)
    ]
    if mismatched:
        raise ValueError("resume configuration mismatch: " + ", ".join(mismatched))
    normalization_fields = ("method", "normalization_id", "condition_unit", "target_unit")
    if any(
        checkpoint["normalization"].get(key) != normalization.get(key)
        for key in normalization_fields
    ):
        raise ValueError("resume checkpoint normalization disagrees with the current dataset")
    output_fields = (
        "channels",
        "length",
        "sampling_rate_hz",
        "target_lead",
        "target_leads",
        "target_lead_indices",
    )
    if any(
        checkpoint["output_spec"].get(key) != output_spec.get(key)
        for key in output_fields
    ):
        raise ValueError("resume checkpoint output specification disagrees with the current run")


def run_training(args, dataset_builder: Callable) -> None:
    """Train from one resolved configuration and record local/W&B evidence."""

    model_family = str(getattr(args, "model_family", "RCFM")).upper()
    if model_family not in {"CFM", "RCFM"}:
        raise ValueError("model_family must be CFM or RCFM")
    experiment_role = str(getattr(args, "experiment_role", "canonical"))
    if experiment_role == "canonical":
        if args.flow_matcher != "conditional" or args.sigma != 0:
            raise ValueError(
                "canonical multistep training requires --flow_matcher conditional --sigma 0"
            )
    elif experiment_role == "path_ablation":
        if model_family != "RCFM":
            raise ValueError("path ablation is defined only for the region-aware model")
        if args.flow_matcher not in {"vp", "target", "sb"}:
            raise ValueError("path ablation requires flow_matcher vp, target, or sb")
        if float(args.sigma) != 0.1:
            raise ValueError("historical path ablation requires sigma=0.1")
        if args.flow_matcher in {"vp", "target"} and bool(args.use_minibatch_ot):
            raise ValueError("VP and target path ablations must disable OT")
        if args.flow_matcher == "sb":
            if not bool(args.use_minibatch_ot):
                raise ValueError("SB path ablation requires one external OT coupling")
            if args.ot_method != "exact" or args.ot_sampling_strategy != "multinomial":
                raise ValueError("SB path ablation requires exact OT with multinomial sampling")
    else:
        raise ValueError("experiment_role must be canonical or path_ablation")
    if model_family == "CFM" and float(args.region_weight) != 0.0:
        raise ValueError("CFM compare requires region_weight=0")
    if args.validation_interval_epochs <= 0 or args.inference_steps <= 0:
        raise ValueError("validation interval and inference steps must be positive")
    heldout_role = getattr(args, "heldout_role", "validation")
    if (
        heldout_role == "upstream_test_final_only"
        and args.validation_interval_epochs != args.epochs
    ):
        raise ValueError(
            "upstream_test_final_only requires validation_interval_epochs to equal epochs"
        )
    provenance_fields = (
        "dataset_version", "split_hash", "condition_unit", "target_unit", "alignment_id"
    )
    unresolved = [
        field
        for field in provenance_fields
        if str(getattr(args, field, "")).startswith("REQUIRED_")
    ]
    if unresolved:
        raise ValueError(f"experiment provenance placeholders must be resolved: {unresolved}")
    from train_rcfm import parse_datasets, parse_lead_indices, set_deterministic

    set_deterministic(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    datasets = parse_datasets(args.datasets)
    target_lead_indices = parse_lead_indices(getattr(args, "target_lead_indices", None))
    if target_lead_indices is None and args.target_lead_index is not None:
        target_lead_indices = [int(args.target_lead_index)]
    dataset_kwargs = {
        "normalization_id": args.normalization_id,
        "condition_lead_index": args.condition_lead_index,
        "target_lead_index": args.target_lead_index,
        "target_lead_indices": target_lead_indices,
        "max_train_records": args.max_train_records,
        "max_heldout_records": args.max_heldout_records,
    }
    region_mask_path = getattr(args, "region_mask_path", None)
    region_mask_manifest = getattr(args, "region_mask_manifest", None)
    mask_method = getattr(args, "mask_method", None) or (
        "cached_target_ecg_r_peak_roi"
        if args.region_weight > 0
        else "cached_target_ecg_r_peak_roi_diagnostics_only"
    )
    if bool(region_mask_path) != bool(region_mask_manifest):
        raise ValueError("--region_mask_path and --region_mask_manifest must be supplied together")
    if region_mask_path:
        dataset_kwargs.update(
            {
                "region_mask_path": region_mask_path,
                "region_mask_manifest": region_mask_manifest,
                "mask_method": mask_method,
                "dataset_version": args.dataset_version,
                "split_hash": args.split_hash,
            }
        )
    train_set, validation_set = dataset_builder(
        args.task,
        datasets,
        args.data_root,
        args.window_size,
        **dataset_kwargs,
    )
    normalization = dict(train_set.normalization_metadata)
    normalization.update(
        {
            "normalization_id": args.normalization_id,
            "condition_unit": args.condition_unit,
            "target_unit": args.target_unit,
        }
    )
    loader = _build_training_loader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    signal_length = args.window_size * 128
    first_target = np.asarray(train_set[0][0])
    if first_target.ndim != 2 or first_target.shape[1] != signal_length:
        raise ValueError("training targets must have shape (channels, signal_length)")
    target_channels = int(first_target.shape[0])
    if target_lead_indices is not None and len(target_lead_indices) != target_channels:
        raise ValueError("target lead indices do not match dataset output channels")
    target_leads = [item.strip() for item in str(args.target_lead).split(",") if item.strip()]
    if len(target_leads) != target_channels:
        raise ValueError("target lead names do not match dataset output channels")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    resolved = vars(args).copy()
    resolved.update(
        {
            "datasets": datasets,
            "run_id": run_id,
            "sample_rate_hz": 128,
            "train_window_seconds": args.window_size,
            "validation_window_seconds": args.window_size,
            "signal_length": signal_length,
            "output_channels": target_channels,
            "target_lead_indices": target_lead_indices,
            "target_leads": target_leads,
            "modality": args.task.split("2", maxsplit=1)[0],
            "model_family": model_family,
            "comparison_role": (
                (
                    "preprocessing_matched_cfm_ot_ablation"
                    if args.use_minibatch_ot
                    else "preprocessing_matched_cfm_control"
                )
                if model_family == "CFM"
                else (
                    f"historical_{args.flow_matcher}_path_ablation"
                    if experiment_role == "path_ablation"
                    else "region_aware_model"
                )
            ),
            "coupling_location": (
                "external_rcfm_single_coupling"
                if experiment_role == "path_ablation" and args.flow_matcher == "sb"
                else (
                    ("external_cfm" if model_family == "CFM" else "external_rcfm")
                    if args.use_minibatch_ot
                    else "none"
                )
            ),
            "heldout_role": heldout_role,
            "mask_method": mask_method,
            "mask_usage": (
                "training_loss_and_diagnostics"
                if args.region_weight > 0
                else "training_diagnostics_only"
            ),
        }
    )
    mask_provenance = getattr(train_set, "region_mask_provenance", None)
    if region_mask_path:
        if not isinstance(mask_provenance, dict):
            raise RuntimeError("external mask dataset did not expose validated provenance")
        resolved["region_mask_provenance"] = mask_provenance
    condition_net = ConditionNet().to(device)
    flow_network = DiffusionUNetCrossAttention(
        signal_length,
        target_channels,
        device=str(device),
        num_heads=int(resolved["attention_heads"]),
    ).to(device)
    rcfm = RegionAwareConditionalFlowMatching(
        flow_model=flow_network,
        flow_matcher_type=str(resolved["flow_matcher"]),
        sigma=float(resolved["sigma"]),
        region_weight=float(resolved["region_weight"]),
        use_minibatch_ot=bool(resolved["use_minibatch_ot"]),
        ot_method=str(resolved["ot_method"]),
        ot_reg=float(resolved["ot_reg"]),
        ot_normalize_cost=bool(resolved["ot_normalize_cost"]),
        ot_diagnostics=bool(resolved["ot_diagnostics"]),
        ot_strict_mode=bool(resolved["ot_strict_mode"]),
        ot_sampling_strategy=str(resolved["ot_sampling_strategy"]),
        association_debug=bool(resolved["association_debug"]),
    ).to(device)
    parameters = list(rcfm.parameters()) + list(condition_net.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    resolved["model_parameter_count"] = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    checkpoint_kind = (
        "path_ablation_rcfm"
        if experiment_role == "path_ablation"
        else (
            (
                "canonical_multistep_cfm_ot"
                if bool(args.use_minibatch_ot)
                else "canonical_multistep_cfm"
            )
            if model_family == "CFM"
            else "canonical_multistep_rcfm"
        )
    )
    output_spec = {
        "channels": target_channels,
        "length": signal_length,
        "sampling_rate_hz": 128,
        "target_lead": args.target_lead,
        "target_leads": target_leads,
        "target_lead_indices": target_lead_indices,
    }
    resume_payload: dict[str, Any] | None = None
    resume_source_best_epoch_by_rmse: int | None = None
    resume_path_value = getattr(args, "resume_checkpoint", None)
    start_epoch = 0
    if resume_path_value:
        if getattr(args, "resume_lr_policy", "restart_cosine") != "restart_cosine":
            raise ValueError("only restart_cosine is supported for resumed multistep training")
        restart_lr_value = getattr(args, "resume_restart_lr", None)
        restart_lr = float(args.lr if restart_lr_value is None else restart_lr_value)
        if restart_lr <= 0 or restart_lr > float(args.lr):
            raise ValueError("resume restart LR must be positive and no greater than the original LR")
        resume_path = Path(resume_path_value).resolve()
        resume_payload = load_checkpoint(resume_path, map_location=device)
        source_metadata_path = resume_path.parent / "run_metadata.json"
        if source_metadata_path.is_file():
            source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
            if source_metadata.get("best_epoch_by_rmse") is not None:
                resume_source_best_epoch_by_rmse = int(
                    source_metadata["best_epoch_by_rmse"]
                )
        _validate_resume_contract(
            resolved,
            resume_payload,
            expected_kind=checkpoint_kind,
            normalization=normalization,
            output_spec=output_spec,
        )
        rcfm.load_state_dict(resume_payload["model_state"], strict=True)
        condition_net.load_state_dict(resume_payload["condition_state"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = restart_lr
            parameter_group["initial_lr"] = restart_lr
        start_epoch = int(resume_payload["epoch"])
        scheduler = _build_scheduler(optimizer, args.epochs - start_epoch, warmup_epochs=0)
        resolved.update(
            {
                "resume_checkpoint": str(resume_path),
                "resume_source_checkpoint_sha256": _file_sha256(resume_path),
                "resume_source_epoch": start_epoch,
                "resume_source_global_step": int(resume_payload["global_step"]),
                "resume_source_best_epoch_by_rmse": resume_source_best_epoch_by_rmse,
                "resume_lr_policy": "restart_cosine",
                "resume_lr_restart": restart_lr,
                "resume_remaining_epochs": int(args.epochs - start_epoch),
                "resume_warmup_epochs": 0,
                "resume_optimizer_moments_restored": True,
                "resume_scheduler_state_restored": False,
                "resume_rng_states_restored": True,
            }
        )
    else:
        if getattr(args, "resume_restart_lr", None) is not None:
            raise ValueError("--resume_restart_lr requires --resume_checkpoint")
        scheduler = _build_scheduler(optimizer, args.epochs, args.warmup_epochs)

    repository_root = Path(__file__).resolve().parents[2]
    git_commit, git_dirty, git_status = _repository_state(repository_root)
    resolved.update({"git_commit": git_commit, "git_dirty": git_dirty})
    run_dir = Path(args.output_dir) / args.task / "-".join(datasets) / run_id
    gpu_model = torch.cuda.get_device_name(device) if device.type == "cuda" else "unavailable"
    environment = "\n".join(
        [
            f"python={platform.python_version()}",
            f"numpy={np.__version__}",
            f"torch={torch.__version__}",
            f"cuda={torch.version.cuda}",
            f"device={device}",
            f"gpu={gpu_model}",
        ]
    )
    initial_metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject_metadata_available": False,
        "clinical_validation_status": "unavailable_without_subject_continuity_metadata",
    }
    if resume_payload is not None:
        initial_metadata.update(
            {
                "resume_source_checkpoint_sha256": resolved[
                    "resume_source_checkpoint_sha256"
                ],
                "resume_source_epoch": start_epoch,
                "resume_source_global_step": int(resume_payload["global_step"]),
                "resume_source_best_epoch_by_rmse": resume_source_best_epoch_by_rmse,
                "resume_lr_policy": "restart_cosine",
                "resume_target_epoch": int(args.epochs),
            }
        )
    artifacts = RunArtifacts(
        run_dir,
        resolved,
        initial_metadata,
        environment,
        f"commit={git_commit}\ndirty={git_dirty}\n{git_status}",
    )
    logger = WandbLogger(
        mode=args.wandb_mode,
        run_dir=run_dir,
        config={
            **resolved,
            "hostname": socket.gethostname(),
            "gpu_model": gpu_model,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        },
        project=args.wandb_project,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        run_name=args.wandb_run_name or run_id,
    )
    print("resolved configuration:")
    for key in (
        "model_family", "task", "datasets", "flow_matcher", "sigma", "use_minibatch_ot",
        "ot_method", "ot_reg", "ot_sampling_strategy", "region_weight",
        "train_window_seconds", "sample_rate_hz", "seed",
    ):
        print(f"  {key}={resolved[key]}")

    global_step = int(resume_payload["global_step"]) if resume_payload is not None else 0
    best_metrics = (
        {key: float(value) for key, value in resume_payload["best_metrics"].items()}
        if resume_payload is not None
        else {
            "val/rmse": float("inf"),
            "val/waveform_fd": float("inf"),
            "val/velocity_mse": float("inf"),
        }
    )
    best_epochs: dict[str, int] = (
        {"val/rmse": resume_source_best_epoch_by_rmse}
        if resume_source_best_epoch_by_rmse is not None
        else {}
    )
    final_validation: dict[str, float] = {}
    start_time = time.monotonic()
    ot_step_values: dict[str, list[float]] = {}
    if resume_payload is not None:
        restore_rng_states(resume_payload["rng_states"])

    def checkpoint_payload(epoch_number: int) -> dict:
        return {
            "schema_version": 2,
            "kind": checkpoint_kind,
            "epoch": epoch_number,
            "global_step": global_step,
            "model_state": rcfm.state_dict(),
            "condition_state": condition_net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": resolved,
            "normalization": normalization,
            "output_spec": output_spec,
            "best_metrics": best_metrics,
            "rng_states": capture_rng_states(),
            "provenance": {
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "command": shlex.join(sys.argv),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        }

    def write_checkpoint(label: str, epoch_number: int) -> None:
        path = run_dir / f"checkpoint_{label}.pt"
        save_checkpoint(checkpoint_payload(epoch_number), path)
        artifacts.update_checkpoint_manifest(
            label,
            {"file": path.name, "epoch": epoch_number, "global_step": global_step},
        )

    try:
        for epoch in range(start_epoch, args.epochs):
            rcfm.train()
            condition_net.train()
            epoch_metrics: dict[str, list[float]] = {}
            epoch_ot_metrics: dict[str, list[float]] = {}
            pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
            for batch_index, (target, condition, region_mask) in enumerate(pbar):
                target = target.float().to(device)
                condition = condition.float().to(device)
                region_mask = region_mask.float().to(device)
                optimizer.zero_grad(set_to_none=True)
                debug_ids = torch.arange(target.shape[0]) if args.association_debug else None
                sample_metadata = (
                    {
                        "target_sample_id": debug_ids,
                        "condition_target_id": debug_ids,
                        "mask_target_id": debug_ids,
                    }
                    if debug_ids is not None
                    else None
                )
                output = rcfm(
                    target=target,
                    conditions=condition_net(condition),
                    region_mask=region_mask,
                    sample_metadata=sample_metadata,
                )
                loss = output["loss"]
                assert torch.is_tensor(loss)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                step_metrics = {
                    key: float(value.detach().cpu())
                    for key, value in output.items()
                    if key != "loss" and torch.is_tensor(value) and value.numel() == 1
                }
                step_metrics.update(
                    {
                        "train/gradient_norm": float(gradient_norm.detach().cpu()),
                        "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "train/epoch": float(epoch + 1),
                        "train/global_step": float(global_step),
                    }
                )
                if not all(np.isfinite(value) for value in step_metrics.values()):
                    raise FloatingPointError("training metrics contain NaN or Inf")
                optimizer.step()
                for key, value in step_metrics.items():
                    epoch_metrics.setdefault(key, []).append(value)
                    if key.startswith("ot/"):
                        epoch_ot_metrics.setdefault(key, []).append(value)
                        ot_step_values.setdefault(key, []).append(value)
                if global_step % args.log_interval_steps == 0:
                    logger.log(
                        {
                            **{
                                key: value
                                for key, value in step_metrics.items()
                                if not key.startswith("ot/")
                            },
                            "flow/path_type": str(output["flow/path_type"]),
                        },
                        step=global_step,
                    )
                if global_step % args.ot_log_interval_steps == 0:
                    logger.log(
                        {key: value for key, value in step_metrics.items() if key.startswith("ot/")},
                        step=global_step,
                    )
                if rcfm._last_ot_histograms and global_step % args.ot_histogram_interval_steps == 0:
                    histograms = {
                        key: logger.histogram(value)
                        for key, value in rcfm._last_ot_histograms.items()
                    }
                    logger.log(
                        {key: value for key, value in histograms.items() if value is not None},
                        step=global_step,
                    )
                global_step += 1
                pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
                if args.max_batches is not None and batch_index + 1 >= args.max_batches:
                    break

            scheduler.step()
            aggregated = _means(epoch_metrics)
            aggregated_ot = _means(epoch_ot_metrics)
            if not aggregated:
                raise ValueError("training loader produced no batches")
            artifacts.append_metrics("epoch_metrics.csv", epoch + 1, global_step, aggregated)
            artifacts.append_metrics("ot_diagnostics.csv", epoch + 1, global_step, aggregated_ot)
            logger.log(aggregated, step=global_step)
            print(
                f"epoch={epoch + 1} total_loss={aggregated['train/total_loss']:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )
            if (epoch + 1) % args.validation_interval_epochs == 0 or epoch + 1 == args.epochs:
                final_validation = validate_epoch(
                    rcfm,
                    condition_net,
                    validation_loader,
                    device,
                    args.inference_steps,
                    args.validation_fixed_noise_seed,
                    args.validation_max_batches,
                    target_leads,
                )
                artifacts.append_metrics(
                    "validation_metrics.csv", epoch + 1, global_step, final_validation
                )
                logger.log(final_validation, step=global_step)
                selections = {
                    "best_rmse": "val/rmse",
                    "best_waveform_fd": "val/waveform_fd",
                    "best_velocity_mse": "val/velocity_mse",
                }
                improved = {
                    label: metric
                    for label, metric in selections.items()
                    if final_validation[metric] < best_metrics[metric]
                }
                for metric in improved.values():
                    best_metrics[metric] = final_validation[metric]
                    best_epochs[metric] = epoch + 1
                if args.checkpoint_policy == "full":
                    for label, metric in improved.items():
                        write_checkpoint(label, epoch + 1)
            write_checkpoint("latest", epoch + 1)
            if args.checkpoint_policy == "full" and (epoch + 1) % args.save_every == 0:
                write_checkpoint(f"epoch_{epoch + 1}", epoch + 1)

        def ot_mean(key: str) -> float:
            values = ot_step_values.get(key, [])
            return float(np.mean(values)) if values else 0.0

        def ot_sum(key: str) -> float:
            return float(np.sum(ot_step_values.get(key, [])))

        summary = {
            "best_epoch_by_rmse": best_epochs.get("val/rmse"),
            "best_validation_rmse": best_metrics["val/rmse"],
            "best_validation_fd": best_metrics["val/waveform_fd"],
            "final_validation_rmse": final_validation.get("val/rmse"),
            "final_validation_fd": final_validation.get("val/waveform_fd"),
            "total_fallback_count": ot_sum("ot/fallback_count"),
            "total_nonfinite_plan_count": ot_sum("ot/nonfinite_plan_count"),
            "mean_ot_cost_reduction_ratio": ot_mean("ot/cost_reduction_ratio"),
            "mean_unique_target_fraction": ot_mean("ot/unique_target_fraction"),
            "training_duration_seconds": time.monotonic() - start_time,
            "resume_source_epoch": start_epoch if resume_payload is not None else None,
            "resume_source_global_step": (
                int(resume_payload["global_step"]) if resume_payload is not None else None
            ),
            "resume_lr_policy": "restart_cosine" if resume_payload is not None else None,
            "status": "completed",
        }
        artifacts.update_run_metadata(
            {**summary, "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        )
        logger.finish(summary)
    except BaseException as error:
        message = str(error).splitlines()[0]
        for private_path in (str(args.data_root), str(args.output_dir)):
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
