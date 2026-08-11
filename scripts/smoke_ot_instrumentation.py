"""Run a synthetic OT instrumentation smoke without patient data or long training."""

from __future__ import annotations

import argparse
import platform
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcfm import RegionAwareConditionalFlowMatching
from src.rcfm.checkpoint import capture_rng_states, save_checkpoint
from src.rcfm.experiment import RunArtifacts, WandbLogger
from src.rcfm.validation import validate_epoch


class TinyCondition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, value: torch.Tensor):
        feature = self.projection(value)
        return {"down_conditions": [feature], "up_conditions": [feature]}


class TinyFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, value, conditions, time):
        del conditions
        return self.projection(value) + time.view(-1, 1, 1) * 0.0


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(31)
    np.random.seed(31)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but torch.cuda.is_available() is false")
    run_dir = Path(args.output_dir) / "synthetic_offline"
    repository_root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=repository_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    )
    config = {
        "task": "ppg2ecg",
        "datasets": ["synthetic"],
        "dataset_version": "synthetic-v1",
        "seed": 31,
        "split_hash": "synthetic-fixed-split-v1",
        "flow_matcher": "conditional",
        "sigma": 0.0,
        "region_weight": 0.01,
        "use_minibatch_ot": True,
        "ot_method": "exact",
        "ot_reg": 0.05,
        "ot_normalize_cost": False,
        "ot_sampling_strategy": "assignment",
        "association_debug": True,
        "normalization_id": "training_global_zscore_v1",
        "condition_unit": "synthetic_normalized_unit",
        "target_unit": "synthetic_normalized_unit",
        "alignment_id": "synthetic-index-aligned-v1",
        "condition_lead": "synthetic_condition",
        "target_lead": "synthetic_target",
        "condition_lead_index": None,
        "target_lead_index": None,
        "window_size": 0.0625,
        "attention_heads": 1,
        "inference_steps": 2,
        "wandb_mode": args.wandb_mode,
        "device": str(device),
        "data_root": "/synthetic/not-used",
        "output_dir": str(args.output_dir),
    }
    artifacts = RunArtifacts(
        run_dir,
        config,
        {"schema_version": 1, "status": "running", "synthetic_only": True},
        f"python={platform.python_version()}\ntorch={torch.__version__}",
        f"commit={commit}\ndirty={dirty}",
    )
    logger = WandbLogger(
        args.wandb_mode,
        run_dir,
        config,
        project="RCFM-instrumentation-smoke",
        group="synthetic",
        job_type="test",
        run_name="synthetic-offline",
    )
    condition_net = TinyCondition().to(device)
    model = RegionAwareConditionalFlowMatching(
        TinyFlow(),
        region_weight=0.01,
        use_minibatch_ot=True,
        ot_method="exact",
        ot_diagnostics=True,
        ot_strict_mode=True,
        ot_sampling_strategy="assignment",
        association_debug=True,
    ).to(device)
    parameters = list(model.parameters()) + list(condition_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    target_cpu = torch.linspace(-1.0, 1.0, 32).reshape(4, 1, 8)
    condition_cpu = torch.flip(target_cpu, dims=(-1,))
    target = target_cpu.to(device)
    condition = condition_cpu.to(device)
    mask = torch.zeros_like(target)
    mask[:, :, 2:5] = 1.0
    metadata = {
        "target_sample_id": torch.arange(4),
        "condition_target_id": torch.arange(4),
        "mask_target_id": torch.arange(4),
    }
    optimizer.zero_grad(set_to_none=True)
    output = model(
        target=target,
        conditions=condition_net(condition),
        source=torch.flip(target, dims=(0,)),
        region_mask=mask,
        sample_metadata=metadata,
    )
    loss = output["loss"]
    assert torch.is_tensor(loss)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    scheduler.step()
    train_metrics = {
        key: float(value.detach())
        for key, value in output.items()
        if key != "loss" and torch.is_tensor(value) and value.numel() == 1
    }
    train_metrics.update(
        {
            "train/gradient_norm": float(gradient_norm),
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "train/epoch": 1.0,
            "train/global_step": 1.0,
        }
    )
    validation = validate_epoch(
        model,
        condition_net,
        DataLoader(TensorDataset(target_cpu, condition_cpu), batch_size=4, shuffle=False),
        device,
        inference_steps=2,
        fixed_noise_seed=2025,
    )
    artifacts.append_metrics("epoch_metrics.csv", 1, 1, train_metrics)
    artifacts.append_metrics(
        "ot_diagnostics.csv",
        1,
        1,
        {key: value for key, value in train_metrics.items() if key.startswith("ot/")},
    )
    artifacts.append_metrics("validation_metrics.csv", 1, 1, validation)
    checkpoint_path = run_dir / "checkpoint_latest.pt"
    save_checkpoint(
        {
            "schema_version": 2,
            "kind": "canonical_multistep_rcfm",
            "epoch": 1,
            "global_step": 1,
            "model_state": model.state_dict(),
            "condition_state": condition_net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
            "normalization": {
                "method": "training_global_zscore",
                "target_mean": 0.0,
                "target_scale": 1.0,
                "condition_mean": 0.0,
                "condition_scale": 1.0,
                "normalization_id": "training_global_zscore_v1",
                "condition_unit": "synthetic_normalized_unit",
                "target_unit": "synthetic_normalized_unit",
            },
            "output_spec": {
                "channels": 1,
                "length": 8,
                "sampling_rate_hz": 128,
                "target_lead": "synthetic_target",
            },
            "best_metrics": validation,
            "rng_states": capture_rng_states(),
            "provenance": {
                "git_commit": commit,
                "git_dirty": dirty,
                "command": shlex.join(sys.argv),
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
        },
        checkpoint_path,
    )
    artifacts.update_checkpoint_manifest(
        "latest", {"file": checkpoint_path.name, "epoch": 1, "global_step": 1}
    )
    summary = {
        "status": "completed",
        "synthetic_only": True,
        "best_validation_rmse": validation["val/rmse"],
        "best_validation_fd": validation["val/waveform_fd"],
        "total_fallback_count": train_metrics["ot/fallback_count"],
        "total_nonfinite_plan_count": train_metrics["ot/nonfinite_plan_count"],
    }
    artifacts.update_run_metadata(summary)
    logger.log({**train_metrics, **validation, "flow/path_type": output["flow/path_type"]}, step=1)
    logger.finish(summary)
    print(run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--wandb_mode", choices=["disabled", "offline"], default="disabled"
    )
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args())
