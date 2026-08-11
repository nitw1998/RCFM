"""Deterministic held-out validation for canonical multistep RCFM."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from src.rcfm.metrics.waveform import waveform_frechet_distance


@torch.no_grad()
def validate_epoch(
    rcfm: torch.nn.Module,
    condition_net: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    inference_steps: int,
    fixed_noise_seed: int,
    max_batches: int | None = None,
    target_leads: list[str] | None = None,
) -> dict[str, float]:
    """Validate with fixed noise while restoring every global torch RNG state."""

    if inference_steps <= 0:
        raise ValueError("inference_steps must be positive")
    was_training = rcfm.training
    condition_was_training = condition_net.training
    rcfm.eval()
    condition_net.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    total_losses: list[float] = []
    velocity_losses: list[float] = []
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        for batch_index, batch in enumerate(loader):
            target, condition = batch[:2]
            target = target.float().to(device)
            condition = condition.float().to(device)
            encoded = condition_net(condition)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(fixed_noise_seed + batch_index)
            noise = torch.randn(target.shape, generator=generator, dtype=target.dtype).to(device)
            torch.manual_seed(fixed_noise_seed + 100_000 + batch_index)
            output = rcfm(target=target, conditions=encoded, source=noise, region_mask=None)
            generated = rcfm.sample(
                conditions=encoded,
                shape=tuple(target.shape),
                steps=inference_steps,
                device=device,
                initial_noise=noise,
            )
            total_losses.append(float(output["loss"].detach().cpu()))
            velocity_losses.append(float(output["train/velocity_mse"].detach().cpu()))
            predictions.append(generated.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    if was_training:
        rcfm.train()
    if condition_was_training:
        condition_net.train()
    if not predictions:
        raise ValueError("validation loader produced no samples")
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    error = prediction - target
    channels = target.shape[1]
    if target_leads is None:
        target_leads = [f"channel_{index}" for index in range(channels)]
    if len(target_leads) != channels:
        raise ValueError("target lead names must match validation channels")
    lead_metrics = {}
    lead_fds = []
    for index, lead in enumerate(target_leads):
        lead_error = error[:, index]
        lead_fd = (
            waveform_frechet_distance(target[:, index], prediction[:, index])
            if len(target) >= 2
            else 0.0
        )
        lead_fds.append(lead_fd)
        lead_metrics.update(
            {
                f"val/lead/{lead}/rmse": float(np.sqrt(np.mean(lead_error**2))),
                f"val/lead/{lead}/mae": float(np.mean(np.abs(lead_error))),
                f"val/lead/{lead}/waveform_fd": lead_fd,
            }
        )
    waveform_fd = float(np.mean(lead_fds))
    metrics = {
        "val/total_loss": float(np.mean(total_losses)),
        "val/velocity_mse": float(np.mean(velocity_losses)),
        "val/rmse": float(np.sqrt(np.mean(error ** 2))),
        "val/mae": float(np.mean(np.abs(error))),
        "val/waveform_fd": waveform_fd,
        "val/num_samples": float(len(target)),
        "val/num_subjects": 0.0,
        "val/subject_metadata_available": 0.0,
        **lead_metrics,
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("validation metrics must be finite")
    return metrics
