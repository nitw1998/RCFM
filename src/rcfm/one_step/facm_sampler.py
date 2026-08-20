"""Teacher-free FACM one-step sampler for conditional physiological signals."""

from __future__ import annotations

from collections.abc import Callable

import torch


@torch.no_grad()
def facm_one_step_sample(
    *,
    student_flow: Callable,
    condition_encoder: Callable,
    source_condition: torch.Tensor,
    initial_noise: torch.Tensor,
) -> torch.Tensor:
    """Generate with exactly one student vector-field call (NFE=1)."""

    if initial_noise.ndim != 3 or source_condition.ndim != 3:
        raise ValueError("source condition and initial noise must have shape (B,C,T)")
    if source_condition.shape[0] != initial_noise.shape[0]:
        raise ValueError("source condition and initial noise batch sizes must match")
    conditions = condition_encoder(source_condition)
    time = torch.zeros(
        initial_noise.shape[0], device=initial_noise.device, dtype=initial_noise.dtype
    )
    average_velocity = student_flow(initial_noise, conditions, time)
    if average_velocity.shape != initial_noise.shape:
        raise ValueError("FACM student output shape must match initial noise")
    return initial_noise + average_velocity
