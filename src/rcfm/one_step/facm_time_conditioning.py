"""Time sampling and task conditioning for the FACM expanded-time strategy.

This is a conditional 1-D adaptation of the expanded-time implementation in
``ali-vilab/FACM`` at commit 8d80d4c65101f814095984a91329ce4aa37be79b
(Apache-2.0). FACM uses ``t`` for the consistency/shortcut task and ``2-t``
for the flow-matching anchor when the data endpoint is at ``t=1``.
"""

from __future__ import annotations

import math

import torch


def sample_facm_time(
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    lognormal_mean: float = 0.8,
    lognormal_std: float = 1.6,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Match the official FACM ``default`` time distribution."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if lognormal_std <= 0:
        raise ValueError("lognormal_std must be positive")
    normal = torch.randn(
        batch_size, device=device, dtype=dtype, generator=generator
    )
    sigma = (normal * lognormal_std - lognormal_mean).exp()
    time = 1.0 - torch.atan(sigma) * (2.0 / math.pi)
    return time.clamp(0.0, 1.0)


def expanded_fm_time(time: torch.Tensor) -> torch.Tensor:
    """Return FACM's expanded-interval condition for the FM-anchor task."""

    if time.ndim != 1:
        raise ValueError("FACM time must have shape (batch,)")
    return 2.0 - time


def broadcast_time(time: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast ``(B,)`` time over all non-batch dimensions of ``reference``."""

    if time.ndim != 1 or time.shape[0] != reference.shape[0]:
        raise ValueError("time batch dimension must match the reference tensor")
    return time.reshape(time.shape[0], *([1] * (reference.ndim - 1)))
