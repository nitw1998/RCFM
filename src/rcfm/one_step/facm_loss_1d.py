"""Source-faithful FACM objective adapted to conditional 1-D signals.

The objective follows ``losses.py`` from the official FACM repository at
commit 8d80d4c65101f814095984a91329ce4aa37be79b (Apache-2.0). Modifications are
scoped to dimension-agnostic reductions, conditional feature pyramids instead
of ImageNet class labels/CFG, and the existing RCFM ``(B,C,T)`` interface.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .facm_time_conditioning import broadcast_time, expanded_fm_time, sample_facm_time


TensorMap = Mapping[str, list[torch.Tensor]]
FlowCall = Callable[[torch.Tensor, TensorMap, torch.Tensor], torch.Tensor]


def _per_example_mean(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise ValueError("FACM signal tensors require batch and signal dimensions")
    return value.flatten(1).mean(dim=1)


class FACMLoss1D(nn.Module):
    """Mixed FACM shortcut and flow-anchor loss for ``(B,C,T)`` ECG targets."""

    def __init__(
        self,
        *,
        relaxation_power: float = 0.5,
        robust_c: float = 1e-3,
        robust_power: float = 0.5,
        cm_weight: float = 1.0,
        fm_weight: float = 1.0,
        time_mean: float = 0.8,
        time_std: float = 1.6,
    ) -> None:
        super().__init__()
        if relaxation_power <= 0:
            raise ValueError("relaxation_power must be positive")
        if robust_c <= 0 or robust_power <= 0:
            raise ValueError("robust loss constants must be positive")
        if cm_weight < 0 or fm_weight < 0 or cm_weight + fm_weight <= 0:
            raise ValueError("FACM objective weights must be nonnegative and nonzero")
        self.relaxation_power = float(relaxation_power)
        self.robust_c = float(robust_c)
        self.robust_power = float(robust_power)
        self.cm_weight = float(cm_weight)
        self.fm_weight = float(fm_weight)
        self.time_mean = float(time_mean)
        self.time_std = float(time_std)

    def flow_matching_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """FACM anchor: per-example MSE plus full-signal cosine distance."""

        if prediction.shape != target.shape:
            raise ValueError("FM prediction and target shapes must match")
        mse = _per_example_mean((prediction - target).square())
        cosine = 1.0 - F.cosine_similarity(
            prediction.flatten(1), target.flatten(1), dim=1, eps=1e-8
        )
        return mse + cosine

    def robust_norm_l2(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("CM prediction and target shapes must match")
        error = _per_example_mean((prediction - target).square())
        denominator = (error + self.robust_c).pow(self.robust_power).detach()
        return error / denominator

    def forward(
        self,
        *,
        student_flow: FlowCall,
        teacher_flow: FlowCall,
        target: torch.Tensor,
        student_conditions: TensorMap,
        teacher_conditions: TensorMap,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the official distillation-style FACM mixed objective."""

        if target.ndim != 3:
            raise ValueError("RCFM FACM targets must have shape (B,C,T)")
        batch_size = target.shape[0]
        if noise is None:
            noise = torch.randn(
                target.shape,
                device=target.device,
                dtype=target.dtype,
                generator=generator,
            )
        if noise.shape != target.shape:
            raise ValueError("noise and target shapes must match")
        if time is None:
            time = sample_facm_time(
                batch_size,
                device=target.device,
                dtype=target.dtype,
                lognormal_mean=self.time_mean,
                lognormal_std=self.time_std,
                generator=generator,
            )
        if time.shape != (batch_size,) or time.device != target.device:
            raise ValueError("time must have shape (B,) on the target device")
        time = time.to(dtype=target.dtype)
        time_view = broadcast_time(time, target)
        x_t = time_view * target + (1.0 - time_view) * noise

        with torch.no_grad():
            teacher_velocity = teacher_flow(x_t, teacher_conditions, time)
        if teacher_velocity.shape != target.shape:
            raise ValueError("teacher velocity shape must match the ECG target")
        teacher_velocity = teacher_velocity.detach()

        fm_prediction = student_flow(x_t, student_conditions, expanded_fm_time(time))
        fm_per_example = self.flow_matching_loss(fm_prediction, teacher_velocity)

        def shortcut_call(x_input: torch.Tensor, t_input: torch.Tensor) -> torch.Tensor:
            return student_flow(x_input, student_conditions, t_input)

        shortcut_prediction, total_derivative = torch.func.jvp(
            shortcut_call,
            (x_t, time),
            (teacher_velocity, torch.ones_like(time)),
        )
        if shortcut_prediction.shape != target.shape or total_derivative.shape != target.shape:
            raise ValueError("student shortcut/JVP output shape must match the ECG target")

        shortcut_sg = shortcut_prediction.detach()
        derivative_sg = total_derivative.detach()
        consistency_operator = teacher_velocity + (1.0 - time_view) * derivative_sg
        residual = shortcut_sg - consistency_operator
        alpha = 1.0 - time_view.pow(self.relaxation_power)
        shortcut_target = shortcut_sg - alpha * residual.clamp(min=-1.0, max=1.0)
        beta = torch.cos(time * (math.pi / 2.0))
        cm_per_example = self.robust_norm_l2(shortcut_prediction, shortcut_target) * beta

        cm_loss = cm_per_example.mean()
        fm_loss = fm_per_example.mean()
        total_loss = self.cm_weight * cm_loss + self.fm_weight * fm_loss
        return {
            "loss": total_loss,
            "train/total_loss": total_loss.detach(),
            "train/facm_cm_loss": cm_loss.detach(),
            "train/facm_fm_anchor_loss": fm_loss.detach(),
            "facm/time_mean": time.mean().detach(),
            "facm/time_std": time.std(unbiased=False).detach(),
            "facm/teacher_velocity_norm": teacher_velocity.flatten(1).norm(dim=1).mean().detach(),
            "facm/shortcut_jvp_norm": derivative_sg.flatten(1).norm(dim=1).mean().detach(),
            "facm/fm_prediction_norm": fm_prediction.flatten(1).norm(dim=1).mean().detach(),
            "facm/cm_prediction_norm": shortcut_prediction.flatten(1).norm(dim=1).mean().detach(),
        }
