"""Region-aware conditional flow matching for 1-D physiological signals."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from conditional_flow_matcher import (
    ConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from optimal_transport import OTPlanSampler


ConditionPyramid = Dict[str, list[torch.Tensor]]


def _index_conditions(conditions: ConditionPyramid, index: torch.Tensor) -> ConditionPyramid:
    return {
        "down_conditions": [feature[index] for feature in conditions["down_conditions"]],
        "up_conditions": [feature[index] for feature in conditions["up_conditions"]],
    }


class RegionAwareConditionalFlowMatching(nn.Module):
    """RCFM module used by the paper training and inference scripts.

    The model learns a vector field from Gaussian noise to the target ECG while
    conditioning on a source modality (reduced-lead ECG, PPG, or RCG). When a
    region mask is provided, the element-wise squared velocity error is weighted
    by ``1 + region_weight * mask``.
    """

    def __init__(
        self,
        flow_model: nn.Module,
        flow_matcher_type: str = "vp",
        sigma: float = 0.1,
        region_weight: float = 1.0,
        use_minibatch_ot: bool = True,
        ot_method: str = "sinkhorn",
        ot_reg: float = 0.05,
    ) -> None:
        super().__init__()
        self.flow_model = flow_model
        self.region_weight = region_weight
        self.use_minibatch_ot = use_minibatch_ot
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=ot_reg) if use_minibatch_ot else None

        if flow_matcher_type == "conditional":
            self.flow_matcher = ConditionalFlowMatcher(sigma=sigma)
        elif flow_matcher_type == "target":
            self.flow_matcher = TargetConditionalFlowMatcher(sigma=sigma)
        elif flow_matcher_type == "sb":
            self.flow_matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method=ot_method)
        elif flow_matcher_type == "vp":
            self.flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=sigma)
        else:
            raise ValueError(
                f"Unknown flow_matcher_type={flow_matcher_type!r}. "
                "Use one of: conditional, target, sb, vp."
            )

    def _apply_minibatch_ot(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        conditions: ConditionPyramid,
        region_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, ConditionPyramid, Optional[torch.Tensor]]:
        if not self.use_minibatch_ot or self.ot_sampler is None:
            return source, target, conditions, region_mask

        pi = self.ot_sampler.get_map(source, target)
        source_idx, target_idx = self.ot_sampler.sample_map(pi, source.shape[0])
        source_idx = torch.as_tensor(source_idx, device=source.device, dtype=torch.long)
        target_idx = torch.as_tensor(target_idx, device=target.device, dtype=torch.long)

        source = source[source_idx]
        target = target[target_idx]
        conditions = _index_conditions(conditions, target_idx)
        if region_mask is not None:
            region_mask = region_mask[target_idx]
        return source, target, conditions, region_mask

    @staticmethod
    def region_weighted_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        region_mask: Optional[torch.Tensor],
        region_weight: float,
    ) -> torch.Tensor:
        squared_error = (prediction - target).pow(2)
        if region_mask is None or region_weight <= 0:
            return squared_error.mean()

        mask = region_mask.to(device=prediction.device, dtype=prediction.dtype).clamp(0.0, 1.0)
        while mask.dim() < prediction.dim():
            mask = mask.unsqueeze(1)
        weights = 1.0 + region_weight * mask
        return (weights * squared_error).mean()

    def forward(
        self,
        target: torch.Tensor,
        conditions: ConditionPyramid,
        source: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if source is None:
            source = torch.randn_like(target)

        source, target, conditions, region_mask = self._apply_minibatch_ot(
            source=source,
            target=target,
            conditions=conditions,
            region_mask=region_mask,
        )
        t, x_t, velocity_target = self.flow_matcher.sample_location_and_conditional_flow(
            source,
            target,
        )
        velocity_pred = self.flow_model(x_t, conditions, t.to(target.device))
        loss = self.region_weighted_mse(
            prediction=velocity_pred,
            target=velocity_target,
            region_mask=region_mask,
            region_weight=self.region_weight,
        )
        return {
            "loss": loss,
            "velocity_loss": (velocity_pred - velocity_target).pow(2).mean().detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        conditions: ConditionPyramid,
        shape: Tuple[int, int, int],
        steps: int = 50,
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")

        if device is None:
            device = conditions["down_conditions"][0].device
        x = torch.randn(shape, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((shape[0],), i / steps, device=device)
            x = x + self.flow_model(x, conditions, t) * dt
        return x
