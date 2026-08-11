"""Region-aware conditional flow matching for 1-D physiological signals."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from conditional_flow_matcher import (
    ConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from optimal_transport import OTPlanSampler
from src.rcfm.ot_diagnostics import from_sampler_diagnostics


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
        flow_matcher_type: str = "conditional",
        sigma: float = 0.0,
        region_weight: float = 0.01,
        use_minibatch_ot: bool = True,
        ot_method: str = "sinkhorn",
        ot_reg: float = 0.05,
        ot_normalize_cost: bool = False,
        ot_diagnostics: bool = True,
        ot_strict_mode: bool = True,
        ot_sampling_strategy: str = "multinomial",
        allow_noncanonical_ot_path: bool = False,
        association_debug: bool = False,
    ) -> None:
        super().__init__()
        self.flow_model = flow_model
        self.region_weight = region_weight
        self.use_minibatch_ot = use_minibatch_ot
        self.flow_matcher_type = flow_matcher_type
        self.sigma = float(sigma)
        self.ot_diagnostics_enabled = ot_diagnostics
        self.ot_strict_mode = ot_strict_mode
        self.ot_sampling_strategy = ot_sampling_strategy
        self.association_debug = association_debug
        self._last_ot_diagnostics: dict[str, float] = {}
        self._last_ot_histograms: dict[str, object] = {}
        self._last_source_indices: Optional[torch.Tensor] = None
        self._last_target_indices: Optional[torch.Tensor] = None
        self.ot_sampler = (
            OTPlanSampler(method=ot_method, reg=ot_reg, normalize_cost=ot_normalize_cost)
            if use_minibatch_ot
            else None
        )

        if flow_matcher_type == "conditional" and sigma != 0:
            raise ValueError(
                "Canonical RCFM uses the deterministic linear path and requires sigma=0."
            )
        if flow_matcher_type == "sb" and not use_minibatch_ot:
            raise ValueError(
                "SB path ablation requires one observable outer minibatch OT coupling "
                "so target, condition, and mask are reindexed together."
            )
        if (
            flow_matcher_type in {"target", "vp"}
            and use_minibatch_ot
            and not allow_noncanonical_ot_path
        ):
            raise ValueError(
                f"{flow_matcher_type} flow matching with outer minibatch OT is "
                "noncanonical and requires allow_noncanonical_ot_path=True."
            )
        if ot_sampling_strategy not in {"multinomial", "assignment"}:
            raise ValueError("ot_sampling_strategy must be multinomial or assignment")

        if flow_matcher_type == "conditional":
            self.flow_matcher = ConditionalFlowMatcher(sigma=sigma)
        elif flow_matcher_type == "target":
            self.flow_matcher = TargetConditionalFlowMatcher(sigma=sigma)
        elif flow_matcher_type == "sb":
            self.flow_matcher = SchrodingerBridgeConditionalFlowMatcher(
                sigma=sigma,
                ot_method=ot_method,
                apply_ot=False,
            )
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
        sample_metadata: Optional[Mapping[str, object]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, ConditionPyramid, Optional[torch.Tensor]]:
        self._validate_target_batch_sizes(target, conditions, region_mask)
        if self.association_debug and self.training and sample_metadata is None:
            raise ValueError("association_debug requires sample_metadata")
        if not self.use_minibatch_ot or self.ot_sampler is None:
            target_idx = torch.arange(target.shape[0], device=target.device)
            self._validate_sample_metadata(sample_metadata, target_idx, target.shape[0])
            return source, target, conditions, region_mask

        source_idx, target_idx = self._sample_ot_indices(source, target)
        self._validate_ot_indices(source_idx, target_idx, source, target)
        self._validate_sample_metadata(sample_metadata, target_idx, target.shape[0])

        source = source[source_idx]
        target = target[target_idx]
        conditions = _index_conditions(conditions, target_idx)
        if region_mask is not None:
            region_mask = region_mask[target_idx]
        return source, target, conditions, region_mask

    @staticmethod
    def _validate_target_batch_sizes(
        target: torch.Tensor,
        conditions: ConditionPyramid,
        region_mask: Optional[torch.Tensor],
    ) -> None:
        batch_size = target.shape[0]
        for branch in ("down_conditions", "up_conditions"):
            if branch not in conditions or not conditions[branch]:
                raise ValueError(f"conditions must contain nonempty {branch}")
            if any(feature.shape[0] != batch_size for feature in conditions[branch]):
                raise ValueError("target and condition batch sizes must match")
        if region_mask is not None and region_mask.shape[0] != batch_size:
            raise ValueError("target and region-mask batch sizes must match")

    def _sample_ot_indices(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.ot_sampler is None:
            raise RuntimeError("OT indices requested while minibatch OT is disabled")
        pi, sampler_diagnostics = self.ot_sampler.get_map(
            source,
            target,
            strict=self.ot_strict_mode,
            return_diagnostics=True,
        )
        source_idx, target_idx = self.ot_sampler.sample_map(
            pi,
            source.shape[0],
            strategy=self.ot_sampling_strategy,
        )
        source_tensor = torch.as_tensor(source_idx, device=source.device, dtype=torch.long)
        target_tensor = torch.as_tensor(target_idx, device=target.device, dtype=torch.long)
        self._validate_ot_indices(source_tensor, target_tensor, source, target)
        self._last_source_indices = source_tensor.detach().clone()
        self._last_target_indices = target_tensor.detach().clone()
        if self.ot_diagnostics_enabled:
            self._last_ot_diagnostics, self._last_ot_histograms = from_sampler_diagnostics(
                sampler_diagnostics,
                pi,
                source_tensor.detach().cpu().numpy(),
                target_tensor.detach().cpu().numpy(),
            )
        else:
            self._last_ot_diagnostics = {}
            self._last_ot_histograms = {}
        return (
            source_tensor,
            target_tensor,
        )

    @staticmethod
    def _validate_ot_indices(
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        if source_idx.dtype != torch.long or target_idx.dtype != torch.long:
            raise TypeError("OT indices must have dtype torch.long")
        if source_idx.device != source.device or target_idx.device != target.device:
            raise ValueError("OT indices must be on the same device as their tensors")
        if source_idx.ndim != 1 or target_idx.ndim != 1:
            raise ValueError("OT indices must be one-dimensional")
        if source_idx.numel() != source.shape[0] or target_idx.numel() != source.shape[0]:
            raise ValueError("OT indices must preserve the source minibatch size")
        if torch.any(source_idx < 0) or torch.any(source_idx >= source.shape[0]):
            raise IndexError("OT source index is out of range")
        if torch.any(target_idx < 0) or torch.any(target_idx >= target.shape[0]):
            raise IndexError("OT target index is out of range")

    @staticmethod
    def _validate_sample_metadata(
        sample_metadata: Optional[Mapping[str, object]],
        target_idx: torch.Tensor,
        target_batch_size: int,
    ) -> None:
        if sample_metadata is None:
            return
        keys = ("target_sample_id", "condition_target_id", "mask_target_id")
        missing = [key for key in keys if key not in sample_metadata]
        if missing:
            raise KeyError(f"sample_metadata is missing association keys: {missing}")
        indexed = []
        cpu_index = target_idx.detach().cpu()
        for key in keys:
            values = sample_metadata[key]
            if torch.is_tensor(values):
                if values.ndim == 0 or values.shape[0] != target_batch_size:
                    raise ValueError(f"sample_metadata[{key!r}] has the wrong batch size")
                indexed.append(values.detach().cpu()[cpu_index])
            else:
                values = list(values)  # type: ignore[arg-type]
                if len(values) != target_batch_size:
                    raise ValueError(f"sample_metadata[{key!r}] has the wrong batch size")
                indexed.append([values[index] for index in cpu_index.tolist()])
        reference = indexed[0]
        for candidate in indexed[1:]:
            if torch.is_tensor(reference) and torch.is_tensor(candidate):
                matches = torch.equal(reference, candidate)
            else:
                matches = list(reference) == list(candidate)
            if not matches:
                raise ValueError(
                    "target, condition, and region-mask metadata associations diverged"
                )

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
        sample_metadata: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, torch.Tensor | str]:
        if source is None:
            source = torch.randn_like(target)

        source, target, conditions, region_mask = self._apply_minibatch_ot(
            source=source,
            target=target,
            conditions=conditions,
            region_mask=region_mask,
            sample_metadata=sample_metadata,
        )
        t, x_t, velocity_target = self.flow_matcher.sample_location_and_conditional_flow(
            source,
            target,
        )
        velocity_pred = self.flow_model(x_t, conditions, t.to(target.device))
        squared_error = (velocity_pred - velocity_target).pow(2)
        velocity_mse = squared_error.mean()
        loss = self.region_weighted_mse(
            prediction=velocity_pred,
            target=velocity_target,
            region_mask=region_mask,
            region_weight=self.region_weight,
        )
        if region_mask is None:
            mask = torch.zeros_like(squared_error)
        else:
            mask = region_mask.to(
                device=velocity_pred.device,
                dtype=velocity_pred.dtype,
            ).clamp(0.0, 1.0)
            while mask.dim() < velocity_pred.dim():
                mask = mask.unsqueeze(1)
            mask = torch.broadcast_to(mask, squared_error.shape)
        roi_mass = mask.sum()
        non_roi_mass = (1.0 - mask).sum()
        roi_mse = (squared_error * mask).sum() / roi_mass.clamp_min(1.0)
        non_roi_mse = (squared_error * (1.0 - mask)).sum() / non_roi_mass.clamp_min(1.0)

        flat_source = source.flatten(1)
        flat_target = target.flatten(1)
        flat_velocity_target = velocity_target.flatten(1)
        flat_velocity_prediction = velocity_pred.flatten(1)
        velocity_cosine_denominator = (
            flat_velocity_target.norm(dim=1) * flat_velocity_prediction.norm(dim=1)
        )
        velocity_cosine = (
            (flat_velocity_target * flat_velocity_prediction).sum(dim=1)
            / velocity_cosine_denominator.clamp_min(1e-12)
        )
        path_error = velocity_pred - velocity_target
        diagnostics: Dict[str, torch.Tensor | str] = {
            "loss": loss,
            "train/total_loss": loss.detach(),
            "train/velocity_mse": velocity_mse.detach(),
            "train/roi_mse": roi_mse.detach(),
            "train/non_roi_mse": non_roi_mse.detach(),
            "train/mask_occupancy": mask.mean().detach(),
            "train/effective_mean_weight": (
                1.0 + self.region_weight * mask.mean()
            ).detach(),
            "flow/source_norm": flat_source.norm(dim=1).mean().detach(),
            "flow/target_norm": flat_target.norm(dim=1).mean().detach(),
            "flow/x_t_norm": x_t.flatten(1).norm(dim=1).mean().detach(),
            "flow/target_velocity_norm": flat_velocity_target.norm(dim=1).mean().detach(),
            "flow/predicted_velocity_norm": flat_velocity_prediction.norm(dim=1).mean().detach(),
            "flow/velocity_error_norm": path_error.flatten(1).norm(dim=1).mean().detach(),
            "flow/velocity_cosine_similarity": velocity_cosine.mean().detach(),
            "flow/time_mean": t.mean().detach(),
            "flow/time_std": t.std(unbiased=False).detach(),
            "flow/path_type": self.flow_matcher_type,
            "flow/sigma": torch.tensor(self.sigma, device=target.device),
        }
        diagnostics.update(
            {
                key: torch.tensor(value, device=target.device)
                for key, value in self._last_ot_diagnostics.items()
            }
        )
        return diagnostics

    @torch.no_grad()
    def sample(
        self,
        conditions: ConditionPyramid,
        shape: Tuple[int, int, int],
        steps: int = 50,
        device: Optional[torch.device | str] = None,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")

        if device is None:
            device = conditions["down_conditions"][0].device
        if initial_noise is None:
            x = torch.randn(shape, device=device)
        else:
            if tuple(initial_noise.shape) != tuple(shape):
                raise ValueError("initial_noise shape must equal the requested sample shape")
            if initial_noise.device != torch.device(device):
                raise ValueError("initial_noise must be on the requested sampling device")
            x = initial_noise.clone()
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((shape[0],), i / steps, device=device)
            x = x + self.flow_model(x, conditions, t) * dt
        return x
