"""Conditional flow matching paths used by RCFM."""

from __future__ import annotations

import math
import warnings
from typing import Union

import torch

from optimal_transport import OTPlanSampler


def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class ConditionalFlowMatcher:
    def __init__(self, sigma: Union[float, int] = 0.0):
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        sigma_t = pad_t_like_x(self.compute_sigma_t(t), x0)
        return self.compute_mu_t(x0, x1, t) + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        if len(t) != x0.shape[0]:
            raise ValueError("t must have the batch dimension")
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        return t, xt, ut

    def compute_lambda(self, t):
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)


class ExactOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        out = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
        if return_noise:
            t, xt, ut, eps = out
            return t, xt, ut, y0, y1, eps
        t, xt, ut = out
        return t, xt, ut, y0, y1


class TargetConditionalFlowMatcher(ConditionalFlowMatcher):
    def compute_mu_t(self, x0, x1, t):
        del x0
        return pad_t_like_x(t, x1) * x1

    def compute_sigma_t(self, t):
        return 1 - (1 - self.sigma) * t

    def compute_conditional_flow(self, x0, x1, t, xt):
        del x0
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)


class SchrodingerBridgeConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self, sigma: Union[float, int] = 1.0, ot_method="exact"):
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        if sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma**2)

    def compute_sigma_t(self, t):
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(self, x0, x1, t, xt):
        t = pad_t_like_x(t, x0)
        mu_t = self.compute_mu_t(x0, x1, t)
        return ((1 - 2 * t) / (2 * t * (1 - t) + 1e-8)) * (xt - mu_t) + x1 - x0

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        out = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
        if return_noise:
            t, xt, ut, eps = out
            return t, xt, ut, y0, y1, eps
        t, xt, ut = out
        return t, xt, ut, y0, y1


class VariancePreservingConditionalFlowMatcher(ConditionalFlowMatcher):
    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def compute_conditional_flow(self, x0, x1, t, xt):
        del xt
        t = pad_t_like_x(t, x0)
        return math.pi / 2 * (
            torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0
        )
