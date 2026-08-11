"""Compact stride-1 residual 1D U-Net for ECG delineation."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from src.rcfm.interpretability.delineation_dataset import EVENT_NAMES, WAVE_NAMES


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        padding = 3 * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 7, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.skip = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        residual = self.skip(signal)
        signal = F.silu(self.norm1(self.conv1(signal)))
        signal = self.norm2(self.conv2(signal))
        return F.silu(signal + residual)


class DownBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv1d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = ResidualBlock1D(out_channels, out_channels)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(signal))


class UpBlock1D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = ResidualBlock1D(out_channels + skip_channels, out_channels)

    def forward(self, signal: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        signal = self.up(signal)
        if signal.shape[-1] != skip.shape[-1]:
            raise ValueError("decoder and skip temporal dimensions do not align")
        return self.block(torch.cat([signal, skip], dim=1))


class DelineationResUNet1D(nn.Module):
    """Single-channel encoder-decoder with separate region and fiducial heads."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        if base_channels <= 0 or base_channels % 2:
            raise ValueError("base_channels must be a positive even integer")
        channels = [base_channels * factor for factor in (1, 2, 4, 8)]
        self.stem = ResidualBlock1D(1, channels[0])
        self.down1 = DownBlock1D(channels[0], channels[1])
        self.down2 = DownBlock1D(channels[1], channels[2])
        self.down3 = DownBlock1D(channels[2], channels[3])
        self.bottleneck = nn.Sequential(
            ResidualBlock1D(channels[3], channels[3], dilation=2),
            ResidualBlock1D(channels[3], channels[3], dilation=4),
            ResidualBlock1D(channels[3], channels[3], dilation=8),
        )
        self.up3 = UpBlock1D(channels[3], channels[2], channels[2])
        self.up2 = UpBlock1D(channels[2], channels[1], channels[1])
        self.up1 = UpBlock1D(channels[1], channels[0], channels[0])
        self.region_head = nn.Conv1d(channels[0], len(WAVE_NAMES), 1)
        self.fiducial_head = nn.Conv1d(channels[0], len(EVENT_NAMES), 1)

    def forward(self, signal: torch.Tensor) -> dict[str, torch.Tensor]:
        if signal.ndim != 3 or signal.shape[1] != 1 or signal.shape[-1] % 8:
            raise ValueError("input must have shape (batch, 1, length) with length divisible by 8")
        skip1 = self.stem(signal)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        latent = self.down3(skip3)
        latent = self.bottleneck(latent)
        decoded = self.up3(latent, skip3)
        decoded = self.up2(decoded, skip2)
        decoded = self.up1(decoded, skip1)
        return {
            "region_logits": self.region_head(decoded),
            "fiducial_logits": self.fiducial_head(decoded),
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def delineation_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    focal_gamma: float = 2.0,
    region_bce_weight: float = 1.0,
    region_dice_weight: float = 1.0,
    fiducial_weight: float = 0.5,
    heatmap_positive_weight: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Masked focal BCE, class-wise soft Dice, and weighted heatmap MSE."""

    region_logits = outputs["region_logits"]
    fiducial_logits = outputs["fiducial_logits"]
    regions = batch["regions"].to(region_logits.dtype)
    region_mask = batch["region_mask"].to(region_logits.dtype)
    heatmaps = batch["heatmaps"].to(fiducial_logits.dtype)
    heatmap_mask = batch["heatmap_mask"].to(fiducial_logits.dtype)
    if region_logits.shape != regions.shape or region_mask.shape != regions.shape:
        raise ValueError("region predictions and supervision do not align")
    if fiducial_logits.shape != heatmaps.shape or heatmap_mask.shape != heatmaps.shape:
        raise ValueError("fiducial predictions and supervision do not align")

    region_bce = F.binary_cross_entropy_with_logits(region_logits, regions, reduction="none")
    region_probabilities = torch.sigmoid(region_logits)
    focal_factor = torch.where(regions > 0.5, 1.0 - region_probabilities, region_probabilities)
    region_focal = _masked_mean(region_bce * focal_factor.pow(focal_gamma), region_mask)
    intersection = (region_probabilities * regions * region_mask).sum(dim=(0, 2))
    denominator = ((region_probabilities + regions) * region_mask).sum(dim=(0, 2))
    class_valid = (region_mask.sum(dim=(0, 2)) > 0).to(region_logits.dtype)
    dice_by_class = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = ((1.0 - dice_by_class) * class_valid).sum() / class_valid.sum().clamp_min(1.0)

    fiducial_probabilities = torch.sigmoid(fiducial_logits)
    heatmap_weights = 1.0 + heatmap_positive_weight * heatmaps
    heatmap_loss = _masked_mean(
        (fiducial_probabilities - heatmaps).square() * heatmap_weights,
        heatmap_mask,
    )
    total = (
        region_bce_weight * region_focal
        + region_dice_weight * dice_loss
        + fiducial_weight * heatmap_loss
    )
    return {
        "loss": total,
        "region_focal_loss": region_focal.detach(),
        "region_dice_loss": dice_loss.detach(),
        "fiducial_heatmap_loss": heatmap_loss.detach(),
        **{
            f"region_soft_dice/{name}": dice_by_class[index].detach()
            for index, name in enumerate(WAVE_NAMES)
        },
    }
