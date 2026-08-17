"""Lightweight deterministic 1-D CNN for direct conditional regression."""

from __future__ import annotations

import torch
from torch import nn


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 5) -> None:
        super().__init__()
        if channels <= 0 or dilation <= 0 or kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("channels/dilation must be positive and kernel size must be odd")
        padding = dilation * (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size, padding=padding,
                dilation=dilation, groups=channels, bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class DirectRegressionCNN(nn.Module):
    """One-pass waveform regression with a full-window dilated receptive field."""

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        width: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    ) -> None:
        super().__init__()
        if input_channels <= 0 or output_channels <= 0 or width <= 0 or not dilations:
            raise ValueError("channel counts, width, and dilations must be positive")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.width = int(width)
        self.dilations = tuple(int(value) for value in dilations)
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, width, 7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(DepthwiseResidualBlock(width, dilation) for dilation in self.dilations)
        )
        self.head = nn.Conv1d(width, output_channels, 1)
        self.linear_skip = nn.Conv1d(input_channels, output_channels, 1)

    @property
    def receptive_field(self) -> int:
        return 7 + 4 * sum(self.dilations)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 3 or condition.shape[1] != self.input_channels:
            raise ValueError("condition must have shape (batch, input_channels, time)")
        return self.head(self.blocks(self.stem(condition))) + self.linear_skip(condition)
