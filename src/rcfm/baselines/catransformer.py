"""Independent paper-based reproduction of CATransformer.

This is not an official implementation. It follows equations (1)--(10) and
Figure 1 of Yuan et al., while exposing omitted architectural choices.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .cat_cycle import CycleViewBuilder, SourceCycleExtractor


def _sinusoidal_positions(length: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(width, 1))
    )
    encoding = torch.zeros((length, width), device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class CycleAwareTransformerBlock(nn.Module):
    """One cycle-aware, Transformer, and adaptive-aggregation layer."""

    def __init__(
        self,
        input_length: int,
        top_k: int,
        patch_width: int,
        d_model: int,
        n_heads: int,
        encoder_layers: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_length = int(input_length)
        self.extractor = SourceCycleExtractor(top_k=top_k)
        self.view_builder = CycleViewBuilder(patch_width=patch_width)
        self.embedding = nn.Linear(patch_width, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=encoder_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(d_model)

    def _restore(
        self,
        encoded: torch.Tensor,
        token_mask: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoded.ndim != 3 or token_mask.shape != encoded.shape[:2]:
            raise ValueError("encoded CAT tokens and padding mask do not align")
        available = token_mask.sum(dim=1) * encoded.shape[-1]
        required = valid_lengths.clamp(max=self.input_length)
        if torch.any(available <= 0) or torch.any(required <= 0):
            raise ValueError("CAT restoration requires nonempty encoded and source sequences")
        maximum_index = torch.minimum(available - 1, required - 1)
        indices = torch.arange(self.input_length, device=encoded.device)[None, :]
        indices = torch.minimum(indices, maximum_index[:, None])
        flattened = encoded.reshape(encoded.shape[0], -1)
        restored = torch.gather(flattened, 1, indices).unsqueeze(1)
        return restored, available < required

    def forward(
        self, source: torch.Tensor, source_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        selection = self.extractor(source, source_mask=source_mask)
        views, token_masks = self.view_builder(source, selection)
        reconstructions = []
        reconstruction_fallbacks = []
        for view, valid in zip(views, token_masks):
            embedded = self.embedding(view)
            embedded = embedded + _sinusoidal_positions(
                embedded.shape[1], embedded.shape[2], embedded.device, embedded.dtype
            )[None, :, :]
            encoded = self.encoder(embedded, src_key_padding_mask=~valid)
            encoded = self.output_norm(encoded)
            restored, padded = self._restore(encoded, valid, selection.valid_lengths)
            reconstructions.append(restored)
            reconstruction_fallbacks.append(padded)
        stacked = torch.stack(reconstructions, dim=1)
        output = torch.sum(stacked * selection.weights[:, :, None, None], dim=1)
        diagnostics = {
            "frequencies": selection.frequencies,
            "periods": selection.periods,
            "weights": selection.weights,
            "cycle_fallback": selection.fallback,
            "reconstruction_fallback": torch.stack(reconstruction_fallbacks, dim=1).any(dim=1),
        }
        return output, diagnostics


class CATransformer(nn.Module):
    """Deterministic CAT-PPG reproduction with source-only cycle extraction."""

    reproduction_label = "CAT-PPG (reproduced)"
    nfe = 1

    def __init__(
        self,
        input_length: int = 512,
        output_channels: int = 1,
        cat_layers: int = 2,
        top_k: int = 2,
        patch_width: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        encoder_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_length <= 0 or output_channels <= 0 or cat_layers <= 0:
            raise ValueError("input_length, output_channels, and cat_layers must be positive")
        if output_channels != 1:
            raise ValueError(
                "paper-faithful CAT-PPG has one output channel; multi-lead outputs require "
                "a separately labelled adaptation head"
            )
        self.input_length = int(input_length)
        self.output_channels = int(output_channels)
        self.blocks = nn.ModuleList(
            [
                CycleAwareTransformerBlock(
                    input_length=input_length,
                    top_k=top_k,
                    patch_width=patch_width,
                    d_model=d_model,
                    n_heads=n_heads,
                    encoder_layers=encoder_layers,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(cat_layers)
            ]
        )

    def forward(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        if source.ndim != 3 or source.shape[1:] != (1, self.input_length):
            raise ValueError(f"CAT source must have shape (batch, 1, {self.input_length})")
        values = source
        diagnostics: dict[str, torch.Tensor] = {}
        for index, block in enumerate(self.blocks, start=1):
            values, block_diagnostics = block(values, source_mask=source_mask)
            diagnostics.update({f"layer_{index}_{key}": value for key, value in block_diagnostics.items()})
        if return_diagnostics:
            return values, diagnostics
        return values


class CATLoss(nn.Module):
    """Equation (10), with an explicit softmax definition for P and Q."""

    def __init__(self, kl_weight: float = 1.0, kl_temperature: float = 1.0) -> None:
        super().__init__()
        if kl_weight < 0 or kl_temperature <= 0:
            raise ValueError("kl_weight must be nonnegative and kl_temperature positive")
        self.kl_weight = float(kl_weight)
        self.kl_temperature = float(kl_temperature)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape:
            raise ValueError("CAT prediction and target shapes must match")
        mse = F.mse_loss(prediction, target)
        prediction_rows = prediction.reshape(-1, prediction.shape[-1]) / self.kl_temperature
        target_rows = target.reshape(-1, target.shape[-1]) / self.kl_temperature
        kl = F.kl_div(
            F.log_softmax(prediction_rows, dim=-1),
            F.softmax(target_rows, dim=-1),
            reduction="batchmean",
        )
        total = mse + self.kl_weight * kl
        return total, {"total_loss": total.detach(), "mse_loss": mse.detach(), "kl_loss": kl.detach()}
