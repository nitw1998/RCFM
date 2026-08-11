"""Source-only FFT cycle views for the CATransformer reproduction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CycleSelection:
    """Per-record top-k FFT frequencies and their derived cycle lengths."""

    frequencies: torch.Tensor
    periods: torch.Tensor
    amplitudes: torch.Tensor
    weights: torch.Tensor
    valid_lengths: torch.Tensor
    fallback: torch.Tensor


def _validate_source(source: torch.Tensor) -> tuple[int, int]:
    if source.ndim != 3 or source.shape[1] != 1:
        raise ValueError("CAT cycle extraction requires source shape (batch, 1, length)")
    if source.shape[-1] < 4:
        raise ValueError("CAT cycle extraction requires at least four samples")
    if not torch.isfinite(source).all():
        raise ValueError("CAT source must contain only finite values")
    return int(source.shape[0]), int(source.shape[-1])


def _prefix_lengths(source_mask: torch.Tensor, batch: int, length: int) -> torch.Tensor:
    mask = torch.as_tensor(source_mask, dtype=torch.bool, device=source_mask.device)
    if mask.shape == (batch, 1, length):
        mask = mask[:, 0]
    if mask.shape != (batch, length):
        raise ValueError("source_mask must have shape (batch, length) or (batch, 1, length)")
    lengths = mask.sum(dim=-1)
    if torch.any(lengths < 4):
        raise ValueError("every masked CAT source must retain at least four samples")
    expected = torch.arange(length, device=mask.device)[None, :] < lengths[:, None]
    if not torch.equal(mask, expected):
        raise ValueError("source_mask must describe a contiguous valid prefix")
    return lengths


class SourceCycleExtractor(nn.Module):
    """Implement paper equations (1)--(2) using only the source waveform."""

    def __init__(self, top_k: int = 2, energy_epsilon: float = 1e-8) -> None:
        super().__init__()
        if top_k <= 0 or energy_epsilon <= 0:
            raise ValueError("top_k and energy_epsilon must be positive")
        self.top_k = int(top_k)
        self.energy_epsilon = float(energy_epsilon)

    def _select_one(self, signal: torch.Tensor) -> tuple[torch.Tensor, ...]:
        length = int(signal.numel())
        spectrum = torch.abs(torch.fft.rfft(signal))
        candidates = spectrum[1 : length // 2 + 1]
        if len(candidates) < self.top_k:
            raise ValueError("source is too short for the requested number of FFT frequencies")
        failed = (not bool(torch.isfinite(candidates).all())) or bool(
            torch.max(candidates) <= self.energy_epsilon
        )
        if failed:
            frequencies = torch.arange(1, self.top_k + 1, device=signal.device)
            amplitudes = torch.zeros(self.top_k, dtype=signal.dtype, device=signal.device)
        else:
            order = torch.argsort(candidates, descending=True, stable=True)[: self.top_k]
            frequencies = order + 1
            amplitudes = candidates[order]
        periods = torch.div(length + frequencies - 1, frequencies, rounding_mode="floor")
        weights = torch.softmax(amplitudes, dim=0)
        return frequencies, periods, amplitudes, weights, torch.tensor(failed, device=signal.device)

    def forward(
        self, source: torch.Tensor, source_mask: torch.Tensor | None = None
    ) -> CycleSelection:
        batch, length = _validate_source(source)
        if source_mask is None:
            valid_lengths = torch.full((batch,), length, dtype=torch.long, device=source.device)
            spectrum = torch.abs(torch.fft.rfft(source[:, 0], dim=-1))
            candidates = spectrum[:, 1 : length // 2 + 1]
            if candidates.shape[1] < self.top_k:
                raise ValueError("source is too short for the requested number of FFT frequencies")
            fallback = (~torch.isfinite(candidates).all(dim=1)) | (
                torch.amax(candidates, dim=1) <= self.energy_epsilon
            )
            safe = torch.where(torch.isfinite(candidates), candidates, torch.zeros_like(candidates))
            order = torch.argsort(safe, dim=1, descending=True, stable=True)[:, : self.top_k]
            frequencies = order + 1
            amplitudes = torch.gather(safe, 1, order)
            if torch.any(fallback):
                defaults = torch.arange(1, self.top_k + 1, device=source.device)[None, :]
                frequencies = torch.where(fallback[:, None], defaults, frequencies)
                amplitudes = torch.where(fallback[:, None], torch.zeros_like(amplitudes), amplitudes)
            periods = torch.div(
                valid_lengths[:, None] + frequencies - 1,
                frequencies,
                rounding_mode="floor",
            )
            weights = torch.softmax(amplitudes, dim=1)
        else:
            valid_lengths = _prefix_lengths(source_mask.to(source.device), batch, length)
            selected = [self._select_one(source[index, 0, : int(valid_lengths[index])]) for index in range(batch)]
            frequencies = torch.stack([item[0] for item in selected])
            periods = torch.stack([item[1] for item in selected])
            amplitudes = torch.stack([item[2] for item in selected])
            weights = torch.stack([item[3] for item in selected])
            fallback = torch.stack([item[4] for item in selected]).bool()
        return CycleSelection(
            frequencies=frequencies.long(), periods=periods.long(), amplitudes=amplitudes,
            weights=weights, valid_lengths=valid_lengths.long(), fallback=fallback.bool(),
        )


class CycleViewBuilder(nn.Module):
    """Implement paper equations (3)--(4) with explicit padding masks."""

    def __init__(self, patch_width: int) -> None:
        super().__init__()
        if patch_width <= 0:
            raise ValueError("patch_width must be positive")
        self.patch_width = int(patch_width)

    def forward(
        self, source: torch.Tensor, selection: CycleSelection
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        batch, _ = _validate_source(source)
        if selection.frequencies.shape[0] != batch:
            raise ValueError("cycle selection and source batch sizes differ")
        views: list[torch.Tensor] = []
        valid_tokens: list[torch.Tensor] = []
        for branch in range(selection.frequencies.shape[1]):
            maximum_period = int(selection.periods[:, branch].max().item())
            view = source.new_zeros((batch, maximum_period, self.patch_width))
            token_mask = torch.zeros((batch, maximum_period), dtype=torch.bool, device=source.device)
            frequency_values = selection.frequencies[:, branch]
            period_values = selection.periods[:, branch]
            group_codes = frequency_values * (source.shape[-1] + 1) + period_values
            for code in torch.unique(group_codes, sorted=True).detach().cpu().tolist():
                rows = torch.nonzero(group_codes == code, as_tuple=False).flatten()
                frequency = int(frequency_values[rows[0]].item())
                period = int(period_values[rows[0]].item())
                valid_lengths = selection.valid_lengths[rows]
                signals = source[rows, 0]
                valid = (
                    torch.arange(source.shape[-1], device=source.device)[None, :]
                    < valid_lengths[:, None]
                )
                signals = torch.where(valid, signals, torch.zeros_like(signals))
                padded_length = period * frequency
                padded = source.new_zeros((len(rows), padded_length))
                copy_length = min(source.shape[-1], padded_length)
                padded[:, :copy_length] = signals[:, :copy_length]
                cycle = padded.reshape(len(rows), period, frequency)
                if frequency < self.patch_width:
                    tail = cycle[:, :, -1:].expand(
                        len(rows), period, self.patch_width - frequency
                    )
                    patch = torch.cat((cycle, tail), dim=2)
                else:
                    patch = cycle[:, :, : self.patch_width]
                view[rows, :period] = patch
                token_mask[rows, :period] = True
            views.append(view)
            valid_tokens.append(token_mask)
        return views, valid_tokens
