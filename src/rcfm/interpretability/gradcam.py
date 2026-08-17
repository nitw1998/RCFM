"""One-dimensional Grad-CAM primitives with explicit temporal stitching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def validation_crop_starts(total_length: int, crop_length: int, stride: int) -> list[int]:
    """Match the PTB-XL benchmark's validation crop enumeration."""

    if total_length < 1 or crop_length < 1 or stride < 1:
        raise ValueError("total_length, crop_length, and stride must be positive")
    if total_length < crop_length:
        raise ValueError("total_length must be at least crop_length")
    return [
        start
        for start in range(0, total_length, stride)
        if min(start + crop_length, total_length) - start >= crop_length
    ]


def covering_crop_starts(total_length: int, crop_length: int, stride: int) -> list[int]:
    """Return full crops and add one end-anchored crop when needed for coverage."""

    starts = validation_crop_starts(total_length, crop_length, stride)
    final_start = total_length - crop_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def gradcam_native_multi_1d(
    model: torch.nn.Module,
    target_layers: Mapping[str, torch.nn.Module],
    inputs: torch.Tensor,
    target_index: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return native-resolution positive Grad-CAMs from one backward pass."""

    if inputs.ndim != 3 or inputs.shape[0] != 1:
        raise ValueError("inputs must have shape (1, channels, time)")
    if target_index < 0:
        raise ValueError("target_index must be nonnegative")
    if not target_layers:
        raise ValueError("target_layers must be nonempty")

    captured: dict[str, torch.Tensor] = {}
    handles = []

    def hook_for(name: str):
        def capture_activation(_module, _module_inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("target layer output must be a tensor")
            output.retain_grad()
            captured[name] = output

        return capture_activation

    for name, layer in target_layers.items():
        if not name:
            raise ValueError("target layer names must be nonempty")
        handles.append(layer.register_forward_hook(hook_for(name)))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("model output must have shape (1, classes)")
        if target_index >= logits.shape[1]:
            raise IndexError("target_index exceeds model output width")
        logits[0, target_index].backward()
        cams: dict[str, np.ndarray] = {}
        for name in target_layers:
            activation = captured.get(name)
            if activation is None or activation.grad is None:
                raise RuntimeError(f"Grad-CAM target layer {name!r} did not retain a gradient")
            if activation.ndim != 3 or activation.shape[0] != 1:
                raise ValueError("target layers must produce shape (1, channels, time)")
            weights = activation.grad.mean(dim=-1, keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1))
            cams[name] = cam[0].detach().cpu().numpy().astype(np.float32, copy=False)
        logits_array = logits[0].detach().cpu().numpy().astype(np.float32, copy=False)
    finally:
        for handle in handles:
            handle.remove()

    if not np.all(np.isfinite(logits_array)) or any(
        not np.all(np.isfinite(cam)) for cam in cams.values()
    ):
        raise ValueError("Grad-CAM or logits contain nonfinite values")
    return cams, logits_array


def gradcam_native_multi_target_1d(
    model: torch.nn.Module,
    target_layers: Mapping[str, torch.nn.Module],
    inputs: torch.Tensor,
    target_indices: Sequence[int],
    reduction: str = "mean",
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return Grad-CAMs for an aggregate of known-positive class logits."""

    indices = [int(index) for index in target_indices]
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError("target_indices must be nonempty, unique, and nonnegative")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be mean or sum")
    if inputs.ndim != 3 or inputs.shape[0] != 1:
        raise ValueError("inputs must have shape (1, channels, time)")
    if not target_layers:
        raise ValueError("target_layers must be nonempty")

    captured: dict[str, torch.Tensor] = {}
    handles = []

    def hook_for(name: str):
        def capture_activation(_module, _module_inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("target layer output must be a tensor")
            output.retain_grad()
            captured[name] = output

        return capture_activation

    for name, layer in target_layers.items():
        if not name:
            raise ValueError("target layer names must be nonempty")
        handles.append(layer.register_forward_hook(hook_for(name)))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("model output must have shape (1, classes)")
        if max(indices) >= logits.shape[1]:
            raise IndexError("a target index exceeds model output width")
        score = logits[0, indices].mean() if reduction == "mean" else logits[0, indices].sum()
        score.backward()
        cams: dict[str, np.ndarray] = {}
        for name in target_layers:
            activation = captured.get(name)
            if activation is None or activation.grad is None:
                raise RuntimeError(f"Grad-CAM target layer {name!r} did not retain a gradient")
            if activation.ndim != 3 or activation.shape[0] != 1:
                raise ValueError("target layers must produce shape (1, channels, time)")
            weights = activation.grad.mean(dim=-1, keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1))
            cams[name] = cam[0].detach().cpu().numpy().astype(np.float32, copy=False)
        logits_array = logits[0].detach().cpu().numpy().astype(np.float32, copy=False)
    finally:
        for handle in handles:
            handle.remove()
    if not np.all(np.isfinite(logits_array)) or any(
        not np.all(np.isfinite(cam)) for cam in cams.values()
    ):
        raise ValueError("Grad-CAM or logits contain nonfinite values")
    return cams, logits_array


def project_cam_to_sample_grid(
    native_cam: np.ndarray,
    output_length: int,
    feature_stride: float,
    feature_offset: float = 0.0,
) -> np.ndarray:
    """Project native CAM bins using their explicit input-sample center coordinates."""

    values = np.asarray(native_cam, dtype=np.float32)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("native_cam must be a finite one-dimensional array")
    if len(values) < 1 or output_length < 1 or feature_stride <= 0:
        raise ValueError("CAM length, output_length, and feature_stride must be positive")
    centers = feature_offset + feature_stride * np.arange(len(values), dtype=np.float64)
    if centers[0] < 0 or centers[-1] > output_length - 1 + feature_stride:
        raise ValueError("feature centers are incompatible with the requested output grid")
    samples = np.arange(output_length, dtype=np.float64)
    if len(values) == 1:
        return np.full(output_length, values[0], dtype=np.float32)
    projected = np.interp(samples, centers, values, left=values[0], right=values[-1])
    return projected.astype(np.float32, copy=False)


def resample_mask_to_sample_grid(
    values: np.ndarray,
    source_rate_hz: float,
    target_rate_hz: float,
    output_length: int | None = None,
) -> np.ndarray:
    """Linearly resample a mask on sample-center time coordinates without filter delay."""

    mask = np.asarray(values, dtype=np.float32)
    if mask.ndim != 1 or not np.all(np.isfinite(mask)):
        raise ValueError("values must be a finite one-dimensional array")
    if len(mask) < 1 or source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("mask length and sampling rates must be positive")
    if output_length is None:
        output_length = int(round(len(mask) * target_rate_hz / source_rate_hz))
    if output_length < 1:
        raise ValueError("output_length must be positive")
    source_times = np.arange(len(mask), dtype=np.float64) / source_rate_hz
    target_times = np.arange(output_length, dtype=np.float64) / target_rate_hz
    output = np.interp(
        target_times,
        source_times,
        mask,
        left=float(mask[0]),
        right=float(mask[-1]),
    )
    return output.astype(np.float32, copy=False)


def gradcam_1d(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    inputs: torch.Tensor,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an unnormalized positive Grad-CAM and logits for one input crop."""

    native, logits_array = gradcam_native_multi_1d(
        model,
        {"target": target_layer},
        inputs,
        target_index,
    )
    cam = torch.from_numpy(native["target"]).reshape(1, 1, -1)
    cam = F.interpolate(cam, size=inputs.shape[-1], mode="linear", align_corners=False)
    cam_array = cam[0, 0].numpy().astype(np.float32, copy=False)

    if not np.all(np.isfinite(cam_array)):
        raise ValueError("Grad-CAM contains nonfinite values")
    return cam_array, logits_array


def stitch_temporal_cams(
    cams: Sequence[np.ndarray],
    starts: Sequence[int],
    total_length: int,
) -> np.ndarray:
    """Mean overlapping crop CAMs on their original temporal coordinates."""

    if len(cams) != len(starts) or not cams:
        raise ValueError("cams and starts must be nonempty and have equal length")
    if total_length < 1:
        raise ValueError("total_length must be positive")
    output = np.zeros(total_length, dtype=np.float64)
    counts = np.zeros(total_length, dtype=np.int64)
    for cam, start in zip(cams, starts):
        values = np.asarray(cam, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("each CAM must be a finite one-dimensional array")
        end = int(start) + len(values)
        if start < 0 or end > total_length:
            raise ValueError("a CAM falls outside total_length")
        output[start:end] += values
        counts[start:end] += 1
    if np.any(counts == 0):
        raise ValueError("crop CAMs do not cover the full temporal record")
    return (output / counts).astype(np.float32)


def normalize_soft_mask(values: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalize one record to [0, 1], reporting a degenerate constant mask."""

    mask = np.asarray(values, dtype=np.float32)
    if mask.ndim != 1 or not np.all(np.isfinite(mask)):
        raise ValueError("values must be a finite one-dimensional array")
    minimum = float(mask.min())
    span = float(mask.max()) - minimum
    if span <= np.finfo(np.float32).eps:
        return np.zeros_like(mask), True
    return ((mask - minimum) / span).astype(np.float32, copy=False), False
