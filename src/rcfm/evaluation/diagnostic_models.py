"""Frozen diagnostic-model adapters used by transfer and mismatch experiments."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import torch

from src.rcfm.interpretability.gradcam import covering_crop_starts
from src.rcfm.interpretability.ptbxl_benchmark_compat import load_xresnet1d101


@dataclass(frozen=True)
class DiagnosticModel:
    model: torch.nn.Module
    class_names: tuple[str, ...]
    device: torch.device
    input_rate_hz: int = 100
    crop_samples: int = 250
    crop_stride: int = 125
    normalization: str = "ptb_scalar"
    scaler_mean: float = 0.0
    scaler_scale: float = 1.0

    def class_index(self, name: str) -> int:
        matches = [index for index, value in enumerate(self.class_names) if value == name]
        if len(matches) != 1:
            raise ValueError(f"class name {name!r} does not occur exactly once")
        return matches[0]


def load_xresnet_adapter(
    *,
    checkpoint: Path,
    benchmark_code_root: Path,
    mlb_path: Path,
    scaler_path: Path,
    device: torch.device,
) -> DiagnosticModel:
    with mlb_path.open("rb") as handle:
        classes = tuple(str(value) for value in pickle.load(handle).classes_)
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)
    mean = float(np.asarray(scaler.mean_).reshape(-1)[0])
    scale = float(np.asarray(scaler.scale_).reshape(-1)[0])
    if len(classes) != 71 or not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid PTB-XL all-statements metadata")
    model = load_xresnet1d101(benchmark_code_root, checkpoint, map_location="cpu")
    return DiagnosticModel(
        model=model.eval().to(device), class_names=classes, device=device,
        normalization="ptb_scalar", scaler_mean=mean, scaler_scale=scale,
    )


def load_torchscript_adapter(
    *,
    checkpoint: Path,
    class_names_path: Path,
    device: torch.device,
    input_rate_hz: int,
    crop_samples: int,
    crop_stride: int,
    normalization: str,
) -> DiagnosticModel:
    names = tuple(
        line.strip() for line in class_names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not names or len(names) != len(set(names)):
        raise ValueError("TorchScript class-name file must contain unique nonempty lines")
    model = torch.jit.load(str(checkpoint.resolve()), map_location=device).eval()
    return DiagnosticModel(
        model=model, class_names=names, device=device, input_rate_hz=input_rate_hz,
        crop_samples=crop_samples, crop_stride=crop_stride, normalization=normalization,
    )


def prepare_record(
    waveform_time_leads: np.ndarray,
    *,
    source_rate_hz: int,
    adapter: DiagnosticModel,
) -> np.ndarray:
    values = np.asarray(waveform_time_leads, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 12 or not np.all(np.isfinite(values)):
        raise ValueError("record must have shape (time, 12) with finite values")
    output = resample_poly(
        values, adapter.input_rate_hz, source_rate_hz, axis=0, padtype="line"
    ).astype(np.float32)
    if adapter.normalization == "ptb_scalar":
        output = (output - adapter.scaler_mean) / adapter.scaler_scale
    elif adapter.normalization == "record_zscore":
        mean, scale = float(np.mean(output)), float(np.std(output))
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError("record has invalid global standard deviation")
        output = (output - mean) / (scale + 1e-8)
    elif adapter.normalization != "none":
        raise ValueError(f"unsupported normalization: {adapter.normalization}")
    return output.astype(np.float32, copy=False)


def record_crops(prepared: np.ndarray, adapter: DiagnosticModel, crop_offset: int = 0) -> tuple[np.ndarray, list[int]]:
    if len(prepared) < adapter.crop_samples:
        raise ValueError("record is shorter than one classifier crop")
    starts = covering_crop_starts(len(prepared), adapter.crop_samples, adapter.crop_stride)
    if crop_offset:
        maximum = len(prepared) - adapter.crop_samples
        starts = sorted(set(min(max(start + crop_offset, 0), maximum) for start in starts))
    crops = np.stack([
        prepared[start : start + adapter.crop_samples].T for start in starts
    ]).astype(np.float32)
    return crops, starts


@torch.inference_mode()
def predict_crops(adapter: DiagnosticModel, crops: np.ndarray, batch_size: int = 64) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(crops), batch_size):
        tensor = torch.from_numpy(crops[start : start + batch_size]).to(
            device=adapter.device, dtype=torch.float32
        )
        logits = adapter.model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("diagnostic model must return a (batch, classes) logit tensor")
        outputs.append(logits.detach().cpu().numpy().astype(np.float32))
    output = np.concatenate(outputs)
    if output.shape[1] != len(adapter.class_names) or not np.all(np.isfinite(output)):
        raise ValueError("diagnostic output violates class-width or finiteness contract")
    return output


def aggregate_crop_logits(logits: np.ndarray, rule: str) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("crop logits must be two-dimensional")
    if rule == "mean_logit":
        return np.mean(values, axis=0)
    if rule == "max_logit":
        return np.max(values, axis=0)
    raise ValueError("crop aggregation must be mean_logit or max_logit")


def predict_record(
    adapter: DiagnosticModel,
    waveform_time_leads: np.ndarray,
    *,
    source_rate_hz: int,
    aggregation: str,
    crop_offset: int = 0,
) -> np.ndarray:
    prepared = prepare_record(waveform_time_leads, source_rate_hz=source_rate_hz, adapter=adapter)
    crops, _ = record_crops(prepared, adapter, crop_offset=crop_offset)
    return aggregate_crop_logits(predict_crops(adapter, crops), aggregation)
