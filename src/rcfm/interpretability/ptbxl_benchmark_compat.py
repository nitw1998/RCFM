"""Compatibility loader for the legacy PTB-XL benchmark XResNet1D checkpoint.

The architecture is imported from a user-supplied checkout of
ecg_ptbxl_benchmarking. Only the small fastai-v1 API surface needed to build
the model head is supplied here, so no third-party architecture source is
copied into this repository.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Collection
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


class LegacyFlatten(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.contiguous().view(inputs.size(0), -1)


def legacy_listify(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def legacy_bn_drop_lin(
    n_in: int,
    n_out: int,
    use_batch_norm: bool = True,
    dropout: float = 0.0,
    activation: nn.Module | None = None,
) -> list[nn.Module]:
    layers: list[nn.Module] = [nn.BatchNorm1d(n_in)] if use_batch_norm else []
    if dropout != 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(n_in, n_out))
    if activation is not None:
        layers.append(activation)
    return layers


def _install_fastai_v1_shim() -> None:
    try:
        importlib.import_module("fastai.layers")
        importlib.import_module("fastai.core")
        return
    except ModuleNotFoundError:
        pass

    fastai = types.ModuleType("fastai")
    layers = types.ModuleType("fastai.layers")
    core = types.ModuleType("fastai.core")
    shared = {
        "Optional": Optional,
        "Collection": Collection,
        "Floats": Union[float, Collection[float]],
        "Flatten": LegacyFlatten,
        "listify": legacy_listify,
        "bn_drop_lin": legacy_bn_drop_lin,
    }
    for module in (layers, core):
        module.__dict__.update(shared)
        module.__all__ = list(shared)
    fastai.layers = layers
    fastai.core = core
    sys.modules["fastai"] = fastai
    sys.modules["fastai.layers"] = layers
    sys.modules["fastai.core"] = core


def load_xresnet1d101(
    benchmark_code_root: Path,
    checkpoint_path: Path,
    map_location: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Build the official benchmark architecture and strictly restore weights."""

    code_root = benchmark_code_root.resolve()
    architecture_path = code_root / "models" / "xresnet1d.py"
    if not architecture_path.is_file():
        raise FileNotFoundError(f"missing benchmark architecture: {architecture_path}")
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

    loaded_models = sys.modules.get("models")
    if loaded_models is not None:
        module_file = Path(getattr(loaded_models, "__file__", "")).resolve()
        if code_root not in module_file.parents:
            raise RuntimeError("an unrelated top-level 'models' package is already imported")

    _install_fastai_v1_shim()
    sys.path.insert(0, str(code_root))
    try:
        module = importlib.import_module("models.xresnet1d")
    finally:
        sys.path.remove(str(code_root))

    model = module.xresnet1d101(
        num_classes=71,
        input_channels=12,
        kernel_size=5,
        ps_head=0.5,
        lin_ftrs_head=[128],
    )
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("legacy checkpoint must be a dictionary containing 'model'")
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint restoration reported incompatible keys")
    return model
