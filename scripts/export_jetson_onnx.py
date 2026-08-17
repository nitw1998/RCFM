#!/usr/bin/env python3
"""Export RCFM or RDDM neural components for TensorRT 8.0 on Jetson Nano.

The stochastic sampling loop deliberately remains outside ONNX.  Every graph has
static batch-1, 512-sample inputs and uses ONNX opset 13 for JetPack 4.6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model import ConditionNet, DiffusionUNetCrossAttention  # noqa: E402


OPSET = 13
LENGTH = 512
CONDITION_NAMES = (
    "condition_down_0", "condition_down_1", "condition_down_2",
    "condition_down_3", "condition_down_4", "condition_down_5",
    "condition_up_0", "condition_up_1", "condition_up_2",
    "condition_up_3", "condition_up_4",
)


class ConditionExport(nn.Module):
    def __init__(self, model: ConditionNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.model(condition)
        return tuple(features["down_conditions"] + features["up_conditions"])


class DenoiserExport(nn.Module):
    def __init__(self, model: DiffusionUNetCrossAttention) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, state: torch.Tensor, time: torch.Tensor, *features: torch.Tensor
    ) -> torch.Tensor:
        conditions = {
            "down_conditions": list(features[:6]),
            "up_conditions": list(features[6:]),
        }
        return self.model(state, conditions, time)


class ExportMultiheadAttention(nn.Module):
    """ONNX-opset-13-friendly equivalent for the repository's MHA usage."""

    def __init__(self, source: nn.MultiheadAttention) -> None:
        super().__init__()
        if not source.batch_first or source.bias_k is not None or source.bias_v is not None:
            raise ValueError("only batch-first MHA without bias_k/bias_v is exportable")
        if source.add_zero_attn or source.kdim != source.embed_dim or source.vdim != source.embed_dim:
            raise ValueError("unsupported MultiheadAttention configuration")
        self.embed_dim = source.embed_dim
        self.num_heads = source.num_heads
        self.head_dim = source.embed_dim // source.num_heads
        self.in_proj_weight = source.in_proj_weight
        self.in_proj_bias = source.in_proj_bias
        self.out_proj = source.out_proj

    def _project(self, value: torch.Tensor, offset: int) -> torch.Tensor:
        stop = offset + self.embed_dim
        bias = None if self.in_proj_bias is None else self.in_proj_bias[offset:stop]
        projected = F.linear(value, self.in_proj_weight[offset:stop], bias)
        batch, length, _ = projected.shape
        return projected.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._project(query, 0)
        k = self._project(key, self.embed_dim)
        v = self._project(value, self.embed_dim * 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attended = torch.matmul(torch.softmax(scores, dim=-1), v)
        batch, _, length, _ = attended.shape
        merged = attended.transpose(1, 2).reshape(batch, length, self.embed_dim)
        output = self.out_proj(merged)
        return output, output.new_empty((0,))


def _replace_attention_for_export(module: nn.Module) -> None:
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, ExportMultiheadAttention(child))
        else:
            _replace_attention_for_export(child)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("rcfm", "rddm"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=OPSET)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--attention-heads", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_with_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError(f"checkpoint contains no state keys with prefix {prefix!r}")
    return selected


def _checkpoint_contract(
    payload: Mapping[str, Any], model_family: str
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    config = payload.get("config", {})
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("checkpoint config must be a mapping")
    if model_family == "rcfm":
        if "model_state" not in payload or "condition_state" not in payload:
            raise ValueError("RCFM checkpoint requires model_state and condition_state")
        model_state = payload["model_state"]
        flow_state = (
            _state_with_prefix(model_state, "flow_model.")
            if any(key.startswith("flow_model.") for key in model_state)
            else dict(model_state)
        )
        states = {"condition": dict(payload["condition_state"]), "flow": flow_state}
    else:
        required = ("rddm_state", "condition_1_state", "condition_2_state")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError("RDDM checkpoint missing: " + ", ".join(missing))
        rddm_state = payload["rddm_state"]
        states = {
            "condition_1": dict(payload["condition_1_state"]),
            "condition_2": dict(payload["condition_2_state"]),
            "region": _state_with_prefix(rddm_state, "region_model."),
            "epsilon": _state_with_prefix(rddm_state, "eps_model."),
        }
    for name, state in states.items():
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"{name} state must be a nonempty mapping")
    return states, dict(config or {})


def _channels(state: Mapping[str, torch.Tensor]) -> int:
    try:
        inputs = int(state["inc_x.double_conv.0.weight"].shape[1])
        outputs = int(state["outc_x.conv.weight"].shape[0])
    except KeyError as error:
        raise ValueError(f"denoiser state lacks channel-defining tensor: {error}") from error
    if inputs != outputs:
        raise ValueError(f"denoiser input/output channels differ: {inputs} != {outputs}")
    return inputs


def _attention_heads(config: Mapping[str, Any], explicit: int | None) -> int:
    value = explicit if explicit is not None else int(config.get("attention_heads", 8))
    if value <= 0 or any(channel % value for channel in (32, 64, 128, 256, 512, 1024)):
        raise ValueError("attention heads must divide every attention embedding dimension")
    return value


def _export(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    module.eval()
    with torch.inference_mode():
        torch.onnx.export(
            module, inputs, str(path), export_params=True, opset_version=OPSET,
            do_constant_folding=True, input_names=input_names, output_names=output_names,
            dynamic_axes=None,
        )
    graph = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(graph)
    if graph.opset_import[0].version != OPSET:
        raise RuntimeError(f"unexpected exported opset {graph.opset_import[0].version}")


def _condition_shapes(features: tuple[torch.Tensor, ...]) -> dict[str, list[int]]:
    return {name: list(tensor.shape) for name, tensor in zip(CONDITION_NAMES, features)}


def main() -> None:
    args = _parse_args()
    if args.opset != OPSET:
        raise ValueError("JetPack 4.6 / TensorRT 8.0 export is frozen to ONNX opset 13")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError("output directory is nonempty; use a new path or --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    payload = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must be a mapping, not a bare state dictionary")
    states, config = _checkpoint_contract(payload, args.model)
    heads = _attention_heads(config, args.attention_heads)
    denoiser_key = "flow" if args.model == "rcfm" else "region"
    channels = _channels(states[denoiser_key])
    condition = torch.randn(1, 1, LENGTH, dtype=torch.float32)
    state = torch.randn(1, channels, LENGTH, dtype=torch.float32)
    time = torch.tensor([0.5], dtype=torch.float32)
    files: dict[str, dict[str, Any]] = {}
    golden: dict[str, np.ndarray] = {
        "condition": condition.numpy(), "state": state.numpy(), "time": time.numpy(),
    }

    condition_keys = ["condition"] if args.model == "rcfm" else ["condition_1", "condition_2"]
    condition_features: dict[str, tuple[torch.Tensor, ...]] = {}
    for key in condition_keys:
        network = ConditionNet().cpu()
        network.load_state_dict(states[key], strict=True)
        wrapper = ConditionExport(network)
        with torch.inference_mode():
            features = wrapper(condition)
        condition_features[key] = features
        path = args.output / f"{args.model}_{key}.onnx"
        _export(wrapper, (condition,), path, ["condition"], list(CONDITION_NAMES))
        files[path.name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for name, tensor in zip(CONDITION_NAMES, features):
            golden[f"{key}_{name}"] = tensor.numpy()

    denoiser_keys = ["flow"] if args.model == "rcfm" else ["region", "epsilon"]
    for key in denoiser_keys:
        network = DiffusionUNetCrossAttention(LENGTH, channels, "cpu", num_heads=heads).cpu()
        network.load_state_dict(states[key], strict=True)
        wrapper = DenoiserExport(network)
        feature_key = "condition" if args.model == "rcfm" else ("condition_2" if key == "region" else "condition_1")
        features = condition_features[feature_key]
        with torch.inference_mode():
            reference_output = wrapper(state, time, *features)
        _replace_attention_for_export(network)
        with torch.inference_mode():
            output = wrapper(state, time, *features)
        if not torch.allclose(reference_output, output, rtol=2e-5, atol=1e-5):
            error = float(torch.max(torch.abs(reference_output - output)))
            raise RuntimeError(f"export attention rewrite changed {key} output; max error={error}")
        path = args.output / f"{args.model}_{key}.onnx"
        _export(wrapper, (state, time, *features), path, ["state", "time", *CONDITION_NAMES], ["output"])
        files[path.name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        golden[f"{key}_output"] = output.numpy()

    golden_path = args.output / "golden_vectors.npz"
    np.savez_compressed(golden_path, **golden)
    files[golden_path.name] = {"sha256": _sha256(golden_path), "bytes": golden_path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "deployment_target": {
            "device": "Jetson Nano 4GB", "jetpack": "4.6", "l4t": "32.6.1",
            "cuda": "10.2", "cudnn": "8.2.1.32", "tensorrt": "8.0.1",
        },
        "model": args.model, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint), "onnx_opset": OPSET,
        "static_batch": 1, "length": LENGTH, "condition_channels": 1,
        "output_channels": channels, "attention_heads": heads,
        "condition_outputs": _condition_shapes(condition_features[condition_keys[0]]),
        "sampling_loop_in_onnx": False,
        "files": files,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
