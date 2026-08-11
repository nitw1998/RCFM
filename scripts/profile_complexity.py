"""Profile executable RCFM and RDDM neural operators and parameters."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import ConditionNet, DiffusionUNetCrossAttention
from src.rcfm.profiling import parameter_counts, profile_forward


def _component(profile: dict[str, object], batch_size: int) -> dict[str, object]:
    macs = int(profile["macs"])
    return {
        **profile,
        "macs_per_sample": macs / batch_size,
        "flops_per_sample": 2 * macs / batch_size,
    }


def _inputs(batch_size: int, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(batch_size, 1, length, device=device),
        torch.randn(batch_size, 1, length, device=device),
    )


def profile_rcfm(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    condition_net = ConditionNet().to(device)
    flow_model = DiffusionUNetCrossAttention(
        args.length,
        args.channels,
        str(device),
        num_heads=args.attention_heads,
    ).to(device)
    condition_signal, state = _inputs(args.batch_size, args.length, device)
    time = torch.full((args.batch_size,), 0.5, device=device)
    conditions, condition_profile = profile_forward(condition_net, condition_signal)
    _, flow_profile = profile_forward(flow_model, state, conditions, time)
    condition_component = _component(condition_profile, args.batch_size)
    flow_component = _component(flow_profile, args.batch_size)
    total_macs = condition_component["macs_per_sample"] + args.steps * flow_component["macs_per_sample"]
    return {
        "model": "RCFM",
        "parameters": parameter_counts([condition_net, flow_model]),
        "component_parameters": {
            "condition_encoder": parameter_counts([condition_net]),
            "flow_model": parameter_counts([flow_model]),
        },
        "components": {
            "condition_encoder_once": condition_component,
            "flow_model_per_nfe": flow_component,
        },
        "sampling_steps": args.steps,
        "neural_function_evaluations": args.steps,
        "condition_encoder_evaluations": 1,
        "actual_component_calls_per_sample": {
            "condition_encoder": 1,
            "flow_model": args.steps,
        },
        "macs_per_neural_function_evaluation": flow_component["macs_per_sample"],
        "flops_per_neural_function_evaluation": flow_component["flops_per_sample"],
        "total_macs_per_generated_sample_including_condition": total_macs,
        "total_flops_per_generated_sample_including_condition": 2 * total_macs,
    }


def profile_rddm(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    condition_net_1 = ConditionNet().to(device)
    condition_net_2 = ConditionNet().to(device)
    region_model = DiffusionUNetCrossAttention(
        args.length,
        args.channels,
        str(device),
        num_heads=args.attention_heads,
    ).to(device)
    epsilon_model = DiffusionUNetCrossAttention(
        args.length,
        args.channels,
        str(device),
        num_heads=args.attention_heads,
    ).to(device)
    condition_signal, state = _inputs(args.batch_size, args.length, device)
    time = torch.full((args.batch_size,), 0.5, device=device)
    conditions_1, condition_1_profile = profile_forward(condition_net_1, condition_signal)
    conditions_2, condition_2_profile = profile_forward(condition_net_2, condition_signal)
    region_output, region_profile = profile_forward(region_model, state, conditions_2, time)
    _, epsilon_profile = profile_forward(epsilon_model, region_output, conditions_1, time)
    condition_1 = _component(condition_1_profile, args.batch_size)
    condition_2 = _component(condition_2_profile, args.batch_size)
    region = _component(region_profile, args.batch_size)
    epsilon = _component(epsilon_profile, args.batch_size)
    condition_macs = condition_1["macs_per_sample"] + condition_2["macs_per_sample"]
    step_macs = region["macs_per_sample"] + epsilon["macs_per_sample"]
    total_macs = condition_macs + args.steps * step_macs
    return {
        "model": "RDDM",
        "parameters": parameter_counts([condition_net_1, condition_net_2, region_model, epsilon_model]),
        "component_parameters": {
            "condition_encoder_1": parameter_counts([condition_net_1]),
            "condition_encoder_2": parameter_counts([condition_net_2]),
            "region_model": parameter_counts([region_model]),
            "epsilon_model": parameter_counts([epsilon_model]),
        },
        "components": {
            "condition_encoder_1_once": condition_1,
            "condition_encoder_2_once": condition_2,
            "region_model_per_step": region,
            "epsilon_model_per_step": epsilon,
        },
        "sampling_steps": args.steps,
        "neural_function_evaluations": 2 * args.steps,
        "condition_encoder_evaluations": 2,
        "actual_component_calls_per_sample": {
            "condition_encoder_1": 1,
            "condition_encoder_2": 1,
            "region_model": args.steps,
            "epsilon_model": args.steps,
        },
        "macs_per_neural_function_evaluation": (region["macs_per_sample"] + epsilon["macs_per_sample"]) / 2,
        "flops_per_neural_function_evaluation": (region["flops_per_sample"] + epsilon["flops_per_sample"]) / 2,
        "macs_per_sampling_step_excluding_condition": step_macs,
        "flops_per_sampling_step_excluding_condition": 2 * step_macs,
        "total_macs_per_generated_sample_including_condition": total_macs,
        "total_flops_per_generated_sample_including_condition": 2 * total_macs,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["rcfm", "rddm", "both"], default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.length <= 0:
        raise ValueError("steps, batch_size, and length must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
    profiles = []
    if args.model in {"rcfm", "both"}:
        profiles.append(profile_rcfm(args, device))
        gc.collect()
    if args.model in {"rddm", "both"}:
        profiles.append(profile_rddm(args, device))
        gc.collect()
    result = {
        "schema_version": 1,
        "backend": "PyTorch forward hooks for executed Conv1d, ConvTranspose1d, Linear, and MultiheadAttention operators",
        "scope_note": "Normalization, activation, softmax, indexing, and elementwise operations are not included in MAC totals.",
        "flop_convention": "1 MAC = 2 FLOPs",
        "input": {
            "batch_size": args.batch_size,
            "length": args.length,
            "input_channels": args.channels,
            "output_channels": args.channels,
            "precision": "float32",
            "device": str(device),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    for profile in profiles:
        print(
            f"{profile['model']}: params={profile['parameters']['total']} "
            f"NFE={profile['neural_function_evaluations']} "
            f"total_MACs={profile['total_macs_per_generated_sample_including_condition']:.0f}"
        )


if __name__ == "__main__":
    main()
