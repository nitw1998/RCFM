"""Measure batch-1 end-to-end RCFM/RDDM sampling latency and peak memory."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion import RDDM
from model import ConditionNet, DiffusionUNetCrossAttention
from rcfm import RegionAwareConditionalFlowMatching
from src.rcfm.profiling import benchmark_callable, parameter_counts


class RCFMGenerator(nn.Module):
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        super().__init__()
        self.steps = args.steps
        self.length = args.length
        self.condition_encoder = ConditionNet()
        flow = DiffusionUNetCrossAttention(
            args.length, 1, str(device), num_heads=args.attention_heads
        )
        self.rcfm = RegionAwareConditionalFlowMatching(
            flow_model=flow,
            flow_matcher_type="conditional",
            sigma=0.0,
            use_minibatch_ot=False,
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        features = self.condition_encoder(condition)
        return self.rcfm.sample(
            conditions=features,
            shape=(condition.shape[0], 1, self.length),
            steps=self.steps,
            device=condition.device,
        )


class RDDMGenerator(nn.Module):
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        super().__init__()
        self.length = args.length
        self.condition_encoder_1 = ConditionNet()
        self.condition_encoder_2 = ConditionNet()
        self.rddm = RDDM(
            eps_model=DiffusionUNetCrossAttention(
                args.length, 1, str(device), num_heads=args.attention_heads
            ),
            region_model=DiffusionUNetCrossAttention(
                args.length, 1, str(device), num_heads=args.attention_heads
            ),
            betas=(1e-4, 0.2),
            n_T=args.steps,
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        condition_1 = self.condition_encoder_1(condition)
        condition_2 = self.condition_encoder_2(condition)
        return self.rddm(
            cond1=condition_1,
            cond2=condition_2,
            mode="sample",
            window_size=self.length,
        )


def _actual_component_calls(generator: nn.Module, condition: torch.Tensor) -> dict[str, int]:
    calls: Counter[str] = Counter()
    names = (
        "condition_encoder",
        "condition_encoder_1",
        "condition_encoder_2",
        "rcfm.flow_model",
        "rddm.region_model",
        "rddm.eps_model",
    )
    handles = []
    modules = dict(generator.named_modules())
    for name in names:
        if name in modules:
            handles.append(modules[name].register_forward_hook(lambda *_, key=name: calls.update([key])))
    with torch.no_grad():
        generator(condition)
    for handle in handles:
        handle.remove()
    return dict(calls)


def profile_generator(
    name: str,
    generator: nn.Module,
    condition: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    generator.eval()
    calls = _actual_component_calls(generator, condition)
    with torch.no_grad():
        timing = benchmark_callable(
            lambda: generator(condition),
            device=device,
            warmup=warmup,
            iterations=iterations,
        )
    return {
        "model": name,
        "parameters": parameter_counts([generator]),
        "actual_component_calls_per_generated_batch": calls,
        **timing,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["rcfm", "rddm", "both"], default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if min(args.steps, args.batch_size, args.length, args.iterations) <= 0 or args.warmup < 0:
        raise ValueError("steps, batch_size, length, and iterations must be positive; warmup must be nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
    condition = torch.randn(args.batch_size, 1, args.length, device=device)
    profiles = []
    if args.model in {"rcfm", "both"}:
        profiles.append(
            profile_generator(
                "RCFM",
                RCFMGenerator(args, device).to(device),
                condition,
                device,
                args.warmup,
                args.iterations,
            )
        )
    if args.model in {"rddm", "both"}:
        profiles.append(
            profile_generator(
                "RDDM",
                RDDMGenerator(args, device).to(device),
                condition,
                device,
                args.warmup,
                args.iterations,
            )
        )
    result = {
        "schema_version": 1,
        "input": {
            "batch_size": args.batch_size,
            "length": args.length,
            "input_channels": 1,
            "output_channels": 1,
            "precision": "float32",
            "steps": args.steps,
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
        print(f"{profile['model']}: median_latency_ms={profile['latency_ms']['median']:.3f}")


if __name__ == "__main__":
    main()
