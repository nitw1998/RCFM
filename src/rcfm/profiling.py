"""Executable-graph parameter, MAC, call-count, and latency helpers."""

from __future__ import annotations

import statistics
import time
from collections import Counter
from contextlib import AbstractContextManager
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn


def parameter_counts(modules: Iterable[nn.Module]) -> dict[str, int]:
    parameters = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    return {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
    }


class OperationCounter(AbstractContextManager):
    """Count multiply-accumulates executed by common neural operators."""

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.macs_by_operator: Counter[str] = Counter()
        self.calls_by_module: Counter[str] = Counter()
        self._handles: list[Any] = []

    @staticmethod
    def _conv1d_macs(layer: nn.Conv1d, output: torch.Tensor) -> int:
        output_elements = output.numel()
        kernel_multiplications = (layer.in_channels // layer.groups) * layer.kernel_size[0]
        return output_elements * kernel_multiplications

    @staticmethod
    def _conv_transpose1d_macs(layer: nn.ConvTranspose1d, inputs: tuple[torch.Tensor, ...]) -> int:
        source = inputs[0]
        return (
            source.shape[0]
            * source.shape[-1]
            * layer.in_channels
            * (layer.out_channels // layer.groups)
            * layer.kernel_size[0]
        )

    @staticmethod
    def _linear_macs(layer: nn.Linear, output: torch.Tensor) -> int:
        return output.numel() * layer.in_features

    @staticmethod
    def _attention_macs(layer: nn.MultiheadAttention, inputs: tuple[torch.Tensor, ...]) -> int:
        query, key, value = inputs[:3]
        if not layer.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        batch, query_length, embed_dim = query.shape
        key_length = key.shape[1]
        value_length = value.shape[1]
        projections = batch * (query_length + key_length + value_length) * embed_dim * embed_dim
        attention_products = 2 * batch * query_length * key_length * embed_dim
        output_projection = batch * query_length * embed_dim * embed_dim
        return projections + attention_products + output_projection

    def _hook(self, name: str, layer: nn.Module) -> Callable[..., None]:
        def record(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            self.calls_by_module[name] += 1
            if isinstance(module, nn.Conv1d):
                self.macs_by_operator["conv1d"] += self._conv1d_macs(module, output)
            elif isinstance(module, nn.ConvTranspose1d):
                self.macs_by_operator["conv_transpose1d"] += self._conv_transpose1d_macs(module, inputs)
            elif isinstance(module, nn.Linear):
                self.macs_by_operator["linear"] += self._linear_macs(module, output)
            elif isinstance(module, nn.MultiheadAttention):
                self.macs_by_operator["multihead_attention"] += self._attention_macs(module, inputs)

        return record

    def __enter__(self) -> "OperationCounter":
        for name, layer in self.module.named_modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear, nn.MultiheadAttention)):
                self._handles.append(layer.register_forward_hook(self._hook(name, layer)))
        return self

    def __exit__(self, *exc_info: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def result(self) -> dict[str, object]:
        total = int(sum(self.macs_by_operator.values()))
        return {
            "macs": total,
            "flops": 2 * total,
            "flop_convention": "1 MAC = 2 FLOPs",
            "counted_operators": sorted(self.macs_by_operator),
            "macs_by_operator": dict(self.macs_by_operator),
            "module_call_counts": dict(self.calls_by_module),
        }


def profile_forward(module: nn.Module, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, object]]:
    training = module.training
    module.eval()
    with torch.no_grad(), OperationCounter(module) as counter:
        output = module(*args, **kwargs)
    module.train(training)
    return output, counter.result()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(
    function: Callable[[], Any],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    for _ in range(warmup):
        function()
        synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings_ms = []
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter()
        function()
        synchronize(device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "synchronization": "torch.cuda.synchronize" if device.type == "cuda" else "not_required_cpu",
        "latency_ms": {
            "mean": statistics.fmean(timings_ms),
            "median": statistics.median(timings_ms),
            "std": statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
            "min": min(timings_ms),
            "max": max(timings_ms),
        },
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_memory_method": "torch.cuda.max_memory_allocated" if device.type == "cuda" else "unsupported_on_cpu",
    }
