"""Small runtime guards shared by instrumented training entries."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def gradients_are_finite(parameters: Iterable[torch.nn.Parameter]) -> bool:
    """Return whether every materialized gradient contains finite values."""

    checks = [
        torch.isfinite(parameter.grad).all()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return not checks or bool(torch.stack(checks).all())


def exception_summary(error: BaseException) -> str:
    """Return a nonempty single-line exception description."""

    lines = str(error).splitlines()
    return lines[0] if lines else type(error).__name__
