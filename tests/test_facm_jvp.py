from __future__ import annotations

import torch
from torch import nn

from src.rcfm.one_step.facm_loss_1d import FACMLoss1D


class TinyConditionalFlow(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.time_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, conditions, time):
        condition = conditions["down_conditions"][0]
        return self.scale * x + self.time_scale * time[:, None, None] + condition


def conditions(value: torch.Tensor) -> dict[str, list[torch.Tensor]]:
    return {"down_conditions": [value], "up_conditions": [value]}


def test_facm_jvp_loss_has_finite_student_gradients() -> None:
    student = TinyConditionalFlow(0.3)
    teacher = TinyConditionalFlow(0.5)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    target = torch.linspace(-1, 1, 32).reshape(2, 1, 16)
    noise = -target
    condition = torch.full_like(target, 0.1, requires_grad=True)
    result = FACMLoss1D()(
        student_flow=student,
        teacher_flow=teacher,
        target=target,
        student_conditions=conditions(condition),
        teacher_conditions=conditions(condition),
        noise=noise,
        time=torch.tensor([0.25, 0.75]),
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert all(parameter.grad is not None for parameter in student.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in student.parameters())
    assert condition.grad is not None and torch.isfinite(condition.grad).all()
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert result["facm/shortcut_jvp_norm"] > 0


def test_facm_anchor_uses_expanded_time_but_teacher_uses_flow_time() -> None:
    calls: dict[str, torch.Tensor] = {}

    def teacher(x, condition, time):
        calls["teacher"] = time.detach().clone()
        return torch.ones_like(x)

    def student(x, condition, time):
        key = "anchor" if torch.all(time >= 1) else "shortcut"
        calls[key] = time.detach().clone()
        return x * 0.1 + time[:, None, None]

    target = torch.zeros(2, 1, 8)
    condition = conditions(torch.zeros_like(target))
    time = torch.tensor([0.2, 0.8])
    FACMLoss1D()(
        student_flow=student,
        teacher_flow=teacher,
        target=target,
        student_conditions=condition,
        teacher_conditions=condition,
        noise=torch.ones_like(target),
        time=time,
    )
    assert torch.equal(calls["teacher"], time)
    assert torch.allclose(calls["anchor"], 2 - time)
    assert torch.equal(calls["shortcut"], time)


def test_facm_jvp_supports_eleven_target_leads() -> None:
    student = TinyConditionalFlow(0.3)
    teacher = TinyConditionalFlow(0.5)
    target = torch.linspace(-1, 1, 2 * 11 * 16).reshape(2, 11, 16)
    source_features = torch.full((2, 1, 16), 0.1)
    result = FACMLoss1D()(
        student_flow=student,
        teacher_flow=teacher,
        target=target,
        student_conditions=conditions(source_features),
        teacher_conditions=conditions(source_features),
        noise=-target,
        time=torch.tensor([0.25, 0.75]),
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert all(parameter.grad is not None for parameter in student.parameters())
