from __future__ import annotations

import inspect

import torch

from src.rcfm.one_step.facm_sampler import facm_one_step_sample


class CountingCondition:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, source):
        self.calls += 1
        return {"down_conditions": [source], "up_conditions": [source]}


class CountingStudent:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, noise, conditions, time):
        self.calls += 1
        return 0.5 * conditions["down_conditions"][0] + time[:, None, None]


def test_one_step_sampler_is_exactly_one_student_call_and_source_conditional() -> None:
    encoder = CountingCondition()
    student = CountingStudent()
    noise = torch.zeros(2, 1, 16)
    source_a = torch.ones_like(noise)
    source_b = -source_a
    output_a = facm_one_step_sample(
        student_flow=student,
        condition_encoder=encoder,
        source_condition=source_a,
        initial_noise=noise,
    )
    output_b = facm_one_step_sample(
        student_flow=student,
        condition_encoder=encoder,
        source_condition=source_b,
        initial_noise=noise,
    )
    assert student.calls == 2
    assert encoder.calls == 2
    assert not torch.equal(output_a, output_b)
    assert torch.equal(output_a, torch.full_like(noise, 0.5))


def test_one_step_public_interface_cannot_accept_target_or_teacher() -> None:
    parameters = inspect.signature(facm_one_step_sample).parameters
    assert "target" not in parameters
    assert "teacher" not in parameters
    assert "region_mask" not in parameters
    assert "ot_solver" not in parameters
