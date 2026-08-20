from __future__ import annotations

import pytest
import torch

from src.rcfm.one_step.facm_loss_1d import FACMLoss1D


def test_facm_rejects_image_rank_or_mismatched_noise() -> None:
    objective = FACMLoss1D()
    condition = {"down_conditions": [torch.zeros(2, 1, 8)], "up_conditions": [torch.zeros(2, 1, 8)]}
    with pytest.raises(ValueError, match="shape"):
        objective(
            student_flow=lambda x, c, t: x,
            teacher_flow=lambda x, c, t: x,
            target=torch.zeros(2, 1, 2, 8),
            student_conditions=condition,
            teacher_conditions=condition,
        )
    with pytest.raises(ValueError, match="noise"):
        objective(
            student_flow=lambda x, c, t: x,
            teacher_flow=lambda x, c, t: x,
            target=torch.zeros(2, 1, 8),
            student_conditions=condition,
            teacher_conditions=condition,
            noise=torch.zeros(2, 1, 7),
        )
