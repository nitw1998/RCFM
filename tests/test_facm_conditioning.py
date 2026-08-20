from __future__ import annotations

import torch

from src.rcfm.one_step.facm_time_conditioning import expanded_fm_time, sample_facm_time


def test_facm_time_distribution_and_expanded_interval() -> None:
    generator = torch.Generator().manual_seed(31)
    time = sample_facm_time(
        1024,
        device="cpu",
        dtype=torch.float32,
        generator=generator,
    )
    assert time.shape == (1024,)
    assert torch.all((time >= 0) & (time <= 1))
    anchor_time = expanded_fm_time(time)
    assert torch.all((anchor_time >= 1) & (anchor_time <= 2))
    assert torch.allclose(time + anchor_time, torch.full_like(time, 2.0))
