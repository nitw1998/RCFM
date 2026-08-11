import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rcfm import RegionAwareConditionalFlowMatching
from src.rcfm.metrics.waveform import waveform_frechet_distance
from src.rcfm.validation import validate_epoch


class IdentityCondition(nn.Module):
    def forward(self, value):
        return {"down_conditions": [value], "up_conditions": [value]}


class ZeroFlow(nn.Module):
    def forward(self, x, conditions, t):
        del conditions, t
        return torch.zeros_like(x)


def test_waveform_fd_is_zero_for_identical_waveform_vectors():
    values = np.arange(24, dtype=np.float64).reshape(3, 1, 8)
    assert waveform_frechet_distance(values, values) < 1e-8


def test_validation_is_deterministic_and_preserves_training_rng():
    target = torch.arange(32, dtype=torch.float32).reshape(4, 1, 8) / 32.0
    condition = torch.flip(target, dims=(-1,))
    loader = DataLoader(TensorDataset(target, condition), batch_size=2, shuffle=False)
    model = RegionAwareConditionalFlowMatching(
        flow_model=ZeroFlow(),
        use_minibatch_ot=False,
    )
    condition_net = IdentityCondition()
    model.train()
    condition_net.train()
    torch.manual_seed(777)
    state_before = torch.random.get_rng_state()

    first = validate_epoch(model, condition_net, loader, torch.device("cpu"), 2, 1234)
    state_after = torch.random.get_rng_state()
    second = validate_epoch(model, condition_net, loader, torch.device("cpu"), 2, 1234)

    assert first == second
    torch.testing.assert_close(state_after, state_before)
    assert model.training and condition_net.training
    assert first["val/num_samples"] == 4.0
    assert first["val/num_subjects"] == 0.0
    assert first["val/subject_metadata_available"] == 0.0
    assert all(np.isfinite(value) for value in first.values())


def test_multilead_validation_reports_each_lead_and_averages_per_lead_fd():
    target = torch.arange(144, dtype=torch.float32).reshape(6, 3, 8) / 144.0
    condition = target[:, :1].flip(-1)
    loader = DataLoader(TensorDataset(target, condition), batch_size=3, shuffle=False)
    model = RegionAwareConditionalFlowMatching(
        flow_model=ZeroFlow(), use_minibatch_ot=False
    )

    metrics = validate_epoch(
        model,
        IdentityCondition(),
        loader,
        torch.device("cpu"),
        inference_steps=2,
        fixed_noise_seed=1234,
        target_leads=["I", "III", "aVR"],
    )

    lead_fds = []
    for lead in ("I", "III", "aVR"):
        assert f"val/lead/{lead}/rmse" in metrics
        assert f"val/lead/{lead}/mae" in metrics
        lead_fds.append(metrics[f"val/lead/{lead}/waveform_fd"])
    assert metrics["val/waveform_fd"] == pytest.approx(np.mean(lead_fds))
