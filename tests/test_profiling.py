import torch
import torch.nn as nn

from diffusion import RDDM
from src.rcfm.profiling import benchmark_callable, parameter_counts, profile_forward


class CountingDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, condition, time):
        del condition, time
        self.calls += 1
        return torch.zeros_like(x)


def test_conv_mac_schema_and_parameter_counts():
    model = nn.Conv1d(2, 3, kernel_size=3, bias=False)
    _, profile = profile_forward(model, torch.ones(2, 2, 10))

    assert profile["macs"] == 2 * 3 * 8 * (2 * 3)
    assert profile["flops"] == 2 * profile["macs"]
    assert profile["flop_convention"] == "1 MAC = 2 FLOPs"
    assert profile["macs_by_operator"] == {"conv1d": profile["macs"]}
    assert parameter_counts([model]) == {"total": 18, "trainable": 18}


def test_rddm_executes_both_subnetworks_each_sampling_step():
    epsilon = CountingDenoiser()
    region = CountingDenoiser()
    model = RDDM(eps_model=epsilon, region_model=region, betas=(1e-4, 0.2), n_T=3)
    conditions = {
        "down_conditions": [torch.zeros(2, 1, 8)],
        "up_conditions": [torch.zeros(2, 1, 8)],
    }

    output = model(cond1=conditions, cond2=conditions, mode="sample", window_size=8)

    assert output.shape == (2, 1, 8)
    assert region.calls == 3
    assert epsilon.calls == 3


def test_latency_schema_is_deterministic_on_cpu():
    result = benchmark_callable(lambda: torch.ones(2) + 1, torch.device("cpu"), warmup=1, iterations=2)

    assert result["warmup_iterations"] == 1
    assert result["timed_iterations"] == 2
    assert result["synchronization"] == "not_required_cpu"
    assert result["peak_memory_bytes"] is None
    assert result["latency_ms"]["min"] >= 0
