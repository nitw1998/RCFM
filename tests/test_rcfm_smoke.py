import torch
import torch.nn as nn

from rcfm import RegionAwareConditionalFlowMatching


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, x, conditions, t):
        del conditions
        return self.net(x) + t.view(-1, 1, 1) * 0.0


def test_rcfm_forward_backward_and_sample():
    torch.manual_seed(7)
    model = RegionAwareConditionalFlowMatching(
        flow_model=TinyFlow(),
        flow_matcher_type="conditional",
        region_weight=2.0,
        use_minibatch_ot=False,
    )
    target = torch.randn(3, 1, 16)
    mask = torch.zeros_like(target)
    mask[:, :, 4:8] = 1.0
    conditions = {
        "down_conditions": [torch.randn(3, 1, 16)],
        "up_conditions": [torch.randn(3, 1, 16)],
    }

    output = model(target=target, conditions=conditions, region_mask=mask)
    output["loss"].backward()
    sample = model.sample(conditions=conditions, shape=target.shape, steps=2)

    assert torch.isfinite(output["loss"])
    assert sample.shape == target.shape
    assert model.flow_model.net.weight.grad is not None
