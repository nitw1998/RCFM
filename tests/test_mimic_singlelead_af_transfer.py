import numpy as np

from scripts.evaluate_singlelead_af_transfer import aggregate_records
from scripts.audit_singlelead_af_faithfulness import batch_gradcam, selected_mask
from scripts.prepare_mimic_leadii_physical_windows import ecg_channel_and_lead


class Header:
    sig_name = ["PPG, fingertip", "ECG, monitor, lead II", "resp"]


def test_ecg_lead_parser_requires_explicit_header_label():
    assert ecg_channel_and_lead(Header()) == (1, "II")


def test_record_aggregation_uses_mean_logit_and_one_label():
    names, logits, labels = aggregate_records(
        np.array([1.0, 3.0, -2.0, -4.0]),
        np.array([1, 1, 0, 0]),
        np.array(["af", "af", "non", "non"]),
    )
    assert names.tolist() == ["af", "non"]
    assert logits.tolist() == [2.0, -3.0]
    assert labels.tolist() == [1, 0]


def test_batched_gradcam_and_exact_top_fraction():
    import torch
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 2, 3, padding=1)
            self.head = nn.Linear(2, 1)

        def forward(self, x):
            return self.head(self.conv(x).relu().mean(-1)).squeeze(-1)

    model = Tiny().eval()
    logits, masks, degenerate = batch_gradcam(model, model.conv, torch.randn(3, 1, 20))
    selected = selected_mask(masks.numpy(), 0.2)
    assert logits.shape == (3,)
    assert masks.shape == (3, 20)
    assert degenerate.shape == (3,)
    assert np.all(selected.sum(axis=1) == 4)
