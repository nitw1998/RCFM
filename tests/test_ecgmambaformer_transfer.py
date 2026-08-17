from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.rcfm.interpretability.ecgmambaformer_compat import (
    diagnostic_class_names,
    ecgmamba_task_head_gradcam,
    record_global_zscore,
)


def test_record_global_zscore_matches_per_record_cross_lead_rule():
    values = np.arange(60, dtype=np.float32).reshape(10, 6)
    normalized, mean, scale = record_global_zscore(values)

    assert mean == np.mean(values)
    assert scale == np.std(values)
    assert abs(float(normalized.mean())) < 1e-6
    assert abs(float(normalized.std()) - 1.0) < 1e-6


def test_diagnostic_class_names_are_sorted_and_filter_non_diagnostic(tmp_path: Path):
    path = tmp_path / "scp.csv"
    path.write_text(
        ",description,diagnostic,form,rhythm\n"
        "NORM,normal,1.0,,\n"
        "AFIB,atrial fibrillation,,,1.0\n"
        "1AVB,first degree,1.0,,\n",
        encoding="utf-8",
    )

    assert diagnostic_class_names(path) == ["1AVB", "NORM"]


class _TinyTaskHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv1d(2, 4, 3, padding=1)
        self.diag = nn.Linear(4, 3)
        self.semantic = nn.Conv1d(4, 4, 1)

    def diagnostic_logits_from_features(self, features):
        return self.diag(features.mean(dim=-1))

    def semantic_logits_from_features(self, features):
        logits = self.semantic(features)
        logits[:, 1, :4] += 3.0
        logits[:, 2, 4:8] += 3.0
        logits[:, 3, 8:] += 3.0
        return logits


def test_task_head_gradcam_uses_diag_and_dense_semantic_scores():
    torch.manual_seed(3)
    model = _TinyTaskHeadModel().eval()
    inputs = torch.randn(1, 2, 12)

    diag_cam, diag_logits, diag_target = ecgmamba_task_head_gradcam(
        model, inputs, task="diag", diagnostic_target_indices=(0, 2)
    )
    semantic_cam, semantic_logits, semantic_target = ecgmamba_task_head_gradcam(
        model, inputs, task="semantic"
    )

    assert diag_cam.shape == semantic_cam.shape == (12,)
    assert diag_logits.shape == (3,)
    assert semantic_logits.shape == (4, 12)
    assert np.isfinite(diag_cam).all() and np.isfinite(semantic_cam).all()
    assert diag_target["diagnostic_target_indices"] == [0, 2]
    assert semantic_target["semantic_target_classes"] == [1, 2, 3]
