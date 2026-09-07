import numpy as np
import pytest
import torch

from scripts.audit_diagmask_faithfulness import _summarize_analysis_rows
from src.rcfm.evaluation.diagnostic_models import (
    load_torchscript_adapter,
    predict_record,
)

from src.rcfm.evaluation.diagnostic_transfer import (
    aggregate_by_group,
    as_time_leads,
    binary_metrics,
    paired_bootstrap_mean,
    paired_stratified_bootstrap_auc_difference,
    reconstruct_twelve_leads,
    stratified_bootstrap_auc,
)


def test_single_lead_adapter_repeats_without_changing_time_axis():
    values = np.arange(20, dtype=np.float32).reshape(2, 10)
    output = as_time_leads(values)
    assert output.shape == (2, 10, 12)
    assert np.array_equal(output[:, :, 0], values)
    assert np.array_equal(output[:, :, -1], values)


def test_group_aggregation_uses_mean_logits_and_rejects_label_conflicts():
    logits, labels, indices = aggregate_by_group(
        np.array([-2.0, 0.0, 1.0, 3.0]),
        np.array([0, 0, 1, 1]),
        np.array(["a", "a", "b", "b"]),
    )
    assert np.allclose(logits, [-1.0, 2.0])
    assert labels.tolist() == [False, True]
    assert indices.tolist() == [0, 1]
    with pytest.raises(ValueError, match="conflicting"):
        aggregate_by_group([0, 1], [0, 1], ["same", "same"])


def test_binary_metrics_and_bootstraps_are_finite_and_paired():
    labels = np.array([0, 0, 0, 1, 1, 1])
    reference = np.array([-2.0, -1.0, 0.5, -0.5, 1.0, 2.0])
    candidate = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    metrics = binary_metrics(labels, candidate)
    assert metrics["auroc"] == 1.0
    interval = stratified_bootstrap_auc(labels, candidate, replicates=20)
    assert interval["auroc_95_ci"] == [1.0, 1.0]
    difference = paired_stratified_bootstrap_auc_difference(
        labels, candidate, reference, replicates=20
    )
    assert difference["auroc_difference"] >= 0
    mean = paired_bootstrap_mean(np.array([1.0, 2.0, 3.0]), replicates=20)
    assert mean["mean"] == 2.0


def test_reconstruct_twelve_leads_preserves_condition_position():
    condition = np.full((2, 1, 8), 7.0, dtype=np.float32)
    generated = np.stack(
        [np.full((2, 8), value, dtype=np.float32) for value in range(11)], axis=1
    )
    output = reconstruct_twelve_leads(condition, generated, condition_lead=1)
    assert output.shape == (2, 12, 8)
    assert np.all(output[:, 1] == 7.0)
    assert np.all(output[:, 0] == 0.0)
    assert np.all(output[:, 11] == 10.0)


def test_torchscript_adapter_runs_record_crop_contract(tmp_path):
    class TinyDiagnostic(torch.nn.Module):
        def forward(self, values):
            mean = values.mean(dim=(1, 2))
            return torch.stack((mean, -mean), dim=1)

    checkpoint = tmp_path / "tiny_diagnostic.ts"
    class_names = tmp_path / "classes.txt"
    traced = torch.jit.trace(TinyDiagnostic(), torch.ones(1, 12, 8))
    torch.jit.save(traced, checkpoint)
    class_names.write_text("AFIB\nOTHER\n", encoding="utf-8")
    adapter = load_torchscript_adapter(
        checkpoint=checkpoint,
        class_names_path=class_names,
        device=torch.device("cpu"),
        input_rate_hz=8,
        crop_samples=8,
        crop_stride=4,
        normalization="none",
    )
    logits = predict_record(
        adapter,
        np.ones((16, 12), dtype=np.float32),
        source_rate_hz=8,
        aggregation="mean_logit",
    )
    assert logits.tolist() == [1.0, -1.0]


def test_faithfulness_summary_keeps_af_positive_as_separate_cohort():
    rows = []
    for index, label in enumerate((1, 1, 0, 0)):
        rows.append(
            {
                "label": label,
                "cam_occupancy": 0.2,
                "cam_degenerate": False,
                "amplitude_cam_spearman": 0.9,
                "noise_cam_spearman": 0.8,
                "deletion_advantage_top10": float(index + 1),
                "insertion_advantage_top10": float(index + 2),
            }
        )
    summary = _summarize_analysis_rows(
        rows, bootstrap_seed=31, bootstrap_replicates=20
    )
    assert summary["AF_positive"]["observations"] == 2
    assert summary["non_AF"]["observations"] == 2
    assert summary["overall"]["observations"] == 4
    assert (
        summary["AF_positive"]["paired_estimands"]["deletion_advantage_top10"][
            "mean"
        ]
        == 1.5
    )
