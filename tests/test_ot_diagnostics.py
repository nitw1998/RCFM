import numpy as np
import pytest
import torch

from optimal_transport import OTPlanSampler
from src.rcfm.ot_diagnostics import compute_ot_diagnostics


def test_nonfinite_plan_is_not_silently_used(monkeypatch):
    sampler = OTPlanSampler(method="sinkhorn", warn=False)
    monkeypatch.setattr(
        sampler,
        "ot_fn",
        lambda _a, _b, _cost: np.full((2, 2), np.nan),
    )
    source = torch.zeros(2, 1)
    target = torch.ones(2, 1)

    with pytest.raises(FloatingPointError, match="nonfinite"):
        sampler.get_map(source, target, strict=True)

    plan, diagnostics = sampler.get_map(
        source,
        target,
        strict=False,
        return_diagnostics=True,
    )
    np.testing.assert_allclose(plan, np.full((2, 2), 0.25))
    assert diagnostics["fallback_count"] == 1
    assert diagnostics["nonfinite_plan_count"] == 1


def test_plan_cost_marginal_and_diversity_diagnostics_are_exact():
    plan = np.array([[0.5, 0.0], [0.0, 0.5]])
    costs = np.array([[4.0, 1.0], [1.0, 4.0]])
    source_indices = np.array([0, 1])
    target_indices = np.array([1, 0])
    scalars, histograms = compute_ot_diagnostics(
        plan=plan,
        cost_matrix=costs,
        source_indices=source_indices,
        target_indices=target_indices,
        intended_source_marginal=np.array([0.5, 0.5]),
        intended_target_marginal=np.array([0.5, 0.5]),
        regularization=0.5,
        method="sinkhorn",
    )

    assert scalars["ot/cost_random_pairing"] == 4.0
    assert scalars["ot/cost_selected_pairing"] == 1.0
    assert scalars["ot/cost_reduction"] == 3.0
    assert scalars["ot/cost_reduction_ratio"] == 0.75
    assert scalars["ot/row_marginal_error"] == 0.0
    assert scalars["ot/column_marginal_error"] == 0.0
    assert scalars["ot/unique_source_count"] == 2.0
    assert scalars["ot/target_duplicate_fraction"] == 0.0
    np.testing.assert_array_equal(histograms["ot/transport_costs"], [1.0, 1.0])
    assert all(np.isfinite(value) for value in scalars.values())


def test_duplicate_rates_refer_to_sampled_indices():
    scalars, _ = compute_ot_diagnostics(
        plan=np.full((3, 3), 1.0 / 9.0),
        cost_matrix=np.ones((3, 3)),
        source_indices=np.array([0, 0, 1]),
        target_indices=np.array([2, 2, 2]),
        intended_source_marginal=np.full(3, 1.0 / 3.0),
        intended_target_marginal=np.full(3, 1.0 / 3.0),
        regularization=0.05,
        method="sinkhorn",
    )

    assert scalars["ot/source_duplicate_fraction"] == pytest.approx(1.0 / 3.0)
    assert scalars["ot/target_duplicate_fraction"] == pytest.approx(2.0 / 3.0)


def test_assignment_sampling_is_one_to_one():
    sampler = OTPlanSampler(method="exact")
    plan = np.array(
        [
            [0.0, 0.3, 0.0],
            [0.3, 0.0, 0.0],
            [0.0, 0.0, 0.4],
        ]
    )
    source_indices, target_indices = sampler.sample_map(
        plan,
        batch_size=3,
        strategy="assignment",
    )

    assert len(np.unique(source_indices)) == 3
    assert len(np.unique(target_indices)) == 3
