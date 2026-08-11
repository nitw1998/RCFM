"""Scalar diagnostics for minibatch optimal-transport plans and samples."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _safe_ratio(numerator: float, denominator: float, epsilon: float = 1e-12) -> float:
    return float(numerator / max(abs(denominator), epsilon))


def compute_ot_diagnostics(
    plan: np.ndarray,
    cost_matrix: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    intended_source_marginal: np.ndarray,
    intended_target_marginal: np.ndarray,
    regularization: float,
    method: str,
    fallback_count: int = 0,
    nonfinite_plan_count: int = 0,
    solver_warning_count: int = 0,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Describe the plan and the exact sampled pairs passed to training."""

    plan = np.asarray(plan, dtype=np.float64)
    costs = np.asarray(cost_matrix, dtype=np.float64)
    source_indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    target_indices = np.asarray(target_indices, dtype=np.int64).reshape(-1)
    source_marginal = np.asarray(intended_source_marginal, dtype=np.float64).reshape(-1)
    target_marginal = np.asarray(intended_target_marginal, dtype=np.float64).reshape(-1)
    if plan.shape != costs.shape:
        raise ValueError("plan and cost_matrix must have the same shape")
    if source_indices.shape != target_indices.shape or source_indices.size == 0:
        raise ValueError("sampled source/target indices must be paired and non-empty")
    if np.any(source_indices < 0) or np.any(source_indices >= plan.shape[0]):
        raise IndexError("sampled source index is out of range")
    if np.any(target_indices < 0) or np.any(target_indices >= plan.shape[1]):
        raise IndexError("sampled target index is out of range")
    if not np.all(np.isfinite(plan)) or not np.all(np.isfinite(costs)):
        raise FloatingPointError("OT diagnostics require finite plan and costs")

    mass = float(plan.sum())
    if mass <= 0:
        raise FloatingPointError("OT diagnostics require positive plan mass")
    probability = np.clip(plan, 0.0, None)
    probability /= probability.sum()
    positive = probability[probability > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    maximum_entropy = float(np.log(probability.size)) if probability.size > 1 else 0.0

    pair_count = source_indices.size
    random_count = min(pair_count, costs.shape[0], costs.shape[1])
    random_costs = costs[np.arange(random_count), np.arange(random_count)]
    selected_costs = costs[source_indices, target_indices]
    random_cost = float(np.mean(random_costs))
    selected_cost = float(np.mean(selected_costs))
    reduction = random_cost - selected_cost
    unique_source = np.unique(source_indices).size
    unique_target = np.unique(target_indices).size
    source_counts = np.bincount(source_indices, minlength=plan.shape[0])
    target_counts = np.bincount(target_indices, minlength=plan.shape[1])
    regularization_applicable = method in {"sinkhorn", "unbalanced", "partial"}

    scalars = {
        "ot/cost_random_pairing": random_cost,
        "ot/cost_selected_pairing": selected_cost,
        "ot/cost_reduction": reduction,
        "ot/cost_reduction_ratio": _safe_ratio(reduction, random_cost),
        "ot/cost_matrix_mean": float(np.mean(costs)),
        "ot/cost_matrix_median": float(np.median(costs)),
        "ot/cost_matrix_max": float(np.max(costs)),
        "ot/cost_to_regularization_ratio": (
            _safe_ratio(float(np.mean(costs)), regularization)
            if regularization_applicable
            else 0.0
        ),
        "ot/regularization_applicable": float(regularization_applicable),
        "ot/plan_mass": mass,
        "ot/plan_entropy": entropy,
        "ot/normalized_plan_entropy": (
            entropy / maximum_entropy if maximum_entropy > 0 else 0.0
        ),
        "ot/plan_max_probability": float(np.max(probability)),
        "ot/plan_nonzero_fraction": float(np.mean(plan > 0)),
        "ot/row_marginal_error": float(
            np.max(np.abs(plan.sum(axis=1) - source_marginal))
        ),
        "ot/column_marginal_error": float(
            np.max(np.abs(plan.sum(axis=0) - target_marginal))
        ),
        "ot/unique_source_count": float(unique_source),
        "ot/unique_target_count": float(unique_target),
        "ot/unique_source_fraction": float(unique_source / pair_count),
        "ot/unique_target_fraction": float(unique_target / pair_count),
        "ot/source_duplicate_fraction": float(1.0 - unique_source / pair_count),
        "ot/target_duplicate_fraction": float(1.0 - unique_target / pair_count),
        "ot/fallback_count": float(fallback_count),
        "ot/nonfinite_plan_count": float(nonfinite_plan_count),
        "ot/solver_warning_count": float(solver_warning_count),
    }
    if not all(np.isfinite(value) for value in scalars.values()):
        raise FloatingPointError("OT scalar diagnostics must be finite")
    histograms = {
        "ot/selected_source_multiplicity": source_counts,
        "ot/selected_target_multiplicity": target_counts,
        "ot/transport_costs": selected_costs,
        "ot/nonzero_transport_probabilities": positive,
    }
    return scalars, histograms


def from_sampler_diagnostics(
    sampler_diagnostics: Mapping[str, object],
    plan: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    return compute_ot_diagnostics(
        plan=plan,
        cost_matrix=np.asarray(sampler_diagnostics["cost_matrix"]),
        source_indices=source_indices,
        target_indices=target_indices,
        intended_source_marginal=np.asarray(
            sampler_diagnostics["intended_source_marginal"]
        ),
        intended_target_marginal=np.asarray(
            sampler_diagnostics["intended_target_marginal"]
        ),
        regularization=float(sampler_diagnostics["regularization"]),
        method=str(sampler_diagnostics["method"]),
        fallback_count=int(sampler_diagnostics["fallback_count"]),
        nonfinite_plan_count=int(sampler_diagnostics["nonfinite_plan_count"]),
        solver_warning_count=int(sampler_diagnostics["solver_warning_count"]),
    )
