import copy

import numpy as np
import pytest

from src.rcfm.metrics.statistics import (
    aggregate_seed_metrics,
    align_subject_metric,
    compare_models,
    hierarchical_bootstrap_mean_difference,
    holm_adjust,
    make_run_record,
)


def _record(model, seed, values, subjects=None):
    subjects = subjects or [f"s{index}" for index in range(len(values))]
    provenance = {
        "model": model,
        "config": f"{model.lower()}.yaml",
        "seed": seed,
        "split_hash": "abc123",
        "checkpoint": f"{model}-{seed}.pt",
        "git_commit": "deadbeef",
        "dataset": "synthetic",
        "task": "ecg2ecg",
        "implementation_status": "proposed",
        "command": f"train --model {model} --seed {seed}",
    }
    metrics = [
        {"subject_id": subject, "metrics": {"rmse": value, "mae": value / 2}}
        for subject, value in zip(subjects, values)
    ]
    return make_run_record(provenance, metrics)


def test_run_record_requires_complete_provenance():
    record = _record("RCFM", 31, [1.0, 2.0])
    broken = copy.deepcopy(record)
    del broken["provenance"]["split_hash"]

    with pytest.raises(ValueError, match="split_hash"):
        make_run_record(broken["provenance"], broken["subject_metrics"])


def test_paired_alignment_rejects_missing_subjects():
    reference = _record("RDDM", 31, [1.0, 2.0], subjects=["a", "b"])
    comparison = _record("RCFM", 31, [0.8, 1.8], subjects=["a", "c"])

    with pytest.raises(ValueError, match="subject sets differ"):
        align_subject_metric(reference, comparison, "rmse")


def test_seed_aggregation_uses_subject_mean_then_seed_distribution():
    records = [
        _record("RCFM", 31, [1.0, 3.0]),
        _record("RCFM", 47, [2.0, 4.0]),
        _record("RCFM", 83, [3.0, 5.0]),
    ]
    result = aggregate_seed_metrics(records)["rmse"]

    assert result["n_seeds"] == 3
    np.testing.assert_allclose(result["mean"], 3.0)
    np.testing.assert_allclose(result["std"], 1.0)
    assert result["status"] == "ok"


def test_seed_aggregation_rejects_duplicate_seed_or_changed_subject_set():
    duplicate = [_record("RCFM", 31, [1.0, 2.0]), _record("RCFM", 31, [1.5, 2.5])]
    with pytest.raises(ValueError, match="unique"):
        aggregate_seed_metrics(duplicate)

    changed_subjects = [
        _record("RCFM", 31, [1.0, 2.0], subjects=["a", "b"]),
        _record("RCFM", 47, [1.5, 2.5], subjects=["a", "c"]),
    ]
    with pytest.raises(ValueError, match="differ across seeds"):
        aggregate_seed_metrics(changed_subjects)


def test_paired_comparison_rejects_duplicate_seed_runs():
    reference = [_record("RDDM", 31, [1.0, 2.0]), _record("RDDM", 31, [1.1, 2.1])]
    comparison = [_record("RCFM", 31, [0.9, 1.9])]

    with pytest.raises(ValueError, match="exactly one run per seed"):
        compare_models(reference, comparison, "rmse")


def test_paired_comparison_aggregates_seeds_with_deterministic_ci():
    reference = [
        _record("RDDM", 31, [1.0, 1.2, 1.4, 1.6, 1.8]),
        _record("RDDM", 47, [1.1, 1.3, 1.5, 1.7, 1.9]),
        _record("RDDM", 83, [0.9, 1.1, 1.3, 1.5, 1.7]),
    ]
    comparison = [
        _record("RCFM", 31, [0.9, 1.0, 1.3, 1.3, 1.7]),
        _record("RCFM", 47, [1.0, 1.1, 1.4, 1.4, 1.8]),
        _record("RCFM", 83, [0.8, 0.9, 1.2, 1.2, 1.6]),
    ]

    first = compare_models(reference, comparison, "rmse", bootstrap_iterations=500, random_seed=7)
    second = compare_models(reference, comparison, "rmse", bootstrap_iterations=500, random_seed=7)

    assert first["n_subjects"] == 5
    assert first["n_seeds"] == 3
    assert first["difference_definition"] == "comparison_minus_reference"
    assert first["confidence_interval"] == second["confidence_interval"]
    assert first["confidence_interval"]["lower"] <= first["mean_difference"]
    assert first["confidence_interval"]["upper"] >= first["mean_difference"]
    assert first["effect_size"]["name"] in {"cohen_dz", "rank_biserial"}


def test_holm_adjustment_is_monotone_in_sorted_order():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})

    np.testing.assert_allclose(adjusted["a"], 0.03)
    np.testing.assert_allclose(adjusted["c"], 0.06)
    np.testing.assert_allclose(adjusted["b"], 0.06)


def test_hierarchical_bootstrap_resamples_subjects_and_windows():
    windows = {
        "a": ([1.0, 2.0], [0.8, 1.7]),
        "b": ([2.0, 3.0, 4.0], [1.9, 2.8, 3.7]),
        "c": ([1.0, 1.5], [0.9, 1.3]),
    }
    first = hierarchical_bootstrap_mean_difference(
        windows, iterations=300, random_seed=9
    )
    second = hierarchical_bootstrap_mean_difference(
        windows, iterations=300, random_seed=9
    )

    assert first == second
    assert first["lower"] <= -0.2 <= first["upper"]
