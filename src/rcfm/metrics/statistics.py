"""Subject-level multi-seed aggregation and paired statistical inference."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


REQUIRED_PROVENANCE_FIELDS = {
    "model",
    "config",
    "seed",
    "split_hash",
    "checkpoint",
    "git_commit",
    "dataset",
    "task",
    "implementation_status",
    "command",
}


def make_run_record(
    provenance: Mapping[str, object],
    subject_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    record = {
        "schema_version": 1,
        "provenance": dict(provenance),
        "subject_metrics": [dict(subject) for subject in subject_metrics],
    }
    validate_run_record(record)
    return record


def validate_run_record(record: Mapping[str, object]) -> None:
    if record.get("schema_version") != 1:
        raise ValueError("run record schema_version must be 1")
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("run record requires a provenance mapping")
    missing = sorted(REQUIRED_PROVENANCE_FIELDS - set(provenance))
    if missing:
        raise ValueError(f"missing provenance fields: {', '.join(missing)}")
    subjects = record.get("subject_metrics")
    if not isinstance(subjects, Sequence) or not subjects:
        raise ValueError("run record requires non-empty subject_metrics")
    subject_ids = []
    for subject in subjects:
        if not isinstance(subject, Mapping) or "subject_id" not in subject or "metrics" not in subject:
            raise ValueError("each subject metric requires subject_id and metrics")
        subject_ids.append(str(subject["subject_id"]))
        metrics = subject["metrics"]
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError("subject metrics must be a non-empty mapping")
        if any(not np.isfinite(float(value)) for value in metrics.values()):
            raise ValueError("subject metrics must be finite numbers")
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("subject_ids must be unique within a run")


def _subject_map(record: Mapping[str, object], metric: str) -> dict[str, float]:
    validate_run_record(record)
    output = {}
    for subject in record["subject_metrics"]:
        if metric in subject["metrics"]:
            output[str(subject["subject_id"])] = float(subject["metrics"][metric])
    return output


def align_subject_metric(
    reference: Mapping[str, object],
    comparison: Mapping[str, object],
    metric: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    for field in ("dataset", "task", "split_hash", "seed"):
        if reference["provenance"][field] != comparison["provenance"][field]:
            raise ValueError(f"paired runs have different {field}")
    reference_values = _subject_map(reference, metric)
    comparison_values = _subject_map(comparison, metric)
    if set(reference_values) != set(comparison_values):
        missing_reference = sorted(set(comparison_values) - set(reference_values))
        missing_comparison = sorted(set(reference_values) - set(comparison_values))
        raise ValueError(
            "paired subject sets differ: "
            f"missing_reference={len(missing_reference)}, missing_comparison={len(missing_comparison)}"
        )
    subjects = sorted(reference_values)
    if not subjects:
        raise ValueError(f"metric {metric!r} has no paired subjects")
    return (
        subjects,
        np.asarray([reference_values[subject] for subject in subjects], dtype=np.float64),
        np.asarray([comparison_values[subject] for subject in subjects], dtype=np.float64),
    )


def aggregate_seed_metrics(records: Iterable[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    records = list(records)
    if not records:
        raise ValueError("at least one run is required")
    for record in records:
        validate_run_record(record)
    by_metric: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    context = {
        field: {str(record["provenance"][field]) for record in records}
        for field in ("model", "dataset", "task", "split_hash")
    }
    for field, values in context.items():
        if len(values) != 1:
            raise ValueError(f"all seed runs must use the same {field}")
    seeds = [int(record["provenance"]["seed"]) for record in records]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed runs must be unique")

    expected_subject_metrics = None
    for record in records:
        seed = int(record["provenance"]["seed"])
        subject_metrics = {
            str(subject["subject_id"]): frozenset(str(name) for name in subject["metrics"])
            for subject in record["subject_metrics"]
        }
        if expected_subject_metrics is None:
            expected_subject_metrics = subject_metrics
        elif subject_metrics != expected_subject_metrics:
            raise ValueError("subject or metric sets differ across seeds")
        for subject in record["subject_metrics"]:
            for metric, value in subject["metrics"].items():
                by_metric[str(metric)][seed].append(float(value))
    output = {}
    for metric, seed_subject_values in sorted(by_metric.items()):
        per_seed = {seed: float(np.mean(values)) for seed, values in sorted(seed_subject_values.items())}
        values = np.asarray(list(per_seed.values()), dtype=np.float64)
        output[metric] = {
            "n_seeds": len(per_seed),
            "seeds": list(per_seed),
            "per_seed_subject_mean": per_seed,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "status": "ok" if len(values) >= 3 else "insufficient_principal_seeds",
        }
    return output


def _aggregate_subjects(records: Sequence[Mapping[str, object]], metric: str) -> tuple[list[str], np.ndarray]:
    if not records:
        raise ValueError("at least one run is required")
    seed_maps = [_subject_map(record, metric) for record in records]
    subject_set = set(seed_maps[0])
    if any(set(values) != subject_set for values in seed_maps[1:]):
        raise ValueError("subject sets differ across seeds")
    subjects = sorted(subject_set)
    values = np.asarray([[seed_map[subject] for seed_map in seed_maps] for subject in subjects])
    return subjects, np.mean(values, axis=1)


def bootstrap_mean_difference(
    differences: Sequence[float],
    iterations: int = 10000,
    confidence: float = 0.95,
    random_seed: int = 2026,
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    if values.size < 2 or iterations <= 0:
        raise ValueError("bootstrap requires at least two subjects and positive iterations")
    rng = np.random.default_rng(random_seed)
    samples = rng.choice(values, size=(iterations, values.size), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "iterations": iterations,
        "confidence": confidence,
        "random_seed": random_seed,
        "lower": float(np.quantile(samples, alpha / 2)),
        "upper": float(np.quantile(samples, 1 - alpha / 2)),
    }


def hierarchical_bootstrap_mean_difference(
    paired_windows: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    iterations: int = 10000,
    confidence: float = 0.95,
    random_seed: int = 2026,
) -> dict[str, float | int]:
    subjects = sorted(paired_windows)
    if len(subjects) < 2:
        raise ValueError("hierarchical bootstrap requires at least two subjects")
    differences = {}
    for subject, (reference, comparison) in paired_windows.items():
        reference_values = np.asarray(reference, dtype=np.float64)
        comparison_values = np.asarray(comparison, dtype=np.float64)
        if reference_values.shape != comparison_values.shape or reference_values.size == 0:
            raise ValueError(f"unpaired or empty windows for subject {subject}")
        differences[subject] = comparison_values - reference_values
    rng = np.random.default_rng(random_seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        subject_means = []
        for subject in sampled_subjects:
            values = differences[str(subject)]
            subject_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        estimates[iteration] = np.mean(subject_means)
    alpha = 1.0 - confidence
    return {
        "iterations": iterations,
        "confidence": confidence,
        "random_seed": random_seed,
        "lower": float(np.quantile(estimates, alpha / 2)),
        "upper": float(np.quantile(estimates, 1 - alpha / 2)),
    }


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(np.sum(ranks[nonzero > 0]))
    negative = float(np.sum(ranks[nonzero < 0]))
    return (positive - negative) / (positive + negative)


def compare_models(
    reference_runs: Sequence[Mapping[str, object]],
    comparison_runs: Sequence[Mapping[str, object]],
    metric: str,
    bootstrap_iterations: int = 10000,
    random_seed: int = 2026,
) -> dict[str, object]:
    if not reference_runs or not comparison_runs:
        raise ValueError("both model run groups are required")
    reference_seed_list = [int(record["provenance"]["seed"]) for record in reference_runs]
    comparison_seed_list = [int(record["provenance"]["seed"]) for record in comparison_runs]
    if len(reference_seed_list) != len(set(reference_seed_list)) or len(comparison_seed_list) != len(
        set(comparison_seed_list)
    ):
        raise ValueError("each model must have exactly one run per seed")
    reference_seeds = set(reference_seed_list)
    comparison_seeds = set(comparison_seed_list)
    if reference_seeds != comparison_seeds:
        raise ValueError("model seed sets differ")
    for field in ("dataset", "task", "split_hash"):
        values = {
            str(record["provenance"][field])
            for record in list(reference_runs) + list(comparison_runs)
        }
        if len(values) != 1:
            raise ValueError(f"model runs have different {field}")
    reference_subjects, reference = _aggregate_subjects(reference_runs, metric)
    comparison_subjects, comparison = _aggregate_subjects(comparison_runs, metric)
    if reference_subjects != comparison_subjects:
        raise ValueError("model subject sets differ")
    differences = comparison - reference
    mean_difference = float(np.mean(differences))
    normality_p = None
    if 3 <= len(differences) <= 5000 and np.std(differences) > 0:
        normality_p = float(stats.shapiro(differences).pvalue)

    if np.allclose(differences, 0):
        test_name = "exact_no_difference"
        statistic = 0.0
        p_value = 1.0
        effect_name = "paired_effect"
        effect_size = 0.0
    elif normality_p is not None and normality_p >= 0.05:
        result = stats.ttest_rel(comparison, reference)
        test_name = "paired_t"
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        effect_name = "cohen_dz"
        effect_size = mean_difference / float(np.std(differences, ddof=1))
    else:
        result = stats.wilcoxon(comparison, reference, zero_method="wilcox", alternative="two-sided")
        test_name = "wilcoxon_signed_rank"
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        effect_name = "rank_biserial"
        effect_size = _rank_biserial(differences)
    return {
        "metric": metric,
        "n_subjects": len(reference_subjects),
        "n_seeds": len(reference_seeds),
        "difference_definition": "comparison_minus_reference",
        "mean_difference": mean_difference,
        "confidence_interval": bootstrap_mean_difference(
            differences,
            iterations=bootstrap_iterations,
            random_seed=random_seed,
        ),
        "normality_test": {
            "method": "shapiro_wilk",
            "p_value": normality_p,
            "status": "evaluated" if normality_p is not None else "not_applicable",
        },
        "test": test_name,
        "statistic": statistic,
        "p_value": p_value,
        "effect_size": {"name": effect_name, "value": effect_size},
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(p_value))
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted
