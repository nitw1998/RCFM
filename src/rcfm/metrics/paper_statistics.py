"""Frozen waveform statistics shared by all paper datasets."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Sequence

import numpy as np

from .waveform import waveform_frechet_distance


ELEVEN_TARGET_LEADS = (
    "I", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
)


def _as_waveforms(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[:, None, :]
    if array.ndim != 3 or len(array) < 2 or array.shape[-1] < 2:
        raise ValueError("waveforms must have shape (samples, channels, time)")
    if not np.all(np.isfinite(array)):
        raise ValueError("waveforms must be finite")
    return array


def per_sample_pearson(reference: np.ndarray, generated: np.ndarray) -> np.ndarray:
    """Pearson r for each paired four-second analysis unit."""

    reference = _as_waveforms(reference)
    generated = _as_waveforms(generated)
    if reference.shape != generated.shape:
        raise ValueError("reference and generated shapes differ")
    x = reference.reshape(len(reference), -1)
    y = generated.reshape(len(generated), -1)
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    output = np.full(len(x), np.nan, dtype=np.float64)
    valid = denominator > np.finfo(np.float64).eps
    output[valid] = np.sum(x[valid] * y[valid], axis=1) / denominator[valid]
    return np.clip(output, -1.0, 1.0)


def _fisher_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    clipped = np.clip(finite, -1.0 + 1e-7, 1.0 - 1e-7)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def hierarchical_pearson(
    reference: np.ndarray,
    generated: np.ndarray,
    group_ids: Sequence[object],
) -> tuple[dict[str, object], np.ndarray, OrderedDict[str, float | None]]:
    """Aggregate window correlations within groups before equal-group reporting."""

    per_sample = per_sample_pearson(reference, generated)
    groups = np.asarray(group_ids).astype(str)
    if groups.shape != (len(per_sample),):
        raise ValueError("group_ids must contain one value per waveform")
    ordered_groups: OrderedDict[str, float | None] = OrderedDict()
    for group in dict.fromkeys(groups.tolist()):
        ordered_groups[group] = _fisher_mean(per_sample[groups == group])
    finite_groups = np.asarray(
        [value for value in ordered_groups.values() if value is not None], dtype=np.float64
    )
    summary = {
        "definition": (
            "Pearson per four-second channel-time-flattened sample; Fisher-z mean "
            "within source subject/patient/record; equal-group summary"
        ),
        "sample_count": int(len(per_sample)),
        "sample_usable": int(np.isfinite(per_sample).sum()),
        "sample_median": float(np.nanmedian(per_sample)) if np.isfinite(per_sample).any() else None,
        "group_count": int(len(ordered_groups)),
        "group_usable": int(len(finite_groups)),
        "group_median_primary": float(np.median(finite_groups)) if len(finite_groups) else None,
        "group_fisher_mean": _fisher_mean(finite_groups),
        "group_q1": float(np.quantile(finite_groups, 0.25)) if len(finite_groups) else None,
        "group_q3": float(np.quantile(finite_groups, 0.75)) if len(finite_groups) else None,
    }
    return summary, per_sample, ordered_groups


def waveform_fd_summary(
    reference: np.ndarray,
    generated: np.ndarray,
    lead_names: Iterable[str] | None = None,
) -> dict[str, object]:
    """Full-test-set wFD, with a fixed macro-lead definition for 11-lead tasks."""

    reference = _as_waveforms(reference)
    generated = _as_waveforms(generated)
    if reference.shape != generated.shape:
        raise ValueError("reference and generated shapes differ")
    if reference.shape[1] == 1:
        names = tuple(lead_names or ())
        if names not in {(), ("ECG",)}:
            raise ValueError("single-lead wFD does not accept multi-lead names")
        value = waveform_frechet_distance(reference, generated)
        return {
            "definition": "full-test-set waveform-vector Frechet distance",
            "aggregation": "single_target_lead",
            "value": value,
            "per_lead": {"ECG": value},
        }
    names = tuple(lead_names or ())
    if reference.shape[1] != 11 or names != ELEVEN_TARGET_LEADS:
        raise ValueError("multi-lead paper wFD requires the frozen 11-lead order")
    per_lead = {
        name: waveform_frechet_distance(reference[:, index], generated[:, index])
        for index, name in enumerate(names)
    }
    return {
        "definition": (
            "wFD independently on each target lead over the complete test split, "
            "then unweighted arithmetic mean over the frozen 11 target leads"
        ),
        "aggregation": "macro_11_target_leads",
        "lead_order": list(names),
        "value": float(np.mean(list(per_lead.values()))),
        "per_lead": per_lead,
    }


def raw_waveform_summary(
    reference: np.ndarray,
    generated: np.ndarray,
    group_ids: Sequence[object],
    lead_names: Iterable[str] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    reference = _as_waveforms(reference)
    generated = _as_waveforms(generated)
    if reference.shape != generated.shape:
        raise ValueError("reference and generated shapes differ")
    error = generated - reference
    pearson, per_sample, per_group = hierarchical_pearson(
        reference, generated, group_ids
    )
    return {
        "support": "raw_full_window",
        "samples": int(len(reference)),
        "channels": int(reference.shape[1]),
        "time_samples": int(reference.shape[2]),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "wfd": waveform_fd_summary(reference, generated, lead_names),
        "pearson": pearson,
    }, {"per_sample_pearson": per_sample, "per_group_pearson": per_group}


def estimate_training_fixed_lag(
    training_source: np.ndarray,
    training_target: np.ndarray,
    max_lag_samples: int,
) -> dict[str, object]:
    """Estimate one global lag using training source-target pairs only."""

    source = _as_waveforms(training_source)
    target = _as_waveforms(training_target)
    if len(source) != len(target) or source.shape[-1] != target.shape[-1]:
        raise ValueError("training source and target must be row/time aligned")
    if source.shape[1] != 1:
        raise ValueError("the fixed-lag estimator requires a single source channel")
    if max_lag_samples <= 0 or 2 * max_lag_samples >= source.shape[-1]:
        raise ValueError("invalid max_lag_samples")
    margin = int(max_lag_samples)
    target_center = target[:, :, margin:-margin]
    target_sum_channels = target_center.sum(axis=1)
    target_sum = float(target_center.sum())
    target_square_sum = float(np.square(target_center).sum())
    point_count = int(target_center.size)
    scores = []
    for lag in range(-margin, margin + 1):
        start = margin - lag
        source_slice = source[:, 0, start : source.shape[-1] - margin - lag]
        channels = target.shape[1]
        source_sum = float(source_slice.sum()) * channels
        source_square_sum = float(np.square(source_slice).sum()) * channels
        cross_sum = float((target_sum_channels * source_slice).sum())
        covariance = cross_sum - target_sum * source_sum / point_count
        target_ss = target_square_sum - target_sum * target_sum / point_count
        source_ss = source_square_sum - source_sum * source_sum / point_count
        denominator = np.sqrt(max(target_ss, 0.0) * max(source_ss, 0.0))
        score = covariance / denominator if denominator > 0 else np.nan
        scores.append((lag, float(score)))
    finite = [(lag, score) for lag, score in scores if np.isfinite(score)]
    if not finite:
        raise FloatingPointError("all training lag scores are nonfinite")
    best_lag, best_score = min(finite, key=lambda item: (-item[1], abs(item[0]), item[0]))
    return {
        "lag_samples": int(best_lag),
        "training_pointwise_pearson": float(best_score),
        "max_lag_samples": margin,
        "selection_data": "training_source_and_training_target_only",
        "test_target_used_for_selection": False,
        "shift_definition": "positive values delay the generated waveform",
        "score_curve": [{"lag_samples": lag, "pearson": score} for lag, score in scores],
    }


def apply_fixed_lag(
    reference: np.ndarray,
    generated: np.ndarray,
    lag_samples: int,
    margin_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return common-support target, unshifted prediction, and fixed-lag prediction."""

    reference = _as_waveforms(reference)
    generated = _as_waveforms(generated)
    if reference.shape != generated.shape:
        raise ValueError("reference and generated shapes differ")
    margin = int(margin_samples)
    lag = int(lag_samples)
    if margin <= 0 or abs(lag) > margin or 2 * margin >= reference.shape[-1]:
        raise ValueError("fixed lag must lie inside a valid common-support margin")
    target = reference[:, :, margin:-margin]
    unshifted = generated[:, :, margin:-margin]
    start = margin - lag
    aligned = generated[:, :, start : generated.shape[-1] - margin - lag]
    if target.shape != aligned.shape:
        raise AssertionError("fixed-lag slicing changed the support shape")
    return target, unshifted, aligned
