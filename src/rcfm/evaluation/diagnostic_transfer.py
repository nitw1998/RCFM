"""Utilities for cross-domain diagnostic-classifier and saliency audits."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_time_leads(values: np.ndarray) -> np.ndarray:
    """Return ECG records as ``(records, time, leads)`` without guessing silently."""

    array = np.asarray(values)
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 12, axis=2).astype(np.float32)
    if array.ndim != 3:
        raise ValueError("waveforms must have shape (N,T), (N,T,C), or (N,C,T)")
    if array.shape[2] in {1, 12} and array.shape[1] > array.shape[2]:
        output = array
    elif array.shape[1] in {1, 12} and array.shape[2] > array.shape[1]:
        output = np.transpose(array, (0, 2, 1))
    else:
        raise ValueError("cannot identify waveform time/lead axes")
    if output.shape[2] == 1:
        output = np.repeat(output, 12, axis=2)
    if output.shape[2] != 12 or not np.all(np.isfinite(output)):
        raise ValueError("diagnostic input must contain 12 finite leads")
    return output.astype(np.float32, copy=False)


def aggregate_by_group(
    logits: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean logits within groups while requiring one binary label per group."""

    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels).astype(bool).reshape(-1)
    group_values = np.asarray(groups).astype(str).reshape(-1)
    if not (len(scores) == len(truth) == len(group_values)):
        raise ValueError("logits, labels, and groups must have identical row counts")
    unique, inverse = np.unique(group_values, return_inverse=True)
    output_scores = np.empty(len(unique), dtype=np.float64)
    output_labels = np.empty(len(unique), dtype=bool)
    for index in range(len(unique)):
        selected = inverse == index
        label_values = np.unique(truth[selected])
        if len(label_values) != 1:
            raise ValueError("a group contains conflicting diagnostic labels")
        output_scores[index] = float(np.mean(scores[selected]))
        output_labels[index] = bool(label_values[0])
    return output_scores, output_labels, np.arange(len(unique), dtype=np.int64)


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    truth = np.asarray(labels).astype(bool).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if bins < 2 or len(truth) != len(scores) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("invalid inputs for calibration error")
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, bins - 1)
    output = 0.0
    for index in range(bins):
        selected = indices == index
        if np.any(selected):
            output += float(np.mean(selected)) * abs(
                float(np.mean(scores[selected])) - float(np.mean(truth[selected]))
            )
    return output


def binary_metrics(labels: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    truth = np.asarray(labels).astype(bool).reshape(-1)
    raw = np.asarray(logits, dtype=np.float64).reshape(-1)
    if len(truth) != len(raw) or len(np.unique(truth)) != 2:
        raise ValueError("binary metrics require aligned positive and negative observations")
    probabilities = expit(raw)
    predicted = probabilities >= threshold
    positive, negative = truth, ~truth
    return {
        "observations": int(len(truth)),
        "positives": int(np.sum(positive)),
        "negatives": int(np.sum(negative)),
        "auroc": float(roc_auc_score(truth, probabilities)),
        "auprc": float(average_precision_score(truth, probabilities)),
        "sensitivity": float(np.mean(predicted[positive])),
        "specificity": float(np.mean(~predicted[negative])),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "brier": float(brier_score_loss(truth, probabilities)),
        "ece_10bin": expected_calibration_error(truth, probabilities, bins=10),
        "probability_threshold": float(threshold),
        "prevalence": float(np.mean(truth)),
    }


def stratified_bootstrap_auc(
    labels: np.ndarray, logits: np.ndarray, *, seed: int = 2031, replicates: int = 2000
) -> dict[str, list[float] | int]:
    truth = np.asarray(labels).astype(bool).reshape(-1)
    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    positives, negatives = np.flatnonzero(truth), np.flatnonzero(~truth)
    if min(len(positives), len(negatives)) < 2 or replicates < 1:
        raise ValueError("stratified bootstrap requires at least two observations per class")
    rng = np.random.default_rng(seed)
    auc = np.empty(replicates, dtype=np.float64)
    ap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draw = np.concatenate([
            rng.choice(positives, len(positives), replace=True),
            rng.choice(negatives, len(negatives), replace=True),
        ])
        auc[index] = roc_auc_score(truth[draw], scores[draw])
        ap[index] = average_precision_score(truth[draw], scores[draw])
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "auroc_95_ci": [float(value) for value in np.quantile(auc, [0.025, 0.975])],
        "auprc_95_ci": [float(value) for value in np.quantile(ap, [0.025, 0.975])],
    }


def paired_stratified_bootstrap_auc_difference(
    labels: np.ndarray,
    candidate_logits: np.ndarray,
    reference_logits: np.ndarray,
    *,
    seed: int = 2031,
    replicates: int = 2000,
) -> dict[str, float | int | list[float]]:
    """Paired class-stratified CI for candidate-minus-reference AUROC/AUPRC."""

    truth = np.asarray(labels).astype(bool).reshape(-1)
    candidate = np.asarray(candidate_logits, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_logits, dtype=np.float64).reshape(-1)
    if not (len(truth) == len(candidate) == len(reference)):
        raise ValueError("paired classifier arrays are not aligned")
    positives, negatives = np.flatnonzero(truth), np.flatnonzero(~truth)
    if min(len(positives), len(negatives)) < 2 or replicates < 1:
        raise ValueError("paired stratified bootstrap requires two observations per class")
    rng = np.random.default_rng(seed)
    auc = np.empty(replicates, dtype=np.float64)
    ap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draw = np.concatenate([
            rng.choice(positives, len(positives), replace=True),
            rng.choice(negatives, len(negatives), replace=True),
        ])
        auc[index] = roc_auc_score(truth[draw], candidate[draw]) - roc_auc_score(
            truth[draw], reference[draw]
        )
        ap[index] = average_precision_score(truth[draw], candidate[draw]) - average_precision_score(
            truth[draw], reference[draw]
        )
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "auroc_difference": float(roc_auc_score(truth, candidate) - roc_auc_score(truth, reference)),
        "auprc_difference": float(
            average_precision_score(truth, candidate) - average_precision_score(truth, reference)
        ),
        "auroc_difference_95_ci": [float(value) for value in np.quantile(auc, [0.025, 0.975])],
        "auprc_difference_95_ci": [float(value) for value in np.quantile(ap, [0.025, 0.975])],
    }


def paired_bootstrap_mean(
    differences: np.ndarray, *, seed: int = 2031, replicates: int = 2000
) -> dict[str, float | int | list[float]]:
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if len(values) < 2 or replicates < 1 or not np.all(np.isfinite(values)):
        raise ValueError("paired bootstrap requires finite paired observations")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = np.mean(values[draws], axis=1)
    return {
        "observations": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "bootstrap_95_ci": [float(value) for value in np.quantile(means, [0.025, 0.975])],
    }


def load_array_spec(specification: str) -> np.ndarray:
    """Load ``path.npy`` or ``path.npz:key`` without pickle support."""

    if ".npz:" in specification:
        path_text, key = specification.rsplit(":", 1)
        path = Path(path_text).resolve()
        with np.load(path, allow_pickle=False) as artifact:
            if key not in artifact.files:
                raise KeyError(f"{key!r} is absent from {path}")
            return np.asarray(artifact[key])
    return np.asarray(np.load(Path(specification).resolve(), allow_pickle=False))


def reconstruct_twelve_leads(condition: np.ndarray, generated: np.ndarray, condition_lead: int = 1) -> np.ndarray:
    source = np.asarray(condition, dtype=np.float32)
    target = np.asarray(generated, dtype=np.float32)
    if source.ndim == 2:
        source = source[:, None, :]
    if target.ndim != 3 or source.ndim != 3 or source.shape[1] != 1 or target.shape[1] != 11:
        raise ValueError("expected condition (N,1,T) and generated targets (N,11,T)")
    if source.shape[0] != target.shape[0] or source.shape[2] != target.shape[2]:
        raise ValueError("condition and generated arrays are not aligned")
    output = np.empty((len(source), 12, source.shape[2]), dtype=np.float32)
    output[:, condition_lead] = source[:, 0]
    output[:, [index for index in range(12) if index != condition_lead]] = target
    return output
