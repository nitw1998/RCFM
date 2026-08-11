"""Paired agreement summaries for clinical parameters."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats


def _finite_pairs(reference: Iterable[float], generated: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    reference_array = np.asarray(reference, dtype=np.float64)
    generated_array = np.asarray(generated, dtype=np.float64)
    if reference_array.shape != generated_array.shape:
        raise ValueError("reference and generated values must have identical shapes")
    finite = np.isfinite(reference_array) & np.isfinite(generated_array)
    return reference_array[finite], generated_array[finite]


def bland_altman(
    reference: Iterable[float],
    generated: Iterable[float],
    limits_z: float = 1.96,
) -> dict[str, object]:
    """Return generated-minus-reference Bland--Altman values and limits."""

    reference_array, generated_array = _finite_pairs(reference, generated)
    if reference_array.size < 2:
        raise ValueError("Bland--Altman analysis requires at least two finite pairs")
    differences = generated_array - reference_array
    pair_means = (generated_array + reference_array) / 2.0
    bias = float(np.mean(differences))
    difference_sd = float(np.std(differences, ddof=1))
    return {
        "n": int(reference_array.size),
        "difference_definition": "generated_minus_reference",
        "bias": bias,
        "difference_sd": difference_sd,
        "lower_limit": bias - limits_z * difference_sd,
        "upper_limit": bias + limits_z * difference_sd,
        "limits_z": float(limits_z),
        "pair_means": pair_means.tolist(),
        "differences": differences.tolist(),
    }


def paired_correlation(reference: Iterable[float], generated: Iterable[float]) -> dict[str, object]:
    """Return Pearson correlation while retaining the number of usable pairs."""

    reference_array, generated_array = _finite_pairs(reference, generated)
    if reference_array.size < 2:
        raise ValueError("Correlation requires at least two finite pairs")
    if np.ptp(reference_array) == 0 or np.ptp(generated_array) == 0:
        return {"n": int(reference_array.size), "r": None, "p_value": None, "status": "constant_input"}
    result = stats.pearsonr(reference_array, generated_array)
    return {
        "n": int(reference_array.size),
        "r": float(result.statistic),
        "p_value": float(result.pvalue),
        "status": "ok",
    }
