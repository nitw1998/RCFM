"""Distribution distance computed directly on waveform vectors."""

from __future__ import annotations

import numpy as np


def _positive_semidefinite_sqrt(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1e-8 * scale:
        raise FloatingPointError("waveform_fd covariance is not positive semidefinite")
    return (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T


def waveform_frechet_distance(real: np.ndarray, generated: np.ndarray) -> float:
    """Return Gaussian Frechet distance in raw normalized waveform-vector space."""

    real = np.asarray(real, dtype=np.float64).reshape(len(real), -1)
    generated = np.asarray(generated, dtype=np.float64).reshape(len(generated), -1)
    if real.shape != generated.shape or real.shape[0] < 2:
        raise ValueError("waveform_fd requires matching arrays with at least two samples")
    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(generated)):
        raise ValueError("waveform_fd inputs must be finite")
    mean_difference = real.mean(axis=0) - generated.mean(axis=0)
    covariance_real = np.atleast_2d(np.cov(real, rowvar=False))
    covariance_generated = np.atleast_2d(np.cov(generated, rowvar=False))
    covariance_real_sqrt = _positive_semidefinite_sqrt(covariance_real)
    covariance_middle = covariance_real_sqrt @ covariance_generated @ covariance_real_sqrt
    covariance_mean = _positive_semidefinite_sqrt(covariance_middle)
    distance = (
        mean_difference @ mean_difference
        + np.trace(covariance_real)
        + np.trace(covariance_generated)
        - 2.0 * np.trace(covariance_mean)
    )
    if not np.isfinite(distance):
        raise FloatingPointError("waveform_fd is nonfinite")
    return float(max(distance, 0.0))
