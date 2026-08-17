"""Metrics with explicit provenance and failure handling.

Import metric functions from their concrete modules so statistics-only tools do
not initialize optional ECG delineation dependencies.
"""

from .paper_statistics import (
    ELEVEN_TARGET_LEADS,
    apply_fixed_lag,
    estimate_training_fixed_lag,
    hierarchical_pearson,
    raw_waveform_summary,
    waveform_fd_summary,
)

__all__ = [
    "ELEVEN_TARGET_LEADS",
    "apply_fixed_lag",
    "estimate_training_fixed_lag",
    "hierarchical_pearson",
    "raw_waveform_summary",
    "waveform_fd_summary",
]
