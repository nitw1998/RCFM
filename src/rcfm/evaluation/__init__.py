"""Evaluation helpers with explicit aggregation and provenance contracts."""

from .diagnostic_transfer import (
    aggregate_by_group,
    as_time_leads,
    binary_metrics,
    load_array_spec,
    paired_bootstrap_mean,
    paired_stratified_bootstrap_auc_difference,
    reconstruct_twelve_leads,
    stratified_bootstrap_auc,
)

__all__ = [
    "aggregate_by_group",
    "as_time_leads",
    "binary_metrics",
    "load_array_spec",
    "paired_bootstrap_mean",
    "paired_stratified_bootstrap_auc_difference",
    "reconstruct_twelve_leads",
    "stratified_bootstrap_auc",
]
