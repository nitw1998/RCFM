"""Interpretability and ECG delineation utilities for region-mask studies."""

from .gradcam import (
    gradcam_1d,
    normalize_soft_mask,
    stitch_temporal_cams,
    validation_crop_starts,
)

__all__ = [
    "gradcam_1d",
    "normalize_soft_mask",
    "stitch_temporal_cams",
    "validation_crop_starts",
]
