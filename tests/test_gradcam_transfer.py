import numpy as np
import pytest
import torch
import torch.nn as nn

from src.rcfm.interpretability.gradcam import (
    covering_crop_starts,
    gradcam_1d,
    gradcam_native_multi_1d,
    gradcam_native_multi_target_1d,
    normalize_soft_mask,
    project_cam_to_sample_grid,
    resample_mask_to_sample_grid,
    stitch_temporal_cams,
    validation_crop_starts,
)
from src.rcfm.interpretability.ptbxl_benchmark_compat import legacy_bn_drop_lin


class _TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(2, 4, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(4, 3)

    def forward(self, inputs):
        features = self.features(inputs)
        return self.head(self.pool(features).flatten(1))


def test_ptbxl_validation_crop_policy_and_overlap_mean_stitching():
    starts = validation_crop_starts(total_length=1000, crop_length=250, stride=125)
    assert starts == [0, 125, 250, 375, 500, 625, 750]

    cams = [np.full(250, index + 1, dtype=np.float32) for index in range(len(starts))]
    stitched = stitch_temporal_cams(cams, starts, total_length=1000)
    assert stitched.shape == (1000,)
    assert stitched[0] == pytest.approx(1.0)
    assert stitched[125] == pytest.approx(1.5)
    assert stitched[875] == pytest.approx(7.0)


def test_gradcam_is_finite_and_has_input_temporal_length():
    torch.manual_seed(7)
    model = _TinyClassifier().eval()
    inputs = torch.randn(1, 2, 37)

    cam, logits = gradcam_1d(model, model.features, inputs, target_index=1)

    assert cam.shape == (37,)
    assert logits.shape == (3,)
    assert np.all(np.isfinite(cam))
    assert np.all(cam >= 0)


def test_native_multi_layer_gradcam_preserves_each_feature_resolution():
    torch.manual_seed(7)
    model = _TinyClassifier().eval()
    inputs = torch.randn(1, 2, 37)

    cams, logits = gradcam_native_multi_1d(
        model,
        {"conv": model.features[0], "features": model.features},
        inputs,
        target_index=1,
    )

    assert cams["conv"].shape == (37,)
    assert cams["features"].shape == (37,)
    assert logits.shape == (3,)
    assert all(np.all(np.isfinite(cam)) for cam in cams.values())


def test_multi_target_gradcam_aggregates_known_positive_logits():
    torch.manual_seed(11)
    model = _TinyClassifier().eval()
    inputs = torch.randn(1, 2, 37)

    cams, logits = gradcam_native_multi_target_1d(
        model,
        {"features": model.features},
        inputs,
        target_indices=[0, 2],
        reduction="mean",
    )

    assert cams["features"].shape == (37,)
    assert logits.shape == (3,)
    assert np.all(np.isfinite(cams["features"]))
    with pytest.raises(ValueError, match="nonempty"):
        gradcam_native_multi_target_1d(model, {"features": model.features}, inputs, [])


def test_explicit_feature_centers_remove_half_bin_projection_shift():
    native = np.zeros(8, dtype=np.float32)
    native[3] = 1.0
    projected = project_cam_to_sample_grid(native, output_length=250, feature_stride=32)

    assert int(np.argmax(projected)) == 96
    assert projected[96] == pytest.approx(1.0)


def test_covering_crops_and_sample_center_mask_resampling():
    assert covering_crop_starts(400, 250, 125) == [0, 125, 150]
    assert covering_crop_starts(1000, 250, 125)[-1] == 750

    source = np.zeros(100, dtype=np.float32)
    source[25] = 1.0
    target = resample_mask_to_sample_grid(source, 100, 128, output_length=128)
    assert int(np.argmax(target)) == 32


def test_soft_mask_normalization_and_degenerate_flag():
    normalized, degenerate = normalize_soft_mask(np.asarray([2.0, 4.0, 6.0]))
    assert not degenerate
    np.testing.assert_allclose(normalized, [0.0, 0.5, 1.0])

    constant, degenerate = normalize_soft_mask(np.ones(4))
    assert degenerate
    np.testing.assert_array_equal(constant, np.zeros(4))


def test_legacy_fastai_head_layer_order_matches_checkpoint_keys():
    layers = legacy_bn_drop_lin(512, 128, True, 0.25, nn.ReLU(inplace=True))
    assert [type(layer) for layer in layers] == [
        nn.BatchNorm1d,
        nn.Dropout,
        nn.Linear,
        nn.ReLU,
    ]
