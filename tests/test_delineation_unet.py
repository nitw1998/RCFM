from pathlib import Path

import numpy as np
import pytest
import torch

from src.rcfm.interpretability.delineation_dataset import (
    EVENT_NAMES,
    annotation_sample_to_output,
    build_window_targets,
    parse_fiducial_events,
    robust_window_normalize,
)
from src.rcfm.interpretability.delineation_unet import (
    DelineationResUNet1D,
    delineation_loss,
)
from scripts.train_delineation_unet import match_event_positions, parse_args_with_config
from src.rcfm.runtime import exception_summary


def _two_beat_annotations():
    notes = []
    samples = []
    for offset in (0, 500):
        for sample, note in [
            (10, "p-wave onset"),
            (20, "p-wave peak"),
            (30, "p-wave offset"),
            (40, "QRS onset"),
            (50, "R peak"),
            (60, "QRS offset"),
            (80, "t-wave onset"),
            (100, "t-wave peak"),
            (130, "t-wave offset"),
        ]:
            samples.append(sample + offset)
            notes.append(note)
    return samples, notes


def test_annotation_mapping_and_complete_wave_triplets():
    samples, notes = _two_beat_annotations()
    positions, counts, valid, unmatched = parse_fiducial_events(samples, notes)

    assert annotation_sample_to_output(500) == 128
    assert counts.tolist() == [2] * len(EVENT_NAMES)
    assert valid.tolist() == [1, 1, 1]
    assert set(unmatched.values()) == {0}
    assert positions[4, :2].tolist() == [13, 141]


def test_afib_disables_p_supervision_without_removing_qrs_or_t():
    samples, notes = _two_beat_annotations()
    _, _, valid, _ = parse_fiducial_events(samples, notes, disable_p_wave=True)
    assert valid.tolist() == [0, 1, 1]


def test_window_targets_preserve_coordinates_and_ignore_partial_edges():
    positions = np.full((9, 4), -1, dtype=np.int16)
    counts = np.full(9, 2, dtype=np.uint8)
    for event_index, values in enumerate(
        ([90, 250], [100, 260], [110, 270], [120, 280], [130, 290], [140, 300], [150, 320], [170, 340], [200, 370])
    ):
        positions[event_index, :2] = values
    targets = build_window_targets(
        positions,
        counts,
        np.ones(3, dtype=np.uint8),
        crop_start=100,
        window_samples=256,
        heatmap_sigma_samples=2,
    )

    assert targets["regions"].shape == (3, 256)
    assert targets["heatmaps"].shape == (9, 256)
    assert targets["heatmaps"][1, 0] == pytest.approx(1.0)
    assert targets["region_mask"][0, :11].sum() == 0
    assert targets["region_mask"][2, -36:].sum() == 0


def test_robust_normalization_is_positive_affine_invariant():
    signal = np.sin(np.linspace(0, 8 * np.pi, 512)).astype(np.float32)
    np.testing.assert_allclose(
        robust_window_normalize(signal),
        robust_window_normalize(7.0 * signal + 3.0),
        atol=2e-6,
    )
    with pytest.raises(ValueError, match="constant"):
        robust_window_normalize(np.ones(512, dtype=np.float32))


def test_dual_head_unet_is_stride_one_and_loss_is_finite():
    model = DelineationResUNet1D(base_channels=8)
    signal = torch.randn(2, 1, 512)
    outputs = model(signal)
    assert outputs["region_logits"].shape == (2, 3, 512)
    assert outputs["fiducial_logits"].shape == (2, 9, 512)
    batch = {
        "regions": torch.zeros(2, 3, 512),
        "region_mask": torch.ones(2, 3, 512),
        "heatmaps": torch.zeros(2, 9, 512),
        "heatmap_mask": torch.ones(2, 9, 512),
    }
    losses = delineation_loss(outputs, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_event_matching_rejects_large_or_duplicate_offsets():
    errors, misses, extras = match_event_positions([9, 21, 90], [10, 20, 40], 5)
    assert errors == [1, 1]
    assert misses == 1
    assert extras == 1


def test_keyboard_interrupt_has_safe_metadata_summary():
    assert exception_summary(KeyboardInterrupt()) == "KeyboardInterrupt"


def test_delineation_config_is_full_online_training():
    config = Path(__file__).resolve().parents[1] / "configs" / "ptbxl" / "delineation_limb_resunet_seed31.yaml"
    parsed = parse_args_with_config(["--config", str(config)])
    assert parsed.leads == ["I", "II", "III", "aVR", "aVL", "aVF"]
    assert parsed.epochs == 50
    assert parsed.batch_size == 2048
    assert parsed.validation_batch_size == 2048
    assert parsed.num_workers == 8
    assert parsed.wandb_mode == "online"
    assert parsed.device == "cuda:0"
    assert parsed.max_batches is None
