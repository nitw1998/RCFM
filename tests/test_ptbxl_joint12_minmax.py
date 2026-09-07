import json
from pathlib import Path

import numpy as np

from data import get_ecg2ecg_datasets
from scripts.preprocess_ptbxl import (
    joint12_minmax_neg1_1,
    record_joint12_minmax_statistics,
)
from train_cfm_compare import parse_args_with_config


TARGET_INDICES = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_joint12_minmax_uses_one_affine_transform_and_preserves_lead_differences():
    time = np.arange(8, dtype=np.float32)
    values = np.stack([time * (lead + 1) - 3 * lead for lead in range(12)], axis=-1)[None]
    minima, ranges = record_joint12_minmax_statistics(values, model_window_samples=8)
    normalized = joint12_minmax_neg1_1(values, minima, ranges)
    assert minima.shape == ranges.shape == (1,)
    assert normalized.min() == -1.0
    assert normalized.max() == 1.0
    np.testing.assert_allclose(
        normalized[0, :, 10] - normalized[0, :, 1],
        2.0 * (values[0, :, 10] - values[0, :, 1]) / ranges[0],
        atol=1e-6,
    )
    restored = (normalized + 1.0) * ranges[:, None, None] / 2.0 + minima[:, None, None]
    np.testing.assert_allclose(restored, values, atol=1e-5)


def test_ecg_loader_applies_same_joint12_coefficients_to_lead2_and_targets(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "PTBXL"
    dataset_root.mkdir()
    ramp = np.arange(128, dtype=np.float32)
    train = np.stack([(lead + 1) * ramp - 4 * lead for lead in range(12)], axis=-1)[None]
    heldout = (train * 0.5) + 7.0
    for split, values, record_id in (
        ("train", train, 101),
        ("val", heldout, 202),
    ):
        np.save(dataset_root / f"X_{split}_resampled.npy", values)
        np.save(dataset_root / f"record_ids_{split}.npy", np.asarray([record_id]))
        minima, ranges = record_joint12_minmax_statistics(values, model_window_samples=128)
        np.save(dataset_root / f"record_joint_minima_{split}.npy", minima)
        np.save(dataset_root / f"record_joint_ranges_{split}.npy", ranges)
    monkeypatch.setattr(
        "data._ecg_region_mask",
        lambda channel, sampling_rate: np.zeros((1, channel.shape[-1]), dtype=np.float32),
    )

    train_set, heldout_set = get_ecg2ecg_datasets(
        DATA_PATH=str(tmp_path),
        datasets=["PTBXL"],
        window_size=1,
        condition_lead=1,
        target_lead=TARGET_INDICES,
        normalization_id="record_joint12_minmax_neg1_1_v1",
    )

    assert train_set.target_ecg.shape == (1, 11, 128)
    assert train_set.condition_signal.shape == (1, 128)
    np.testing.assert_allclose(
        train_set.target_scales,
        np.broadcast_to(train_set.condition_scales[:, None], (1, 11)),
    )
    np.testing.assert_allclose(
        heldout_set.target_offsets,
        np.broadcast_to(heldout_set.condition_offsets[:, None], (1, 11)),
    )
    restored = (
        (heldout_set.target_ecg + 1.0) * heldout_set.target_scales[..., None] / 2.0
        + heldout_set.target_offsets[..., None]
    )
    np.testing.assert_allclose(
        restored, np.transpose(heldout[:, :, TARGET_INDICES], (0, 2, 1)), atol=1e-4
    )
    assert train_set.normalization_metadata["heldout_target_statistics_used"] is True


def test_joint12_lead2_training_config_and_launcher_contract():
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/ptbxl/cfm_compare_record_joint12_minmax_neg1_1_no_ot_seed31.yaml"
    args = parse_args_with_config(["--config", str(config)])
    assert args.condition_lead_index == 1
    assert args.target_lead_indices == TARGET_INDICES
    assert args.normalization_id == "record_joint12_minmax_neg1_1_v1"
    assert args.region_weight == 0.0
    assert args.use_minibatch_ot is False
    assert args.split_hash == "8c14a068e98ab485fb9f519b1ff9f414b81be08cc43e41fef55cc4ef081d07cd"
    json.loads(config.read_text(encoding="utf-8"))
    launcher = (root / "scripts/train_ptbxl_lead2_to_other11_joint12.sh").read_text()
    assert "PTBXL_JOINT12_DATA_ROOT" in launcher
    assert "train_cfm_compare.py" in launcher
    assert "record_joint_minima_train.npy" in launcher
