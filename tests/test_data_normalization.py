import numpy as np
import pytest

import data
from data import (
    PairedSignalDataset,
    _record_minmax_coefficients,
    _record_minmax_neg1_1,
    _record_zscore,
    _rddm_window_minmax_neg1_1,
    _training_global_zscore,
    get_ecg2ecg_datasets,
    get_ppg2ecg_datasets,
)


def test_global_zscore_fits_training_only_and_is_invertible():
    train_target = np.array([[0.0, 2.0], [4.0, 6.0]])
    train_condition = np.array([[10.0, 14.0], [18.0, 22.0]])
    test_target = np.array([[100.0, 102.0]])
    test_condition = np.array([[200.0, 204.0]])

    normalized = _training_global_zscore(
        train_target,
        train_condition,
        test_target,
        test_condition,
    )
    normalized_train_target, _, normalized_test_target, _, metadata = normalized

    np.testing.assert_allclose(np.mean(normalized_train_target), 0.0, atol=1e-7)
    assert np.mean(normalized_test_target) > 10
    restored_test = (
        normalized_test_target * metadata["target_scale"] + metadata["target_mean"]
    )
    np.testing.assert_allclose(restored_test, test_target)


def test_inference_reuses_checkpoint_stats_without_refitting():
    train_target = np.array([[0.0, 2.0], [4.0, 6.0]])
    train_condition = np.array([[10.0, 14.0], [18.0, 22.0]])
    test_target = np.array([[8.0, 10.0]])
    test_condition = np.array([[26.0, 30.0]])
    fitted = _training_global_zscore(
        train_target,
        train_condition,
        test_target,
        test_condition,
    )
    metadata = fitted[-1]

    shifted_training = _training_global_zscore(
        train_target + 1000,
        train_condition + 1000,
        test_target,
        test_condition,
        metadata=metadata,
    )

    np.testing.assert_allclose(shifted_training[2], fitted[2])
    np.testing.assert_allclose(shifted_training[3], fitted[3])


def test_record_zscore_is_per_record_and_exactly_invertible():
    signals = np.array([[1.0, 2.0, 3.0], [10.0, 14.0, 18.0]], dtype=np.float32)
    means = signals.mean(axis=1)
    scales = signals.std(axis=1)

    normalized = _record_zscore(signals, means, scales)

    np.testing.assert_allclose(normalized.mean(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(normalized * scales[:, None] + means[:, None], signals)


def test_record_zscore_rejects_zero_scale():
    with pytest.raises(ValueError, match="positive"):
        _record_zscore(
            np.ones((1, 4), dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
        )


def test_record_zscore_normalizes_each_record_and_lead_independently():
    signals = np.arange(48, dtype=np.float32).reshape(2, 3, 8)
    signals = signals * np.array([1.0, 2.0, 4.0], dtype=np.float32)[None, :, None]
    means = signals.mean(axis=-1)
    scales = signals.std(axis=-1)

    normalized = _record_zscore(signals, means, scales)

    np.testing.assert_allclose(normalized.mean(axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        normalized * scales[..., None] + means[..., None], signals, atol=1e-6
    )


def test_record_minmax_neg1_1_is_per_record_and_invertible():
    signals = np.array([[2.0, 4.0, 8.0], [-3.0, 1.0, 5.0]], dtype=np.float32)
    offsets, ranges = _record_minmax_coefficients(signals)
    normalized = _record_minmax_neg1_1(signals, offsets, ranges)

    np.testing.assert_allclose(normalized.min(axis=1), -1.0)
    np.testing.assert_allclose(normalized.max(axis=1), 1.0)
    restored = (normalized + 1.0) * ranges[:, None] / 2.0 + offsets[:, None]
    np.testing.assert_allclose(restored, signals)


def test_record_minmax_rejects_constant_record():
    with pytest.raises(ValueError, match="ranges must be finite and positive"):
        _record_minmax_coefficients(np.ones((2, 8), dtype=np.float32))


def test_rddm_window_minmax_matches_per_window_nan_to_num_contract():
    signals = np.array(
        [[np.nan, 2.0, 4.0, 8.0], [10.0, 12.0, 14.0, 16.0]], dtype=np.float32
    )

    normalized = _rddm_window_minmax_neg1_1(signals)

    np.testing.assert_allclose(normalized.min(axis=1), -1.0)
    np.testing.assert_allclose(normalized.max(axis=1), 1.0)
    assert normalized.dtype == np.float32


def test_rddm_ppg_loader_minmaxes_before_cleaning_and_omits_heldout_mask(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "synthetic"
    dataset_root.mkdir()
    ramp = np.arange(128, dtype=np.float32)
    for split, offset in (("train", 0.0), ("test", 10.0)):
        np.save(dataset_root / f"ecg_{split}_1sec.npy", (ramp + offset)[None, :])
        np.save(dataset_root / f"ppg_{split}_1sec.npy", (2 * ramp + offset + 1)[None, :])

    clean_inputs = []

    def fake_ecg_clean(values, sampling_rate, method):
        clean_inputs.append(("ecg", values.copy(), sampling_rate, method))
        return values

    def fake_ppg_clean(values, sampling_rate):
        clean_inputs.append(("ppg", values.copy(), sampling_rate, None))
        return values

    monkeypatch.setattr(data.nk, "ecg_clean", fake_ecg_clean)
    monkeypatch.setattr(data.nk, "ppg_clean", fake_ppg_clean)
    monkeypatch.setattr(
        data,
        "_ecg_region_mask",
        lambda target, sampling_rate: np.zeros((1, target.shape[-1]), dtype=np.float32),
    )

    train_set, heldout_set = get_ppg2ecg_datasets(
        DATA_PATH=str(tmp_path),
        datasets=["synthetic"],
        window_size=1,
        normalization_id="rddm_window_minmax_neg1_1_v1",
    )

    assert len(train_set[0]) == 3
    assert len(heldout_set[0]) == 2
    assert len(clean_inputs) == 4
    for _kind, values, sampling_rate, _method in clean_inputs:
        np.testing.assert_allclose(
            [values.min(), values.max()], [-1.0, 1.0], atol=2e-7
        )
        assert sampling_rate == 128
    assert {entry[3] for entry in clean_inputs if entry[0] == "ecg"} == {
        "pantompkins1985"
    }
    assert train_set.normalization_metadata["stats_scope"] == "per_window_per_modality"


def test_rddm_ppg_loader_rejects_residual_all_zero_condition(tmp_path):
    dataset_root = tmp_path / "synthetic"
    dataset_root.mkdir()
    np.save(dataset_root / "ecg_test_1sec.npy", np.arange(128, dtype=np.float32)[None, :])
    np.save(dataset_root / "ppg_test_1sec.npy", np.zeros((1, 128), dtype=np.float32))

    with pytest.raises(ValueError, match="all-zero PPG windows"):
        get_ppg2ecg_datasets(
            DATA_PATH=str(tmp_path),
            datasets=["synthetic"],
            window_size=1,
            normalization_metadata={
                "method": "rddm_window_minmax_neg1_1",
                "stats_scope": "per_window_per_modality",
                "stats_source": "selected_ecg_or_ppg_window",
                "feature_range": [-1.0, 1.0],
                "preprocessing_order": "nan_to_num_float32_then_minmax_then_neurokit_clean",
                "inverse_transform": "unavailable_after_per_window_scaling_and_neurokit_cleaning",
                "generated_inverse_policy": "normalized_domain_only",
            },
            normalization_id="rddm_window_minmax_neg1_1_v1",
            load_train=False,
        )


def test_generic_window_minmax_does_not_apply_modality_specific_cleaning(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "synthetic"
    dataset_root.mkdir()
    ramp = np.arange(128, dtype=np.float32)
    for split in ("train", "test"):
        np.save(dataset_root / f"ecg_{split}_1sec.npy", ramp[None, :])
        np.save(dataset_root / f"ppg_{split}_1sec.npy", (2 * ramp + 1)[None, :])

    monkeypatch.setattr(
        data.nk,
        "ecg_clean",
        lambda *_args, **_kwargs: pytest.fail("generic scaling must not clean ECG"),
    )
    monkeypatch.setattr(
        data.nk,
        "ppg_clean",
        lambda *_args, **_kwargs: pytest.fail("RCG must not be cleaned as PPG"),
    )
    monkeypatch.setattr(
        data,
        "_ecg_region_mask",
        lambda target, sampling_rate: np.zeros((1, target.shape[-1]), dtype=np.float32),
    )

    train_set, heldout_set = get_ppg2ecg_datasets(
        DATA_PATH=str(tmp_path),
        datasets=["synthetic"],
        window_size=1,
        normalization_id="window_minmax_neg1_1_v1",
    )

    for dataset in (train_set, heldout_set):
        target, condition = dataset[0][:2]
        np.testing.assert_allclose([target.min(), target.max()], [-1.0, 1.0])
        np.testing.assert_allclose([condition.min(), condition.max()], [-1.0, 1.0])
        assert dataset.normalization_metadata["method"] == "window_minmax_neg1_1"


def test_ppg_rcg_smoke_caps_are_applied_after_normalization(tmp_path, monkeypatch):
    dataset_root = tmp_path / "synthetic"
    dataset_root.mkdir()
    base = np.arange(128, dtype=np.float32)
    for split, count in (("train", 5), ("test", 4)):
        values = np.stack([base + index for index in range(count)])
        np.save(dataset_root / f"ecg_{split}_1sec.npy", values)
        np.save(dataset_root / f"ppg_{split}_1sec.npy", values * 2 + 1)
    monkeypatch.setattr(
        data,
        "_ecg_region_mask",
        lambda target, sampling_rate: np.zeros((1, target.shape[-1]), dtype=np.float32),
    )

    train_set, heldout_set = get_ppg2ecg_datasets(
        DATA_PATH=str(tmp_path),
        datasets=["synthetic"],
        window_size=1,
        normalization_id="window_minmax_neg1_1_v1",
        max_train_records=2,
        max_heldout_records=1,
    )

    assert len(train_set) == 2
    assert len(heldout_set) == 1


def test_normalization_rejects_nonfinite_training_or_evaluation_values():
    finite = np.array([[0.0, 1.0]], dtype=np.float32)
    nonfinite = np.array([[0.0, np.nan]], dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        _training_global_zscore(nonfinite, finite, finite, finite)
    with pytest.raises(ValueError, match="finite"):
        _training_global_zscore(finite, finite, nonfinite, finite)


def test_region_masks_are_cached_and_not_recomputed(monkeypatch):
    calls = []

    def fake_mask(target, sampling_rate):
        calls.append((target.copy(), sampling_rate))
        return np.ones((1, target.shape[-1]), dtype=np.float32)

    monkeypatch.setattr(data, "_ecg_region_mask", fake_mask)
    dataset = PairedSignalDataset(
        target_ecg=np.arange(12, dtype=np.float32).reshape(3, 4),
        condition_signal=np.zeros((3, 4), dtype=np.float32),
    )

    assert len(calls) == 3
    first = dataset[0]
    second = dataset[0]
    assert len(calls) == 3
    np.testing.assert_array_equal(first[2], second[2])


def test_evaluation_dataset_does_not_compute_or_return_target_mask(monkeypatch):
    monkeypatch.setattr(
        data,
        "_ecg_region_mask",
        lambda *_args, **_kwargs: pytest.fail("evaluation must not derive a target mask"),
    )
    dataset = PairedSignalDataset(
        target_ecg=np.ones((2, 4), dtype=np.float32),
        condition_signal=np.zeros((2, 4), dtype=np.float32),
        return_region_mask=False,
    )

    sample = dataset[0]
    assert len(sample) == 2


def test_ecg_loader_rejects_missing_held_out_split(tmp_path):
    dataset_root = tmp_path / "synthetic"
    dataset_root.mkdir()
    np.save(dataset_root / "X_train_resampled.npy", np.zeros((2, 128, 12), dtype=np.float32))

    with pytest.raises(FileNotFoundError, match="Missing held-out ECG split"):
        get_ecg2ecg_datasets(DATA_PATH=str(tmp_path), datasets=["synthetic"], window_size=1)


def test_inference_only_ecg_loader_skips_training_data_and_masks(tmp_path, monkeypatch):
    dataset_root = tmp_path / "synthetic"
    dataset_root.mkdir()
    held_out = np.zeros((2, 128, 12), dtype=np.float32)
    held_out[:, :, 2] = 12.0
    held_out[:, :, 10] = 24.0
    np.save(dataset_root / "X_val_resampled.npy", held_out)
    metadata = {
        "method": "training_global_zscore",
        "target_mean": 4.0,
        "target_scale": 2.0,
        "condition_mean": 2.0,
        "condition_scale": 5.0,
    }
    monkeypatch.setattr(
        data,
        "_ecg_region_mask",
        lambda *_args, **_kwargs: pytest.fail("inference must not derive target masks"),
    )

    train_set, test_set = get_ecg2ecg_datasets(
        DATA_PATH=str(tmp_path),
        datasets=["synthetic"],
        window_size=1,
        condition_lead=2,
        target_lead=10,
        normalization_metadata=metadata,
        load_train=False,
    )

    assert train_set is None
    target, condition = test_set[0]
    np.testing.assert_allclose(target, 10.0)
    np.testing.assert_allclose(condition, 2.0)


def test_ecg_loader_can_select_record_disjoint_test_split(tmp_path):
    dataset_root = tmp_path / "synthetic"
    dataset_root.mkdir()
    held_out = np.zeros((1, 128, 12), dtype=np.float32)
    held_out[:, :, 2] = 7.0
    held_out[:, :, 10] = 11.0
    np.save(dataset_root / "X_test_resampled.npy", held_out)
    metadata = {
        "method": "training_global_zscore",
        "target_mean": 1.0,
        "target_scale": 2.0,
        "condition_mean": 2.0,
        "condition_scale": 5.0,
    }

    _, test_set = get_ecg2ecg_datasets(
        DATA_PATH=str(tmp_path),
        datasets=["synthetic"],
        window_size=1,
        condition_lead=2,
        target_lead=10,
        normalization_metadata=metadata,
        load_train=False,
        heldout_split="test",
    )

    target, condition = test_set[0]
    np.testing.assert_allclose(target, 5.0)
    np.testing.assert_allclose(condition, 1.0)


def test_ecg_loader_uses_aligned_record_zscore_sidecars(tmp_path, monkeypatch):
    dataset_root = tmp_path / "synthetic"
    dataset_root.mkdir()
    ramp = np.arange(128, dtype=np.float32)
    train = np.zeros((2, 128, 12), dtype=np.float32)
    heldout = np.zeros((1, 128, 12), dtype=np.float32)
    train[0, :, 2], train[1, :, 2] = ramp, 2 * ramp + 10
    train[0, :, 10], train[1, :, 10] = 3 * ramp - 4, 4 * ramp + 20
    heldout[0, :, 2], heldout[0, :, 10] = 5 * ramp + 30, 6 * ramp - 12
    np.save(dataset_root / "X_train_resampled.npy", train)
    np.save(dataset_root / "X_val_resampled.npy", heldout)
    for split, values, ids in (
        ("train", train, np.array(["A0001", "A0002"])),
        ("val", heldout, np.array(["A0003"])),
    ):
        np.save(dataset_root / f"record_ids_{split}.npy", ids)
        np.save(dataset_root / f"record_means_{split}.npy", values.mean(axis=1))
        np.save(dataset_root / f"record_scales_{split}.npy", values.std(axis=1))
    monkeypatch.setattr(
        data,
        "_ecg_region_mask",
        lambda target, sampling_rate: np.zeros((1, target.shape[-1]), dtype=np.float32),
    )

    train_set, heldout_set = get_ecg2ecg_datasets(
        DATA_PATH=str(tmp_path),
        datasets=["synthetic"],
        window_size=1,
        condition_lead=2,
        target_lead=10,
        normalization_id="record_zscore_v1",
    )

    np.testing.assert_allclose(train_set.target_ecg.mean(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(train_set.target_ecg.std(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(heldout_set.record_ids, ["A0003"])
    restored = (
        heldout_set.target_ecg * heldout_set.target_scales[:, None]
        + heldout_set.target_means[:, None]
    )
    np.testing.assert_allclose(restored, heldout[:, :, 10], atol=1e-5)
    assert heldout_set.normalization_metadata["method"] == "record_zscore"


def test_ecg_loader_applies_record_minmax_neg1_1_and_keeps_coefficients(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "synthetic"
    dataset_root.mkdir()
    ramp = np.arange(128, dtype=np.float32)
    train = np.zeros((2, 128, 12), dtype=np.float32)
    heldout = np.zeros((1, 128, 12), dtype=np.float32)
    train[0, :, 2], train[1, :, 2] = ramp, 2 * ramp + 10
    train[0, :, 10], train[1, :, 10] = 3 * ramp - 4, 4 * ramp + 20
    heldout[0, :, 2], heldout[0, :, 10] = 5 * ramp + 30, 6 * ramp - 12
    np.save(dataset_root / "X_train_resampled.npy", train)
    np.save(dataset_root / "X_val_resampled.npy", heldout)
    np.save(dataset_root / "record_ids_train.npy", np.array(["A0001", "A0002"]))
    np.save(dataset_root / "record_ids_val.npy", np.array(["A0003"]))
    monkeypatch.setattr(
        data,
        "_ecg_region_mask",
        lambda target, sampling_rate: np.zeros((1, target.shape[-1]), dtype=np.float32),
    )

    train_set, heldout_set = get_ecg2ecg_datasets(
        DATA_PATH=str(tmp_path),
        datasets=["synthetic"],
        window_size=1,
        condition_lead=2,
        target_lead=10,
        normalization_id="record_minmax_neg1_1_v1",
    )

    np.testing.assert_allclose(train_set.target_ecg.min(axis=1), -1.0)
    np.testing.assert_allclose(train_set.target_ecg.max(axis=1), 1.0)
    np.testing.assert_array_equal(heldout_set.record_ids, ["A0003"])
    restored = (
        (heldout_set.target_ecg + 1.0) * heldout_set.target_scales[:, None] / 2.0
        + heldout_set.target_offsets[:, None]
    )
    np.testing.assert_allclose(restored, heldout[:, :, 10], atol=5e-5)
    assert heldout_set.normalization_metadata["method"] == "record_minmax_neg1_1"


def test_ecg_loader_jointly_returns_lead_ii_to_other_eleven_with_per_lead_masks(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "synthetic"
    dataset_root.mkdir()
    ramp = np.arange(128, dtype=np.float32)
    train = np.empty((2, 128, 12), dtype=np.float32)
    heldout = np.empty((1, 128, 12), dtype=np.float32)
    for lead in range(12):
        train[0, :, lead] = (lead + 1) * ramp + 10 * lead
        train[1, :, lead] = (lead + 2) * ramp - 5 * lead
        heldout[0, :, lead] = (lead + 3) * ramp + 2 * lead
    np.save(dataset_root / "X_train_resampled.npy", train)
    np.save(dataset_root / "X_val_resampled.npy", heldout)
    np.save(dataset_root / "record_ids_train.npy", np.array(["A0001", "A0002"]))
    np.save(dataset_root / "record_ids_val.npy", np.array(["A0003"]))
    mask_calls = []

    def fake_mask(channel, sampling_rate):
        mask_calls.append((channel.copy(), sampling_rate))
        return np.ones((1, channel.shape[-1]), dtype=np.float32)

    monkeypatch.setattr(data, "_ecg_region_mask", fake_mask)
    target_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    train_set, heldout_set = get_ecg2ecg_datasets(
        DATA_PATH=str(tmp_path),
        datasets=["synthetic"],
        window_size=1,
        condition_lead=1,
        target_lead=target_indices,
        normalization_id="record_minmax_neg1_1_v1",
    )

    target, condition, mask = train_set[0]
    assert target.shape == (11, 128)
    assert condition.shape == (1, 128)
    assert mask.shape == (11, 128)
    assert len(mask_calls) == len(train_set) * 11
    np.testing.assert_allclose(train_set.target_ecg.min(axis=-1), -1.0)
    np.testing.assert_allclose(train_set.target_ecg.max(axis=-1), 1.0)
    np.testing.assert_allclose(train_set.condition_signal.min(axis=-1), -1.0)
    np.testing.assert_allclose(train_set.condition_signal.max(axis=-1), 1.0)
    assert len(heldout_set[0]) == 2
    assert heldout_set[0][0].shape == (11, 128)
    restored = (
        (heldout_set.target_ecg + 1.0) * heldout_set.target_scales[..., None] / 2.0
        + heldout_set.target_offsets[..., None]
    )
    np.testing.assert_allclose(
        restored, np.transpose(heldout[:, :, target_indices], (0, 2, 1)), atol=1e-4
    )


def test_ecg_loader_joint_target_record_zscore_sidecars_have_eleven_columns(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "synthetic"
    dataset_root.mkdir()
    ramp = np.arange(128, dtype=np.float32)
    train = np.stack([(lead + 1) * ramp for lead in range(12)], axis=-1)[None]
    heldout = train + np.arange(12, dtype=np.float32)[None, None, :]
    np.save(dataset_root / "X_train_resampled.npy", train)
    np.save(dataset_root / "X_val_resampled.npy", heldout)
    for split, values, record_id in (
        ("train", train, "A0001"),
        ("val", heldout, "A0002"),
    ):
        np.save(dataset_root / f"record_ids_{split}.npy", np.array([record_id]))
        np.save(dataset_root / f"record_means_{split}.npy", values.mean(axis=1))
        np.save(dataset_root / f"record_scales_{split}.npy", values.std(axis=1))
    monkeypatch.setattr(
        data,
        "_ecg_region_mask",
        lambda channel, sampling_rate: np.zeros((1, channel.shape[-1]), dtype=np.float32),
    )
    target_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    train_set, heldout_set = get_ecg2ecg_datasets(
        DATA_PATH=str(tmp_path),
        datasets=["synthetic"],
        window_size=1,
        condition_lead=1,
        target_lead=target_indices,
        normalization_id="record_zscore_v1",
    )

    assert train_set.target_means.shape == (1, 11)
    assert train_set.target_scales.shape == (1, 11)
    np.testing.assert_allclose(train_set.target_ecg.mean(axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(train_set.target_ecg.std(axis=-1), 1.0, atol=2e-6)
    assert len(heldout_set[0]) == 2


def test_ecg_loader_rejects_condition_lead_in_joint_targets(tmp_path):
    with pytest.raises(ValueError, match="condition lead"):
        get_ecg2ecg_datasets(
            DATA_PATH=str(tmp_path),
            condition_lead=1,
            target_lead=[0, 1, 2],
        )


def test_ecg_smoke_caps_apply_after_full_train_normalization(tmp_path, monkeypatch):
    dataset_root = tmp_path / "synthetic"
    dataset_root.mkdir()
    train = np.zeros((4, 128, 12), dtype=np.float32)
    train[:, :, 2] = np.arange(4, dtype=np.float32)[:, None]
    train[:, :, 10] = (2 * np.arange(4, dtype=np.float32))[:, None]
    heldout = np.zeros((3, 128, 12), dtype=np.float32)
    np.save(dataset_root / "X_train_resampled.npy", train)
    np.save(dataset_root / "X_val_resampled.npy", heldout)
    monkeypatch.setattr(
        data,
        "_ecg_region_mask",
        lambda target, sampling_rate: np.zeros((1, target.shape[-1]), dtype=np.float32),
    )

    train_set, heldout_set = get_ecg2ecg_datasets(
        DATA_PATH=str(tmp_path),
        datasets=["synthetic"],
        window_size=1,
        condition_lead=2,
        target_lead=10,
        max_train_records=1,
        max_heldout_records=2,
    )

    assert len(train_set) == 1
    assert len(heldout_set) == 2
    assert train_set.normalization_metadata["condition_mean"] == pytest.approx(1.5)
    assert train_set.normalization_metadata["target_mean"] == pytest.approx(3.0)


def test_ecg_loader_broadcasts_one_temporal_cached_mask_to_all_target_leads(tmp_path):
    dataset_root = tmp_path / "synthetic"
    dataset_root.mkdir()
    rng = np.random.default_rng(17)
    np.save(dataset_root / "X_train_resampled.npy", rng.normal(size=(3, 128, 12)).astype(np.float32))
    np.save(dataset_root / "X_val_resampled.npy", rng.normal(size=(2, 128, 12)).astype(np.float32))
    masks = np.linspace(0, 1, 3 * 128, dtype=np.float32).reshape(3, 1, 128)

    train_set, heldout_set = get_ecg2ecg_datasets(
        DATA_PATH=str(tmp_path), datasets=["synthetic"], window_size=1,
        condition_lead=1, target_lead=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        region_masks_train=masks, max_train_records=2,
    )

    returned = train_set[0][2]
    assert returned.shape == (11, 128)
    np.testing.assert_array_equal(returned[0], masks[0, 0])
    np.testing.assert_array_equal(returned[-1], masks[0, 0])
    assert len(heldout_set[0]) == 2


@pytest.mark.parametrize("field", ["max_train_records", "max_heldout_records"])
def test_ecg_smoke_caps_must_be_positive(tmp_path, field):
    kwargs = {field: 0}
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        get_ecg2ecg_datasets(DATA_PATH=str(tmp_path), **kwargs)
