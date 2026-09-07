"""Dataset helpers for paired physiological signal generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import neurokit2 as nk
import numpy as np
from sklearn.preprocessing import minmax_scale
from torch.utils.data import Dataset


SAMPLE_RATE = 128


def _as_2d_windows(array: np.ndarray, samples: int) -> np.ndarray:
    return np.asarray(array, dtype=np.float32).reshape(-1, samples)


def _select_ecg_leads(
    array: np.ndarray,
    samples: int,
    lead_indices: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("ECG arrays must have shape (records, samples, leads)")
    if not lead_indices or len(set(lead_indices)) != len(lead_indices):
        raise ValueError("ECG lead indices must be nonempty and unique")
    if min(lead_indices) < 0 or max(lead_indices) >= values.shape[2]:
        raise ValueError("ECG lead index is outside the waveform array")
    selected = np.transpose(values[:, :samples, list(lead_indices)], (0, 2, 1))
    return selected[:, 0] if len(lead_indices) == 1 else selected


def _training_global_zscore(
    train_target: np.ndarray,
    train_condition: np.ndarray,
    test_target: np.ndarray,
    test_condition: np.ndarray,
    metadata: dict[str, float | str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | str]]:
    if metadata is None:
        if not np.all(np.isfinite(train_target)) or not np.all(np.isfinite(train_condition)):
            raise ValueError("training signals must contain only finite values")
        target_mean = float(np.mean(train_target))
        target_scale = float(np.std(train_target))
        condition_mean = float(np.mean(train_condition))
        condition_scale = float(np.std(train_condition))
        if target_scale <= 0 or condition_scale <= 0:
            raise ValueError("training signals must have nonzero global standard deviation")
        metadata = {
            "method": "training_global_zscore",
            "target_mean": target_mean,
            "target_scale": target_scale,
            "condition_mean": condition_mean,
            "condition_scale": condition_scale,
        }
    fitted_metadata = _validate_normalization_metadata(metadata)
    normalized_train_target, normalized_train_condition = _normalize_pair(
        train_target, train_condition, fitted_metadata
    )
    normalized_test_target, normalized_test_condition = _normalize_pair(
        test_target, test_condition, fitted_metadata
    )
    return (
        normalized_train_target,
        normalized_train_condition,
        normalized_test_target,
        normalized_test_condition,
        fitted_metadata,
    )


def _validate_normalization_metadata(
    metadata: Mapping[str, float | str],
) -> dict[str, float | str]:
    required = {
        "method",
        "target_mean",
        "target_scale",
        "condition_mean",
        "condition_scale",
    }
    missing = required - set(metadata)
    if missing or metadata["method"] != "training_global_zscore":
        raise ValueError(f"invalid training_global_zscore metadata; missing={sorted(missing)}")
    numeric_values = [
        float(metadata["target_mean"]),
        float(metadata["target_scale"]),
        float(metadata["condition_mean"]),
        float(metadata["condition_scale"]),
    ]
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("normalization metadata must contain only finite values")
    target_scale = float(metadata["target_scale"])
    condition_scale = float(metadata["condition_scale"])
    if target_scale <= 0 or condition_scale <= 0:
        raise ValueError("normalization scales must be positive")
    return dict(metadata)


def _normalize_pair(
    target: np.ndarray,
    condition: np.ndarray,
    metadata: Mapping[str, float | str],
) -> tuple[np.ndarray, np.ndarray]:
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(condition)):
        raise ValueError("signals must contain only finite values")
    target_mean = float(metadata["target_mean"])
    target_scale = float(metadata["target_scale"])
    condition_mean = float(metadata["condition_mean"])
    condition_scale = float(metadata["condition_scale"])
    return (
        ((target - target_mean) / target_scale).astype(np.float32),
        ((condition - condition_mean) / condition_scale).astype(np.float32),
    )


def _rddm_window_minmax_neg1_1(signals: np.ndarray) -> np.ndarray:
    """Match RDDM's float32/nan-to-num/per-window min-max preprocessing."""

    values = np.nan_to_num(np.asarray(signals, dtype=np.float32))
    if values.ndim != 2:
        raise ValueError("RDDM window min-max expects a 2D window array")
    if not np.all(np.isfinite(values)):
        raise ValueError("RDDM window min-max inputs must be finite after nan_to_num")
    return np.asarray(minmax_scale(values, (-1, 1), axis=1), dtype=np.float32)


def _rddm_window_minmax_metadata(
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    expected = {
        "method": "rddm_window_minmax_neg1_1",
        "stats_scope": "per_window_per_modality",
        "stats_source": "selected_ecg_or_ppg_window",
        "feature_range": [-1.0, 1.0],
        "preprocessing_order": "nan_to_num_float32_then_minmax_then_neurokit_clean",
        "inverse_transform": "unavailable_after_per_window_scaling_and_neurokit_cleaning",
        "generated_inverse_policy": "normalized_domain_only",
    }
    if metadata is not None:
        mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatched:
            raise ValueError(f"invalid RDDM window min-max metadata fields: {mismatched}")
    return expected


def _window_minmax_neg1_1_metadata(
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Describe modality-agnostic per-window scaling without signal cleaning."""

    expected = {
        "method": "window_minmax_neg1_1",
        "stats_scope": "per_window_per_modality",
        "stats_source": "selected_ecg_or_condition_window",
        "feature_range": [-1.0, 1.0],
        "preprocessing_order": "finite_float32_then_minmax_no_signal_cleaning",
        "inverse_transform": "unavailable_without_saved_per_window_minima_and_ranges",
        "generated_inverse_policy": "normalized_domain_only",
    }
    if metadata is not None:
        mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatched:
            raise ValueError(f"invalid window min-max metadata fields: {mismatched}")
    return expected


def _source_record_minmax_neg1_1_metadata(
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Describe reversible scaling fixed over each continuous source record."""

    expected = {
        "method": "source_record_minmax_neg1_1",
        "stats_scope": "per_source_continuous_record_per_modality",
        "stats_source": "dataset_sidecars_computed_before_window_split",
        "feature_range": [-1.0, 1.0],
        "inverse_transform": "x=(x_scaled+1)*source_record_range/2+source_record_min",
        "generated_inverse_policy": "ground_truth_target_scaler_is_oracle_only",
        "cross_modality_scaler_shared": False,
    }
    if metadata is not None:
        mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatched:
            raise ValueError(
                f"invalid source-record min-max metadata fields: {mismatched}"
            )
    return expected


def _record_zscore(
    signals: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Apply reversible per-record z-score normalization."""

    signals = np.asarray(signals, dtype=np.float32)
    means = np.asarray(means, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    expected_shape = signals.shape[:-1]
    if signals.ndim not in {2, 3} or means.shape != expected_shape or scales.shape != expected_shape:
        raise ValueError("record z-score signals and coefficients must align by record")
    if not np.all(np.isfinite(signals)) or not np.all(np.isfinite(means)):
        raise ValueError("record z-score inputs must be finite")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("record z-score scales must be finite and positive")
    return ((signals - means[..., None]) / scales[..., None]).astype(np.float32)


def _record_zscore_metadata(
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    expected = {
        "method": "record_zscore",
        "stats_scope": "per_record_per_lead",
        "stats_source": "dataset_sidecar",
        "inverse_transform": "x=x_z*record_scale+record_mean",
        "generated_inverse_policy": "ground_truth_target_scaler_is_oracle_only",
    }
    if metadata is not None:
        mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatched:
            raise ValueError(f"invalid record_zscore metadata fields: {mismatched}")
    return expected


def _record_minmax_coefficients(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-record minima and ranges for reversible scaling."""

    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim not in {2, 3} or not np.all(np.isfinite(signals)):
        raise ValueError("record min-max signals must be a finite 2D or 3D array")
    offsets = signals.min(axis=-1)
    ranges = signals.max(axis=-1) - offsets
    if np.any(ranges <= 0) or not np.all(np.isfinite(ranges)):
        raise ValueError("record min-max ranges must be finite and positive")
    return offsets.astype(np.float32), ranges.astype(np.float32)


def _record_minmax_neg1_1(
    signals: np.ndarray,
    offsets: np.ndarray,
    ranges: np.ndarray,
) -> np.ndarray:
    """Scale each record independently to [-1, 1]."""

    signals = np.asarray(signals, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    ranges = np.asarray(ranges, dtype=np.float32)
    expected_shape = signals.shape[:-1]
    if signals.ndim not in {2, 3} or offsets.shape != expected_shape or ranges.shape != expected_shape:
        raise ValueError("record min-max signals and coefficients must align by record")
    if not np.all(np.isfinite(signals)) or not np.all(np.isfinite(offsets)):
        raise ValueError("record min-max inputs must be finite")
    if not np.all(np.isfinite(ranges)) or np.any(ranges <= 0):
        raise ValueError("record min-max ranges must be finite and positive")
    return (2.0 * (signals - offsets[..., None]) / ranges[..., None] - 1.0).astype(np.float32)


def _record_minmax_neg1_1_metadata(
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    expected = {
        "method": "record_minmax_neg1_1",
        "stats_scope": "per_record_per_lead",
        "stats_source": "selected_waveform_record",
        "feature_range": [-1.0, 1.0],
        "inverse_transform": "x=(x_scaled+1)*record_range/2+record_min",
        "generated_inverse_policy": "ground_truth_target_scaler_is_oracle_only",
    }
    if metadata is not None:
        mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatched:
            raise ValueError(f"invalid record_minmax_neg1_1 metadata fields: {mismatched}")
    return expected


def _record_joint12_minmax_neg1_1_metadata(
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    expected = {
        "method": "record_joint12_minmax_neg1_1",
        "stats_scope": "per_record_shared_all_12_leads",
        "stats_source": "dataset_sidecar_fixed_first_model_window",
        "feature_range": [-1.0, 1.0],
        "inverse_transform": "x=(x_scaled+1)*record_joint_range/2+record_joint_min",
        "preserves_interlead_relative_amplitudes_and_offsets": True,
        "heldout_target_statistics_used": True,
        "deployment_scope": "paired_benchmark_only_not_lead_II_only_inference",
        "generated_inverse_policy": "ground_truth_joint_12lead_scaler_is_oracle_only",
    }
    if metadata is not None:
        mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatched:
            raise ValueError(f"invalid record_joint12_minmax metadata fields: {mismatched}")
    return expected


def _source_record_joint12_minmax_neg1_1_metadata(
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    expected = {
        "method": "source_record_joint12_minmax_neg1_1",
        "stats_scope": "per_source_record_shared_full_10s_all_12_leads",
        "stats_source": "dataset_sidecar_full_source_record",
        "feature_range": [-1.0, 1.0],
        "inverse_transform": "x=(x_scaled+1)*source_record_joint_range/2+source_record_joint_min",
        "preserves_interlead_relative_amplitudes_and_offsets": True,
        "preserves_within_record_interwindow_scale": True,
        "heldout_target_statistics_used": True,
        "deployment_scope": "paired_benchmark_only_not_lead_II_only_inference",
        "generated_inverse_policy": "ground_truth_full_record_joint_12lead_scaler_is_oracle_only",
    }
    if metadata is not None:
        mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatched:
            raise ValueError(f"invalid source-record joint12 minmax metadata fields: {mismatched}")
    return expected


def _load_record_joint12_coefficients(
    dataset_root: Path, split: str, expected_records: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids_path = dataset_root / f"record_ids_{split}.npy"
    minima_path = dataset_root / f"record_joint_minima_{split}.npy"
    ranges_path = dataset_root / f"record_joint_ranges_{split}.npy"
    missing = [path.name for path in (ids_path, minima_path, ranges_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"joint12 min-max normalization requires sidecars: {missing}")
    record_ids = np.load(ids_path, allow_pickle=False).reshape(-1)
    minima = np.asarray(np.load(minima_path, allow_pickle=False), dtype=np.float32).reshape(-1)
    ranges = np.asarray(np.load(ranges_path, allow_pickle=False), dtype=np.float32).reshape(-1)
    if len(record_ids) != expected_records or minima.shape != (expected_records,):
        raise ValueError("joint 12-lead min-max IDs/minima do not align with waveform split")
    if ranges.shape != minima.shape or not np.all(np.isfinite(minima)):
        raise ValueError("joint 12-lead min-max coefficients do not align")
    if not np.all(np.isfinite(ranges)) or np.any(ranges <= 0):
        raise ValueError("joint 12-lead min-max ranges must be finite and positive")
    return record_ids.astype(str), minima, ranges


def _load_record_ids(dataset_root: Path, split: str, expected_records: int) -> np.ndarray:
    ids_path = dataset_root / f"record_ids_{split}.npy"
    if not ids_path.exists():
        raise FileNotFoundError(f"record normalization requires {ids_path.name}")
    record_ids = np.load(ids_path, allow_pickle=False).reshape(-1)
    if len(record_ids) != expected_records:
        raise ValueError("record IDs do not align with the waveform split")
    return record_ids.astype(str)


def _load_record_coefficients(
    dataset_root: Path,
    split: str,
    expected_records: int,
    condition_lead: int,
    target_lead: int | tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids_path = dataset_root / f"record_ids_{split}.npy"
    means_path = dataset_root / f"record_means_{split}.npy"
    scales_path = dataset_root / f"record_scales_{split}.npy"
    missing = [path.name for path in (ids_path, means_path, scales_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"record_zscore_v1 requires dataset sidecars: {missing}")
    record_ids = np.load(ids_path, allow_pickle=False).reshape(-1)
    means = np.asarray(np.load(means_path, allow_pickle=False), dtype=np.float32)
    scales = np.asarray(np.load(scales_path, allow_pickle=False), dtype=np.float32)
    if len(record_ids) != expected_records or means.shape != (expected_records, 12):
        raise ValueError("record z-score means/IDs do not align with the waveform split")
    if scales.shape != means.shape:
        raise ValueError("record z-score scales do not align with means")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(scales)):
        raise ValueError("record z-score sidecars must be finite")
    target_leads = (target_lead,) if isinstance(target_lead, int) else target_lead
    selected_scales = scales[:, (condition_lead, *target_leads)]
    if np.any(selected_scales <= 0):
        raise ValueError("selected record z-score scales must be positive")
    return (
        record_ids.astype(str),
        means[:, condition_lead],
        scales[:, condition_lead],
        means[:, target_leads[0]] if len(target_leads) == 1 else means[:, target_leads],
        scales[:, target_leads[0]] if len(target_leads) == 1 else scales[:, target_leads],
    )


def _ecg_region_mask(ecg: np.ndarray, sampling_rate: int = SAMPLE_RATE, roi_size: int = 32) -> np.ndarray:
    window_size = ecg.shape[-1]
    mask = np.zeros((1, window_size), dtype=np.float32)
    try:
        _, info = nk.ecg_peaks(
            ecg.reshape(window_size),
            sampling_rate=sampling_rate,
            method="pantompkins1985",
            correct_artifacts=True,
            show=False,
        )
        peaks = info.get("ECG_R_Peaks", [])
    except Exception:
        peaks = []

    for peak in peaks:
        start = max(0, int(peak) - roi_size // 2)
        end = min(start + roi_size, window_size)
        mask[0, start:end] = 1.0
    return mask


class PairedSignalDataset(Dataset):
    """Return paired signals and, for training sets, a cached ECG region mask."""

    def __init__(
        self,
        target_ecg: np.ndarray,
        condition_signal: np.ndarray,
        clean_target: bool = False,
        clean_condition_ppg: bool = False,
        normalization_metadata: dict[str, object] | None = None,
        return_region_mask: bool = True,
        cached_region_masks: np.ndarray | None = None,
        record_ids: np.ndarray | None = None,
        target_means: np.ndarray | None = None,
        target_scales: np.ndarray | None = None,
        condition_means: np.ndarray | None = None,
        condition_scales: np.ndarray | None = None,
        target_offsets: np.ndarray | None = None,
        condition_offsets: np.ndarray | None = None,
    ) -> None:
        if len(target_ecg) != len(condition_signal):
            raise ValueError("target_ecg and condition_signal must have the same number of windows")
        self.target_ecg = np.asarray(target_ecg, dtype=np.float32)
        self.condition_signal = np.asarray(condition_signal, dtype=np.float32)
        self.normalization_metadata = normalization_metadata
        self.return_region_mask = return_region_mask
        if cached_region_masks is not None and not return_region_mask:
            raise ValueError("cached region masks require return_region_mask=True")
        zscore_values = (
            record_ids,
            target_means,
            target_scales,
            condition_means,
            condition_scales,
        )
        minmax_values = (
            record_ids,
            target_offsets,
            target_scales,
            condition_offsets,
            condition_scales,
        )
        has_zscore = any(value is not None for value in (target_means, condition_means))
        has_minmax = any(value is not None for value in (target_offsets, condition_offsets))
        if has_zscore and has_minmax:
            raise ValueError("z-score and min-max record coefficients are mutually exclusive")
        coefficient_values = zscore_values if has_zscore else minmax_values
        if any(value is not None for value in coefficient_values) and not all(
            value is not None for value in coefficient_values
        ):
            raise ValueError("record IDs and all normalization coefficients must be supplied together")
        self.record_ids = None if record_ids is None else np.asarray(record_ids).reshape(-1)
        self.target_means = None if target_means is None else np.asarray(target_means, dtype=np.float32)
        self.target_scales = None if target_scales is None else np.asarray(target_scales, dtype=np.float32)
        self.condition_means = (
            None if condition_means is None else np.asarray(condition_means, dtype=np.float32)
        )
        self.condition_scales = (
            None if condition_scales is None else np.asarray(condition_scales, dtype=np.float32)
        )
        self.target_offsets = (
            None if target_offsets is None else np.asarray(target_offsets, dtype=np.float32)
        )
        self.condition_offsets = (
            None if condition_offsets is None else np.asarray(condition_offsets, dtype=np.float32)
        )
        for values in (
            self.record_ids,
            self.target_means,
            self.target_scales,
            self.condition_means,
            self.condition_scales,
            self.target_offsets,
            self.condition_offsets,
        ):
            if values is not None and len(values) != len(self.target_ecg):
                raise ValueError("record metadata must align with paired signals")

        if clean_target:
            if self.target_ecg.ndim != 2:
                raise ValueError("NeuroKit target cleaning only supports single-channel ECG")
            self.target_ecg = np.stack(
                [
                    nk.ecg_clean(
                        target.reshape(-1),
                        sampling_rate=SAMPLE_RATE,
                        method="pantompkins1985",
                    )
                    for target in self.target_ecg
                ]
            ).astype(np.float32)
        if clean_condition_ppg:
            self.condition_signal = np.stack(
                [
                    nk.ppg_clean(condition.reshape(-1), sampling_rate=SAMPLE_RATE)
                    for condition in self.condition_signal
                ]
            ).astype(np.float32)

        self.region_masks = None
        if cached_region_masks is not None:
            masks = np.asarray(cached_region_masks, dtype=np.float32)
            if masks.ndim == 2:
                masks = masks[:, None, :]
            target_channels = 1 if self.target_ecg.ndim == 2 else self.target_ecg.shape[1]
            expected_shapes = {
                (len(self.target_ecg), 1, self.target_ecg.shape[-1]),
                (len(self.target_ecg), target_channels, self.target_ecg.shape[-1]),
            }
            if masks.shape not in expected_shapes:
                raise ValueError(
                    f"cached region masks have shape {masks.shape}; expected one of "
                    f"{sorted(expected_shapes)}"
                )
            if not np.all(np.isfinite(masks)):
                raise ValueError("cached region masks must contain only finite values")
            if np.any(masks < 0) or np.any(masks > 1):
                raise ValueError("cached region masks must be within [0, 1]")
            self.region_masks = np.broadcast_to(
                masks, (len(self.target_ecg), target_channels, self.target_ecg.shape[-1])
            )
        elif return_region_mask:
            masks = []
            for target in self.target_ecg:
                target_channels = target[None, :] if target.ndim == 1 else target
                masks.append(
                    np.concatenate(
                        [
                            _ecg_region_mask(channel, sampling_rate=SAMPLE_RATE)
                            for channel in target_channels
                        ],
                        axis=0,
                    )
                )
            self.region_masks = np.stack(masks)

    def __getitem__(self, index: int):
        target = self.target_ecg[index].astype(np.float32)
        condition = self.condition_signal[index].astype(np.float32)
        window_size = target.shape[-1]
        target = target.reshape(1, window_size) if target.ndim == 1 else target
        condition = condition.reshape(1, window_size) if condition.ndim == 1 else condition
        paired = (target.copy(), condition.copy())
        if self.region_masks is None:
            return paired
        return (*paired, self.region_masks[index].copy())

    def __len__(self) -> int:
        return len(self.target_ecg)


def get_ppg2ecg_datasets(
    DATA_PATH: str = "/data/user/RCFM/data/",
    datasets: Iterable[str] = ("MIMIC-AFib",),
    window_size: int = 4,
    clean_condition_ppg: bool = False,
    normalization_metadata: dict[str, object] | None = None,
    normalization_id: str = "training_global_zscore_v1",
    load_train: bool = True,
    max_train_records: int | None = None,
    max_heldout_records: int | None = None,
    return_region_mask_train: bool = True,
    region_masks_train: np.ndarray | None = None,
):
    """Load paired PPG/RCG-to-ECG windows.

    The mmECG preprocessed RCG arrays are stored with the historical ``ppg_*``
    file names, so the same loader is used for both PPG-to-ECG and RCG-to-ECG.
    """

    if region_masks_train is not None and not load_train:
        raise ValueError("training region masks cannot be supplied for inference-only loading")
    if region_masks_train is not None and not return_region_mask_train:
        raise ValueError("training region masks require return_region_mask_train=True")

    samples = SAMPLE_RATE * window_size
    root = Path(DATA_PATH)
    train_ecg, train_condition, test_ecg, test_condition = [], [], [], []
    train_target_minima, train_target_ranges = [], []
    train_condition_minima, train_condition_ranges = [], []
    test_target_minima, test_target_ranges = [], []
    test_condition_minima, test_condition_ranges = [], []
    train_source_record_ids, test_source_record_ids = [], []

    for dataset in datasets:
        dataset_root = root / dataset
        if load_train:
            train_ecg.append(_as_2d_windows(np.load(dataset_root / f"ecg_train_{window_size}sec.npy"), samples))
            train_condition.append(_as_2d_windows(np.load(dataset_root / f"ppg_train_{window_size}sec.npy"), samples))
        test_ecg.append(_as_2d_windows(np.load(dataset_root / f"ecg_test_{window_size}sec.npy"), samples))
        test_condition.append(_as_2d_windows(np.load(dataset_root / f"ppg_test_{window_size}sec.npy"), samples))
        if normalization_id == "source_record_minmax_neg1_1_v1":
            if load_train:
                train_source_record_ids.append(
                    np.load(dataset_root / "subject_ids_train.npy")
                )
                train_target_minima.append(
                    np.load(dataset_root / "target_record_minima_train.npy")
                )
                train_target_ranges.append(
                    np.load(dataset_root / "target_record_ranges_train.npy")
                )
                train_condition_minima.append(
                    np.load(dataset_root / "condition_record_minima_train.npy")
                )
                train_condition_ranges.append(
                    np.load(dataset_root / "condition_record_ranges_train.npy")
                )
            test_target_minima.append(
                np.load(dataset_root / "target_record_minima_test.npy")
            )
            test_source_record_ids.append(
                np.load(dataset_root / "subject_ids_test.npy")
            )
            test_target_ranges.append(
                np.load(dataset_root / "target_record_ranges_test.npy")
            )
            test_condition_minima.append(
                np.load(dataset_root / "condition_record_minima_test.npy")
            )
            test_condition_ranges.append(
                np.load(dataset_root / "condition_record_ranges_test.npy")
            )

    concatenated_test_target = np.concatenate(test_ecg)
    concatenated_test_source = np.concatenate(test_condition)
    if normalization_id == "source_record_minmax_neg1_1_v1":
        fitted_metadata = _source_record_minmax_neg1_1_metadata(normalization_metadata)
        test_target = _record_minmax_neg1_1(
            concatenated_test_target,
            np.concatenate(test_target_minima),
            np.concatenate(test_target_ranges),
        )
        test_source = _record_minmax_neg1_1(
            concatenated_test_source,
            np.concatenate(test_condition_minima),
            np.concatenate(test_condition_ranges),
        )
        if load_train:
            train_record_kwargs = {
                "record_ids": np.concatenate(train_source_record_ids),
                "target_offsets": np.concatenate(train_target_minima),
                "target_scales": np.concatenate(train_target_ranges),
                "condition_offsets": np.concatenate(train_condition_minima),
                "condition_scales": np.concatenate(train_condition_ranges),
            }
            train_target = _record_minmax_neg1_1(
                np.concatenate(train_ecg),
                train_record_kwargs["target_offsets"],
                train_record_kwargs["target_scales"],
            )
            train_source = _record_minmax_neg1_1(
                np.concatenate(train_condition),
                train_record_kwargs["condition_offsets"],
                train_record_kwargs["condition_scales"],
            )
            train_masks = region_masks_train
            if max_train_records is not None:
                train_target = train_target[:max_train_records]
                train_source = train_source[:max_train_records]
                train_record_kwargs = {
                    key: values[:max_train_records]
                    for key, values in train_record_kwargs.items()
                }
                if train_masks is not None:
                    train_masks = train_masks[:max_train_records]
            train_set = PairedSignalDataset(
                train_target,
                train_source,
                clean_target=False,
                clean_condition_ppg=clean_condition_ppg,
                normalization_metadata=fitted_metadata,
                return_region_mask=return_region_mask_train,
                cached_region_masks=train_masks,
                **train_record_kwargs,
            )
        else:
            train_set = None
        if max_heldout_records is not None:
            test_target = test_target[:max_heldout_records]
            test_source = test_source[:max_heldout_records]
        test_set = PairedSignalDataset(
            test_target,
            test_source,
            clean_target=False,
            clean_condition_ppg=clean_condition_ppg,
            normalization_metadata=fitted_metadata,
            return_region_mask=False,
            record_ids=np.concatenate(test_source_record_ids)[:max_heldout_records],
            target_offsets=np.concatenate(test_target_minima)[:max_heldout_records],
            target_scales=np.concatenate(test_target_ranges)[:max_heldout_records],
            condition_offsets=np.concatenate(test_condition_minima)[:max_heldout_records],
            condition_scales=np.concatenate(test_condition_ranges)[:max_heldout_records],
        )
        return train_set, test_set
    if normalization_id in {
        "rddm_window_minmax_neg1_1_v1",
        "window_minmax_neg1_1_v1",
    }:
        raw_sources = [concatenated_test_source]
        if load_train:
            raw_sources.append(np.concatenate(train_condition))
        if any(np.any(np.all(np.nan_to_num(source.astype(np.float32)) == 0, axis=1)) for source in raw_sources):
            raise ValueError(
                "RDDM-compatible loading requires all-zero PPG windows to be filtered first"
            )
        rddm_compatible = normalization_id == "rddm_window_minmax_neg1_1_v1"
        fitted_metadata = (
            _rddm_window_minmax_metadata(normalization_metadata)
            if rddm_compatible
            else _window_minmax_neg1_1_metadata(normalization_metadata)
        )
        test_target = _rddm_window_minmax_neg1_1(concatenated_test_target)
        test_source = _rddm_window_minmax_neg1_1(concatenated_test_source)
        if load_train:
            train_target = _rddm_window_minmax_neg1_1(np.concatenate(train_ecg))
            train_source = _rddm_window_minmax_neg1_1(np.concatenate(train_condition))
            train_masks = region_masks_train
            if max_train_records is not None:
                train_target = train_target[:max_train_records]
                train_source = train_source[:max_train_records]
                if train_masks is not None:
                    train_masks = train_masks[:max_train_records]
            train_set = PairedSignalDataset(
                train_target,
                train_source,
                clean_target=rddm_compatible,
                clean_condition_ppg=rddm_compatible,
                normalization_metadata=fitted_metadata,
                return_region_mask=return_region_mask_train,
                cached_region_masks=train_masks,
            )
        else:
            train_set = None
        if max_heldout_records is not None:
            test_target = test_target[:max_heldout_records]
            test_source = test_source[:max_heldout_records]
        test_set = PairedSignalDataset(
            test_target,
            test_source,
            clean_target=rddm_compatible,
            clean_condition_ppg=rddm_compatible,
            normalization_metadata=fitted_metadata,
            return_region_mask=False,
        )
        return train_set, test_set
    if normalization_id != "training_global_zscore_v1":
        raise ValueError(f"unsupported PPG/RCG normalization_id: {normalization_id}")
    if load_train:
        normalized = _training_global_zscore(
            np.concatenate(train_ecg),
            np.concatenate(train_condition),
            concatenated_test_target,
            concatenated_test_source,
            normalization_metadata,
        )
        train_target, train_source, test_target, test_source, fitted_metadata = normalized
        train_masks = region_masks_train
        if max_train_records is not None:
            train_target = train_target[:max_train_records]
            train_source = train_source[:max_train_records]
            if train_masks is not None:
                train_masks = train_masks[:max_train_records]
        train_set = PairedSignalDataset(
            train_target,
            train_source,
            clean_target=False,
            clean_condition_ppg=clean_condition_ppg,
            normalization_metadata=fitted_metadata,
            return_region_mask=return_region_mask_train,
            cached_region_masks=train_masks,
        )
    else:
        if normalization_metadata is None:
            raise ValueError("inference-only loading requires saved normalization metadata")
        fitted_metadata = _validate_normalization_metadata(normalization_metadata)
        test_target, test_source = _normalize_pair(
            concatenated_test_target, concatenated_test_source, fitted_metadata
        )
        train_set = None
    if max_heldout_records is not None:
        test_target = test_target[:max_heldout_records]
        test_source = test_source[:max_heldout_records]
    test_set = PairedSignalDataset(
        test_target,
        test_source,
        clean_target=False,
        clean_condition_ppg=clean_condition_ppg,
        normalization_metadata=fitted_metadata,
        return_region_mask=False,
    )
    return train_set, test_set


def get_ecg2ecg_datasets(
    DATA_PATH: str = "/data/user/RCFM/data/",
    datasets: Iterable[str] = ("PTBXL",),
    window_size: int = 4,
    condition_lead: int = 2,
    target_lead: int | Iterable[int] = 10,
    normalization_metadata: dict[str, object] | None = None,
    normalization_id: str = "training_global_zscore_v1",
    load_train: bool = True,
    heldout_split: str = "val",
    max_train_records: int | None = None,
    max_heldout_records: int | None = None,
    return_region_mask_train: bool = True,
    region_masks_train: np.ndarray | None = None,
):
    """Load single- or multi-target ECG windows from PTBXL/ICBEB style arrays."""

    if heldout_split not in {"val", "test"}:
        raise ValueError("heldout_split must be val or test")
    if max_train_records is not None and max_train_records <= 0:
        raise ValueError("max_train_records must be positive")
    if max_heldout_records is not None and max_heldout_records <= 0:
        raise ValueError("max_heldout_records must be positive")
    if region_masks_train is not None and not load_train:
        raise ValueError("training region masks cannot be supplied for inference-only loading")
    if region_masks_train is not None and not return_region_mask_train:
        raise ValueError("training region masks require return_region_mask_train=True")
    samples = SAMPLE_RATE * window_size
    target_leads = (
        (int(target_lead),)
        if isinstance(target_lead, (int, np.integer))
        else tuple(int(index) for index in target_lead)
    )
    if condition_lead in target_leads:
        raise ValueError("condition lead must not also be a target lead")
    root = Path(DATA_PATH)
    if normalization_id not in {
        "training_global_zscore_v1",
        "record_zscore_v1",
        "record_minmax_neg1_1_v1",
        "record_joint12_minmax_neg1_1_v1",
        "source_record_joint12_minmax_neg1_1_v1",
    }:
        raise ValueError("unsupported ECG normalization_id")
    train_target, train_condition, test_target, test_condition = [], [], [], []
    train_record_ids: list[np.ndarray] = []
    train_condition_means: list[np.ndarray] = []
    train_condition_scales: list[np.ndarray] = []
    train_target_means: list[np.ndarray] = []
    train_target_scales: list[np.ndarray] = []
    heldout_record_ids: list[np.ndarray] = []
    heldout_condition_means: list[np.ndarray] = []
    heldout_condition_scales: list[np.ndarray] = []
    heldout_target_means: list[np.ndarray] = []
    heldout_target_scales: list[np.ndarray] = []
    train_minmax_record_ids: list[np.ndarray] = []
    heldout_minmax_record_ids: list[np.ndarray] = []
    train_joint_record_ids: list[np.ndarray] = []
    train_joint_offsets: list[np.ndarray] = []
    train_joint_ranges: list[np.ndarray] = []
    heldout_joint_record_ids: list[np.ndarray] = []
    heldout_joint_offsets: list[np.ndarray] = []
    heldout_joint_ranges: list[np.ndarray] = []

    for dataset in datasets:
        dataset_root = root / dataset
        if load_train:
            train = np.load(dataset_root / "X_train_resampled.npy", allow_pickle=False)
            train_condition.append(_select_ecg_leads(train, samples, (condition_lead,)))
            train_target.append(_select_ecg_leads(train, samples, target_leads))
            if normalization_id == "record_zscore_v1":
                coefficients = _load_record_coefficients(
                    dataset_root, "train", len(train), condition_lead, target_leads
                )
                ids, condition_means, condition_scales, target_means, target_scales = coefficients
                train_record_ids.append(ids)
                train_condition_means.append(condition_means)
                train_condition_scales.append(condition_scales)
                train_target_means.append(target_means)
                train_target_scales.append(target_scales)
            elif normalization_id == "record_minmax_neg1_1_v1":
                train_minmax_record_ids.append(_load_record_ids(dataset_root, "train", len(train)))
            elif normalization_id in {
                "record_joint12_minmax_neg1_1_v1",
                "source_record_joint12_minmax_neg1_1_v1",
            }:
                ids, offsets, ranges = _load_record_joint12_coefficients(
                    dataset_root, "train", len(train)
                )
                train_joint_record_ids.append(ids)
                train_joint_offsets.append(offsets)
                train_joint_ranges.append(ranges)
        heldout_path = dataset_root / f"X_{heldout_split}_resampled.npy"
        if not heldout_path.exists():
            raise FileNotFoundError(
                f"Missing held-out ECG split: {heldout_path}. "
                "Evaluation must never fall back to the training array."
            )
        test = np.load(heldout_path, allow_pickle=False)

        test_condition.append(_select_ecg_leads(test, samples, (condition_lead,)))
        test_target.append(_select_ecg_leads(test, samples, target_leads))
        if normalization_id == "record_zscore_v1":
            coefficients = _load_record_coefficients(
                dataset_root, heldout_split, len(test), condition_lead, target_leads
            )
            ids, condition_means, condition_scales, target_means, target_scales = coefficients
            heldout_record_ids.append(ids)
            heldout_condition_means.append(condition_means)
            heldout_condition_scales.append(condition_scales)
            heldout_target_means.append(target_means)
            heldout_target_scales.append(target_scales)
        elif normalization_id == "record_minmax_neg1_1_v1":
            heldout_minmax_record_ids.append(
                _load_record_ids(dataset_root, heldout_split, len(test))
            )
        elif normalization_id in {
            "record_joint12_minmax_neg1_1_v1",
            "source_record_joint12_minmax_neg1_1_v1",
        }:
            ids, offsets, ranges = _load_record_joint12_coefficients(
                dataset_root, heldout_split, len(test)
            )
            heldout_joint_record_ids.append(ids)
            heldout_joint_offsets.append(offsets)
            heldout_joint_ranges.append(ranges)

    concatenated_test_target = np.concatenate(test_target)
    concatenated_test_source = np.concatenate(test_condition)
    record_kwargs: dict[str, np.ndarray] = {}
    heldout_record_kwargs: dict[str, np.ndarray] = {}
    if normalization_id == "record_zscore_v1":
        fitted_metadata = _record_zscore_metadata(normalization_metadata)
        heldout_record_kwargs = {
            "record_ids": np.concatenate(heldout_record_ids),
            "condition_means": np.concatenate(heldout_condition_means),
            "condition_scales": np.concatenate(heldout_condition_scales),
            "target_means": np.concatenate(heldout_target_means),
            "target_scales": np.concatenate(heldout_target_scales),
        }
        normalized_test_target = _record_zscore(
            concatenated_test_target,
            heldout_record_kwargs["target_means"],
            heldout_record_kwargs["target_scales"],
        )
        normalized_test_source = _record_zscore(
            concatenated_test_source,
            heldout_record_kwargs["condition_means"],
            heldout_record_kwargs["condition_scales"],
        )
        if load_train:
            record_kwargs = {
                "record_ids": np.concatenate(train_record_ids),
                "condition_means": np.concatenate(train_condition_means),
                "condition_scales": np.concatenate(train_condition_scales),
                "target_means": np.concatenate(train_target_means),
                "target_scales": np.concatenate(train_target_scales),
            }
            normalized_train_target = _record_zscore(
                np.concatenate(train_target),
                record_kwargs["target_means"],
                record_kwargs["target_scales"],
            )
            normalized_train_source = _record_zscore(
                np.concatenate(train_condition),
                record_kwargs["condition_means"],
                record_kwargs["condition_scales"],
            )
            if max_train_records is not None:
                normalized_train_target = normalized_train_target[:max_train_records]
                normalized_train_source = normalized_train_source[:max_train_records]
                record_kwargs = {
                    key: values[:max_train_records] for key, values in record_kwargs.items()
                }
            train_masks = region_masks_train
            if train_masks is not None and max_train_records is not None:
                train_masks = train_masks[:max_train_records]
            train_set = PairedSignalDataset(
                normalized_train_target,
                normalized_train_source,
                clean_target=False,
                clean_condition_ppg=False,
                normalization_metadata=fitted_metadata,
                return_region_mask=return_region_mask_train,
                cached_region_masks=train_masks,
                **record_kwargs,
            )
        else:
            train_set = None
    elif normalization_id == "record_minmax_neg1_1_v1":
        fitted_metadata = _record_minmax_neg1_1_metadata(normalization_metadata)
        target_offsets, target_ranges = _record_minmax_coefficients(concatenated_test_target)
        condition_offsets, condition_ranges = _record_minmax_coefficients(concatenated_test_source)
        normalized_test_target = _record_minmax_neg1_1(
            concatenated_test_target, target_offsets, target_ranges
        )
        normalized_test_source = _record_minmax_neg1_1(
            concatenated_test_source, condition_offsets, condition_ranges
        )
        heldout_record_kwargs = {
            "record_ids": np.concatenate(heldout_minmax_record_ids),
            "target_offsets": target_offsets,
            "target_scales": target_ranges,
            "condition_offsets": condition_offsets,
            "condition_scales": condition_ranges,
        }
        if load_train:
            concatenated_train_target = np.concatenate(train_target)
            concatenated_train_source = np.concatenate(train_condition)
            train_target_offsets, train_target_ranges = _record_minmax_coefficients(
                concatenated_train_target
            )
            train_condition_offsets, train_condition_ranges = _record_minmax_coefficients(
                concatenated_train_source
            )
            normalized_train_target = _record_minmax_neg1_1(
                concatenated_train_target, train_target_offsets, train_target_ranges
            )
            normalized_train_source = _record_minmax_neg1_1(
                concatenated_train_source, train_condition_offsets, train_condition_ranges
            )
            record_kwargs = {
                "record_ids": np.concatenate(train_minmax_record_ids),
                "target_offsets": train_target_offsets,
                "target_scales": train_target_ranges,
                "condition_offsets": train_condition_offsets,
                "condition_scales": train_condition_ranges,
            }
            if max_train_records is not None:
                normalized_train_target = normalized_train_target[:max_train_records]
                normalized_train_source = normalized_train_source[:max_train_records]
                record_kwargs = {
                    key: values[:max_train_records] for key, values in record_kwargs.items()
                }
            train_masks = region_masks_train
            if train_masks is not None and max_train_records is not None:
                train_masks = train_masks[:max_train_records]
            train_set = PairedSignalDataset(
                normalized_train_target,
                normalized_train_source,
                clean_target=False,
                clean_condition_ppg=False,
                normalization_metadata=fitted_metadata,
                return_region_mask=return_region_mask_train,
                cached_region_masks=train_masks,
                **record_kwargs,
            )
        else:
            train_set = None
    elif normalization_id in {
        "record_joint12_minmax_neg1_1_v1",
        "source_record_joint12_minmax_neg1_1_v1",
    }:
        if normalization_id == "record_joint12_minmax_neg1_1_v1":
            fitted_metadata = _record_joint12_minmax_neg1_1_metadata(normalization_metadata)
        else:
            fitted_metadata = _source_record_joint12_minmax_neg1_1_metadata(
                normalization_metadata
            )
        heldout_offsets = np.concatenate(heldout_joint_offsets)
        heldout_ranges = np.concatenate(heldout_joint_ranges)
        heldout_target_offsets = np.broadcast_to(
            heldout_offsets[:, None], concatenated_test_target.shape[:-1]
        ).copy()
        heldout_target_ranges = np.broadcast_to(
            heldout_ranges[:, None], concatenated_test_target.shape[:-1]
        ).copy()
        normalized_test_target = _record_minmax_neg1_1(
            concatenated_test_target, heldout_target_offsets, heldout_target_ranges
        )
        normalized_test_source = _record_minmax_neg1_1(
            concatenated_test_source, heldout_offsets, heldout_ranges
        )
        heldout_record_kwargs = {
            "record_ids": np.concatenate(heldout_joint_record_ids),
            "target_offsets": heldout_target_offsets,
            "target_scales": heldout_target_ranges,
            "condition_offsets": heldout_offsets,
            "condition_scales": heldout_ranges,
        }
        if load_train:
            concatenated_train_target = np.concatenate(train_target)
            concatenated_train_source = np.concatenate(train_condition)
            train_offsets = np.concatenate(train_joint_offsets)
            train_ranges = np.concatenate(train_joint_ranges)
            train_target_offsets = np.broadcast_to(
                train_offsets[:, None], concatenated_train_target.shape[:-1]
            ).copy()
            train_target_ranges = np.broadcast_to(
                train_ranges[:, None], concatenated_train_target.shape[:-1]
            ).copy()
            normalized_train_target = _record_minmax_neg1_1(
                concatenated_train_target, train_target_offsets, train_target_ranges
            )
            normalized_train_source = _record_minmax_neg1_1(
                concatenated_train_source, train_offsets, train_ranges
            )
            record_kwargs = {
                "record_ids": np.concatenate(train_joint_record_ids),
                "target_offsets": train_target_offsets,
                "target_scales": train_target_ranges,
                "condition_offsets": train_offsets,
                "condition_scales": train_ranges,
            }
            if max_train_records is not None:
                normalized_train_target = normalized_train_target[:max_train_records]
                normalized_train_source = normalized_train_source[:max_train_records]
                record_kwargs = {
                    key: values[:max_train_records] for key, values in record_kwargs.items()
                }
            train_masks = region_masks_train
            if train_masks is not None and max_train_records is not None:
                train_masks = train_masks[:max_train_records]
            train_set = PairedSignalDataset(
                normalized_train_target,
                normalized_train_source,
                clean_target=False,
                clean_condition_ppg=False,
                normalization_metadata=fitted_metadata,
                return_region_mask=return_region_mask_train,
                cached_region_masks=train_masks,
                **record_kwargs,
            )
        else:
            train_set = None
    elif load_train:
        normalized = _training_global_zscore(
            np.concatenate(train_target),
            np.concatenate(train_condition),
            concatenated_test_target,
            concatenated_test_source,
            normalization_metadata,
        )
        normalized_train_target, normalized_train_source, normalized_test_target, normalized_test_source, fitted_metadata = normalized
        if max_train_records is not None:
            normalized_train_target = normalized_train_target[:max_train_records]
            normalized_train_source = normalized_train_source[:max_train_records]
        train_masks = region_masks_train
        if train_masks is not None and max_train_records is not None:
            train_masks = train_masks[:max_train_records]
        train_set = PairedSignalDataset(
            normalized_train_target,
            normalized_train_source,
            clean_target=False,
            clean_condition_ppg=False,
            normalization_metadata=fitted_metadata,
            return_region_mask=return_region_mask_train,
            cached_region_masks=train_masks,
        )
    else:
        if normalization_metadata is None:
            raise ValueError("inference-only loading requires saved normalization metadata")
        fitted_metadata = _validate_normalization_metadata(normalization_metadata)
        normalized_test_target, normalized_test_source = _normalize_pair(
            concatenated_test_target, concatenated_test_source, fitted_metadata
        )
        train_set = None
    if max_heldout_records is not None:
        normalized_test_target = normalized_test_target[:max_heldout_records]
        normalized_test_source = normalized_test_source[:max_heldout_records]
        if heldout_record_kwargs:
            heldout_record_kwargs = {
                key: values[:max_heldout_records]
                for key, values in heldout_record_kwargs.items()
            }
    test_set = PairedSignalDataset(
        normalized_test_target,
        normalized_test_source,
        clean_target=False,
        clean_condition_ppg=False,
        normalization_metadata=fitted_metadata,
        return_region_mask=False,
        **heldout_record_kwargs,
    )
    return train_set, test_set


def get_datasets(*args, **kwargs):
    """Backward-compatible alias used by older scripts."""

    return get_ppg2ecg_datasets(*args, **kwargs)
