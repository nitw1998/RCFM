"""Dataset helpers for paired physiological signal generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import neurokit2 as nk
import numpy as np
import sklearn.preprocessing as skp
from torch.utils.data import Dataset


SAMPLE_RATE = 128


def _as_2d_windows(array: np.ndarray, samples: int) -> np.ndarray:
    return np.asarray(array, dtype=np.float32).reshape(-1, samples)


def _minmax_per_window(array: np.ndarray) -> np.ndarray:
    return skp.minmax_scale(np.nan_to_num(array).astype(np.float32), (-1, 1), axis=1)


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
    """Return target ECG, conditioning signal, and target ECG region mask."""

    def __init__(
        self,
        target_ecg: np.ndarray,
        condition_signal: np.ndarray,
        clean_target: bool = False,
        clean_condition_ppg: bool = False,
    ) -> None:
        if len(target_ecg) != len(condition_signal):
            raise ValueError("target_ecg and condition_signal must have the same number of windows")
        self.target_ecg = target_ecg
        self.condition_signal = condition_signal
        self.clean_target = clean_target
        self.clean_condition_ppg = clean_condition_ppg

    def __getitem__(self, index: int):
        target = self.target_ecg[index].astype(np.float32)
        condition = self.condition_signal[index].astype(np.float32)
        window_size = target.shape[-1]

        if self.clean_target:
            target = nk.ecg_clean(target.reshape(window_size), sampling_rate=SAMPLE_RATE, method="pantompkins1985")
        if self.clean_condition_ppg:
            condition = nk.ppg_clean(condition.reshape(window_size), sampling_rate=SAMPLE_RATE)

        region_mask = _ecg_region_mask(target, sampling_rate=SAMPLE_RATE)
        return (
            target.reshape(1, window_size).copy(),
            condition.reshape(1, window_size).copy(),
            region_mask.copy(),
        )

    def __len__(self) -> int:
        return len(self.target_ecg)


def get_ppg2ecg_datasets(
    DATA_PATH: str = "/data/user/RCFM/data/",
    datasets: Iterable[str] = ("MIMIC-AFib",),
    window_size: int = 4,
    clean_condition_ppg: bool = True,
):
    """Load paired PPG/RCG-to-ECG windows.

    The mmECG preprocessed RCG arrays are stored with the historical ``ppg_*``
    file names, so the same loader is used for both PPG-to-ECG and RCG-to-ECG.
    """

    samples = SAMPLE_RATE * window_size
    root = Path(DATA_PATH)
    train_ecg, train_condition, test_ecg, test_condition = [], [], [], []

    for dataset in datasets:
        dataset_root = root / dataset
        train_ecg.append(_as_2d_windows(np.load(dataset_root / f"ecg_train_{window_size}sec.npy"), samples))
        train_condition.append(_as_2d_windows(np.load(dataset_root / f"ppg_train_{window_size}sec.npy"), samples))
        test_ecg.append(_as_2d_windows(np.load(dataset_root / f"ecg_test_{window_size}sec.npy"), samples))
        test_condition.append(_as_2d_windows(np.load(dataset_root / f"ppg_test_{window_size}sec.npy"), samples))

    train_set = PairedSignalDataset(
        _minmax_per_window(np.concatenate(train_ecg)),
        _minmax_per_window(np.concatenate(train_condition)),
        clean_target=True,
        clean_condition_ppg=clean_condition_ppg,
    )
    test_set = PairedSignalDataset(
        _minmax_per_window(np.concatenate(test_ecg)),
        _minmax_per_window(np.concatenate(test_condition)),
        clean_target=True,
        clean_condition_ppg=clean_condition_ppg,
    )
    return train_set, test_set


def get_ecg2ecg_datasets(
    DATA_PATH: str = "/data/user/RCFM/data/",
    datasets: Iterable[str] = ("PTBXL",),
    window_size: int = 4,
    condition_lead: int = 2,
    target_lead: int = 10,
):
    """Load reduced-lead ECG-to-ECG windows from PTBXL/ICBEB style arrays."""

    samples = SAMPLE_RATE * window_size
    root = Path(DATA_PATH)
    train_target, train_condition, test_target, test_condition = [], [], [], []

    for dataset in datasets:
        dataset_root = root / dataset
        train = np.load(dataset_root / "X_train_resampled.npy", allow_pickle=True)
        val_path = dataset_root / "X_val_resampled.npy"
        test = np.load(val_path, allow_pickle=True) if val_path.exists() else train

        train_condition.append(_as_2d_windows(train[:, :samples, condition_lead], samples))
        train_target.append(_as_2d_windows(train[:, :samples, target_lead], samples))
        test_condition.append(_as_2d_windows(test[:, :samples, condition_lead], samples))
        test_target.append(_as_2d_windows(test[:, :samples, target_lead], samples))

    train_set = PairedSignalDataset(
        _minmax_per_window(np.concatenate(train_target)),
        _minmax_per_window(np.concatenate(train_condition)),
        clean_target=False,
        clean_condition_ppg=False,
    )
    test_set = PairedSignalDataset(
        _minmax_per_window(np.concatenate(test_target)),
        _minmax_per_window(np.concatenate(test_condition)),
        clean_target=False,
        clean_condition_ppg=False,
    )
    return train_set, test_set


def get_datasets(*args, **kwargs):
    """Backward-compatible alias used by older scripts."""

    return get_ppg2ecg_datasets(*args, **kwargs)
