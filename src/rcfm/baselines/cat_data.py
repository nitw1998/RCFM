"""Isolated frozen-data adapter for CAT-PPG on MIMIC-AFib."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data import PairedSignalDataset, _rddm_window_minmax_metadata, _rddm_window_minmax_neg1_1


MIMIC_DATASET_VERSION = "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1"
MIMIC_SPLIT_HASH = "a7e388293adaa7b48d3493efc505dd8750520730cae9fd7649157866efa86a51"


def load_mimic_cat_datasets(
    data_root: Path | str,
    max_train_records: int | None = None,
    max_test_records: int | None = None,
) -> tuple[PairedSignalDataset, PairedSignalDataset, dict[str, object]]:
    """Load the frozen arrays without constructing target-derived region masks."""

    root = Path(data_root) / "MIMIC-AFib"
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != MIMIC_DATASET_VERSION:
        raise ValueError("MIMIC CAT dataset version changed")
    if manifest.get("split_membership_hash") != MIMIC_SPLIT_HASH:
        raise ValueError("MIMIC CAT split membership changed")
    arrays = {
        f"{kind}_{split}": np.asarray(
            np.load(root / f"{kind}_{split}_4sec.npy", allow_pickle=False), dtype=np.float32
        ).reshape(-1, 512)
        for split in ("train", "test")
        for kind in ("ecg", "ppg")
    }
    if len(arrays["ecg_train"]) != 8400 or len(arrays["ecg_test"]) != 1800:
        raise ValueError("MIMIC CAT arrays violate the frozen 8400/1800 split")
    for split in ("train", "test"):
        if len(arrays[f"ecg_{split}"]) != len(arrays[f"ppg_{split}"]):
            raise ValueError("MIMIC ECG and PPG rows are not paired")
        ppg = np.nan_to_num(arrays[f"ppg_{split}"])
        if np.any(np.all(ppg == 0, axis=1)):
            raise ValueError("MIMIC CAT input still contains an all-zero PPG window")
    train_count = 8400 if max_train_records is None else min(max_train_records, 8400)
    test_count = 1800 if max_test_records is None else min(max_test_records, 1800)
    metadata = _rddm_window_minmax_metadata()

    def make(split: str, count: int) -> PairedSignalDataset:
        target = _rddm_window_minmax_neg1_1(arrays[f"ecg_{split}"][:count])
        source = _rddm_window_minmax_neg1_1(arrays[f"ppg_{split}"][:count])
        return PairedSignalDataset(
            target,
            source,
            clean_target=True,
            clean_condition_ppg=True,
            normalization_metadata=metadata,
            return_region_mask=False,
        )

    return make("train", train_count), make("test", test_count), metadata
