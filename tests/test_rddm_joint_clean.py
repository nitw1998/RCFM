import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import prepare_rddm_mimic_wesad_clean as prep
from train_rddm_joint_clean import (
    CleanRDDMDataset,
    _lr_factor,
    _validate_artifact,
    parse_args_with_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_joint_config_freezes_upstream_derived_training_contract():
    args = parse_args_with_config(
        ["--config", str(ROOT / "configs/rddm/mimic_wesad_joint_official_clean_seed31.yaml")]
    )
    assert args.epochs == 1000
    assert args.batch_size == 512
    assert args.warmup_epochs == 20
    assert args.nT == 10
    assert args.data_parallel is True
    assert args.expected_mimic_train_windows == 8400
    assert args.expected_wesad_train_windows == 17494


def test_paper_inferred_warmup_cosine_has_expected_boundaries():
    assert _lr_factor(0, 1000, 20) == pytest.approx(0.05)
    assert _lr_factor(19, 1000, 20) == pytest.approx(1.0)
    assert _lr_factor(999, 1000, 20) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        _lr_factor(0, 20, 20)


def test_cleaning_order_is_minmax_then_neurokit(monkeypatch):
    samples = np.linspace(10, 20, 512, dtype=np.float32)[None, :]
    observed = []

    def fake_ppg(values, sampling_rate):
        observed.append(("ppg", values.copy(), sampling_rate))
        return values + 2

    def fake_ecg(values, sampling_rate, method):
        observed.append(("ecg", values.copy(), sampling_rate, method))
        return values - 2

    monkeypatch.setattr(prep.nk, "ppg_clean", fake_ppg)
    monkeypatch.setattr(prep.nk, "ecg_clean", fake_ecg)
    ecg, ppg = prep._clean_arrays(samples, samples * 3, workers=1, chunksize=1)
    assert [row[0] for row in observed] == ["ppg", "ecg"]
    for row in observed:
        np.testing.assert_allclose([row[1].min(), row[1].max()], [-1, 1], atol=1e-6)
    assert observed[0][2] == 128
    assert observed[1][2:] == (128, "pantompkins1985")
    np.testing.assert_allclose([ecg.min(), ecg.max()], [-3, -1], atol=1e-6)
    np.testing.assert_allclose([ppg.min(), ppg.max()], [1, 3], atol=1e-6)


def test_clean_dataset_requires_precomputed_float32_masks(tmp_path):
    root = tmp_path / "WESAD"
    root.mkdir()
    np.save(root / "ecg_train_4sec.npy", np.zeros((2, 512), dtype=np.float32))
    np.save(root / "ppg_train_4sec.npy", np.ones((2, 512), dtype=np.float32))
    np.save(root / "region_masks_train.npy", np.zeros((2, 1, 512), dtype=np.float32))
    dataset = CleanRDDMDataset(root)
    target, condition, mask = dataset[0]
    assert target.shape == condition.shape == mask.shape == (1, 512)


def test_preparer_writes_no_test_mask_and_manifest_validates(tmp_path, monkeypatch):
    mimic, wesad, output = tmp_path / "mimic", tmp_path / "wesad", tmp_path / "out"
    for base, dataset in ((mimic, "MIMIC-AFib"), (wesad, "WESAD")):
        root = base / dataset
        root.mkdir(parents=True)
        values = np.tile(np.linspace(-2, 2, 512, dtype=np.float32), (2, 1))
        for split in ("train", "test"):
            np.save(root / f"ecg_{split}_4sec.npy", values)
            np.save(root / f"ppg_{split}_4sec.npy", values + 0.5)
    monkeypatch.setattr(prep, "_clean_pair", lambda pair: pair)
    monkeypatch.setattr(prep, "_region_mask", lambda _row: np.zeros((1, 512), dtype=np.float32))
    args = argparse.Namespace(
        mimic_root=mimic, wesad_root=wesad, output_root=output,
        workers=1, chunksize=1, max_windows=None,
    )
    prep.prepare(args)
    assert not (output / "MIMIC-AFib/region_masks_test.npy").exists()
    manifest, counts = _validate_artifact(output)
    assert counts == {"MIMIC-AFib": 2, "WESAD": 2}
    assert manifest["preprocessing"]["test_mask_generated"] is False
