from __future__ import annotations

import os
from pathlib import Path
import subprocess


def test_appendix_c_entry_maps_every_model_dataset_pair(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    entry = root / "scripts/train_evidence_appendix_c.sh"
    env = {
        **os.environ,
        "RCFM_DATA_ROOT": str(tmp_path / "data"),
        "RCFM_RUNS_ROOT": str(tmp_path / "runs"),
        "RCFM_PYTHON": "python",
    }
    expected = {
        ("direct_cnn", "mimic_afib"): "direct_cnn_regression_minmax_zero_qc_seed31.yaml",
        ("direct_cnn", "ptbxl"): "direct_cnn_regression_record_minmax_seed31.yaml",
        ("direct_cnn", "cpsc2018"): "direct_cnn_regression_record_minmax_seed31.yaml",
        ("direct_cnn", "wesad"): "direct_cnn_regression_window_minmax_seed31.yaml",
        ("direct_cnn", "mmecg"): "direct_cnn_regression_window_minmax_seed31.yaml",
        ("cat", "mimic_afib"): "cat_ppg_mimic_afib_seed31.yaml",
        ("cat", "ptbxl"): "cat_ecg_adapted_record_minmax_seed31.yaml",
        ("cat", "cpsc2018"): "cat_ecg_adapted_record_minmax_seed31.yaml",
        ("cat", "wesad"): "cat_ppg_reproduced_window_minmax_seed31.yaml",
        ("cat", "mmecg"): "cat_rcg_adapted_window_minmax_seed31.yaml",
    }
    for (model, dataset), config_name in expected.items():
        result = subprocess.run(
            ["bash", str(entry), model, dataset, "--dry-run", "--wandb_mode", "disabled"],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert config_name in result.stdout
        assert "--data_root" in result.stdout
        assert "--output_dir" in result.stdout
        assert "--wandb_mode disabled" in result.stdout


def test_appendix_c_entry_rejects_unknown_pair(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", "scripts/train_evidence_appendix_c.sh", "unknown", "ptbxl"],
        cwd=root,
        env={**os.environ, "RCFM_DATA_ROOT": str(tmp_path), "RCFM_RUNS_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr
