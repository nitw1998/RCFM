import json
from pathlib import Path

import pytest
import torch

from src.rcfm.baselines import DirectRegressionCNN
from train_direct_cnn import parse_args_with_config


def test_direct_cnn_is_one_pass_shape_preserving_and_lightweight():
    model = DirectRegressionCNN(input_channels=1, output_channels=11, width=32)
    output = model(torch.randn(3, 1, 512))
    assert output.shape == (3, 11, 512)
    assert model.receptive_field >= 512
    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000


def test_direct_cnn_rejects_wrong_input_channels():
    model = DirectRegressionCNN(input_channels=1, output_channels=1)
    with pytest.raises(ValueError, match="condition"):
        model(torch.randn(2, 2, 512))


def test_frozen_direct_cnn_configs():
    root = Path(__file__).resolve().parents[1] / "configs"
    paths = {
        "mimic": root / "mimic_afib/direct_cnn_regression_minmax_zero_qc_seed31.yaml",
        "ptbxl": root / "ptbxl/direct_cnn_regression_record_minmax_seed31.yaml",
        "cpsc": root / "cpsc2018/direct_cnn_regression_record_minmax_seed31.yaml",
        "wesad": root / "wesad/direct_cnn_regression_window_minmax_seed31.yaml",
        "mmecg": root / "mmecg/direct_cnn_regression_window_minmax_seed31.yaml",
    }
    configs = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    for config in configs.values():
        assert config["epochs"] == 500
        assert config["batch_size"] == 128
        assert config["save_every"] == 25
        assert config["wandb_mode"] == "online"
        assert config["width"] == 32
        assert config["dilations"] == "1,2,4,8,16,32,64"
    assert configs["mimic"]["heldout_role"] == "upstream_test_final_only"
    assert configs["ptbxl"]["heldout_role"] == "validation"
    assert configs["cpsc"]["heldout_role"] == "validation"
    assert configs["wesad"]["heldout_role"] == "upstream_test_final_only"
    assert configs["mmecg"]["heldout_role"] == "upstream_test_final_only"
    for name in ("ptbxl", "cpsc"):
        assert configs[name]["target_lead_indices"] == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    expected_datasets = {
        "mimic": "MIMIC-AFib", "ptbxl": "PTBXL", "cpsc": "CPSC2018",
        "wesad": "WESAD", "mmecg": "mmECG",
    }
    for name, path in paths.items():
        assert parse_args_with_config(["--config", str(path)]).datasets == expected_datasets[name]
