from __future__ import annotations

from pathlib import Path

import pytest

from train_direct_cnn import parse_args_with_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "ptbxl": "configs/ptbxl/direct_cnn_random_window80_20_joint12_seed31.yaml",
    "cpsc2018": "configs/cpsc2018/direct_cnn_random_window80_20_joint12_seed31.yaml",
    "mimic-afib": "configs/mimic_afib/direct_cnn_random_window80_20_seed31.yaml",
    "wesad": "configs/wesad/direct_cnn_random_window80_20_record_minmax_seed31.yaml",
    "mmecg": "configs/mmecg/direct_cnn_random_window80_20_seed31.yaml",
}


@pytest.mark.parametrize("dataset_key", CONFIGS)
def test_random_window_direct_cnn_config_contract(dataset_key: str) -> None:
    args = parse_args_with_config(["--config", str(ROOT / CONFIGS[dataset_key])])
    assert args.seed == 31
    assert args.epochs == 500
    assert args.heldout_role == "validation"
    assert args.expected_train_windows > args.expected_heldout_windows > 0


def test_random_window_direct_cnn_rejects_wrong_split_hash() -> None:
    with pytest.raises(ValueError, match="frozen data contract"):
        parse_args_with_config(
            [
                "--config",
                str(ROOT / CONFIGS["ptbxl"]),
                "--split_hash",
                "wrong",
            ]
        )
