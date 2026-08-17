import json
from pathlib import Path

import pytest

from train_cfm_ot import parse_args_with_config


CONFIGS = (
    "mimic_afib/cfm_ot_minmax_zero_qc_seed31.yaml",
    "ptbxl/cfm_ot_record_minmax_neg1_1_seed31.yaml",
    "cpsc2018/cfm_ot_record_minmax_neg1_1_seed31.yaml",
    "wesad/cfm_ot_window_minmax_seed31.yaml",
    "mmecg/cfm_ot_window_minmax_seed31.yaml",
)


@pytest.mark.parametrize("relative", CONFIGS)
def test_cfm_ot_configs_freeze_the_factorial_contract(relative):
    path = Path(__file__).resolve().parents[1] / "configs" / relative
    config = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args_with_config(["--config", str(path)])
    assert args.model_family == "CFM"
    assert args.flow_matcher == "conditional" and args.sigma == 0.0
    assert args.region_weight == 0.0
    assert args.use_minibatch_ot is True
    assert args.ot_method == "exact"
    assert args.ot_sampling_strategy == "assignment"
    assert args.ot_strict_mode is True
    assert args.epochs == 500 and args.batch_size == 128
    assert args.save_every == 25 and args.checkpoint_policy == "full"
    assert args.wandb_mode == "online"
    assert config["split_hash"] and config["dataset_version"]


def test_cfm_ot_rejects_region_loss_and_disabled_ot():
    path = Path(__file__).resolve().parents[1] / "configs/ptbxl/cfm_ot_record_minmax_neg1_1_seed31.yaml"
    with pytest.raises(ValueError, match="region_weight=0"):
        parse_args_with_config(["--config", str(path), "--region_weight", "0.01"])
    with pytest.raises(ValueError, match="requires minibatch OT enabled"):
        parse_args_with_config(["--config", str(path), "--no-use_minibatch_ot"])
