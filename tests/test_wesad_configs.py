from pathlib import Path

from train_cfm_compare import parse_args_with_config as parse_cfm
from train_rcfm import parse_args_with_config as parse_rcfm
from train_rddm_compare import parse_args_with_config as parse_rddm


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "wesad"
SPLIT_HASH = "0b90bffe7c3032243c803e534de3592618284d26792c2b612ce7af17a81a85cd"


def _assert_shared(args):
    assert args.task == "ppg2ecg"
    assert args.datasets == "WESAD"
    assert args.normalization_id == "window_minmax_neg1_1_v1"
    assert args.split_hash == SPLIT_HASH
    assert args.epochs == 500
    assert args.batch_size == 128
    assert args.save_every == 25
    assert args.wandb_mode == "online"


def test_flow_configs_freeze_comparable_wesad_protocol():
    cfm = parse_cfm(["--config", str(CONFIG_ROOT / "cfm_window_minmax_no_ot_seed31.yaml")])
    rcfm = parse_rcfm(["--config", str(CONFIG_ROOT / "rcfm_window_minmax_no_ot_seed31.yaml")])
    ot = parse_rcfm(["--config", str(CONFIG_ROOT / "rcfm_window_minmax_exact_ot_seed31.yaml")])
    for args in (cfm, rcfm, ot):
        _assert_shared(args)
        assert args.heldout_role == "upstream_test_final_only"
        assert args.validation_interval_epochs == args.epochs
    assert cfm.region_weight == 0.0 and not cfm.use_minibatch_ot
    assert rcfm.region_weight == 0.01 and not rcfm.use_minibatch_ot
    assert ot.region_weight == 0.01 and ot.use_minibatch_ot
    assert ot.ot_method == "exact" and ot.ot_sampling_strategy == "assignment"


def test_rddm_config_uses_same_wesad_split_and_schedule():
    args = parse_rddm(["--config", str(CONFIG_ROOT / "rddm_matched_window_minmax_seed31.yaml")])
    _assert_shared(args)
    assert args.expected_train_windows == 17494
    assert args.expected_test_windows == 4213
    assert args.nT == 10
    assert args.reproduction_label == "RDDM-PPG (matched-protocol reproduction)"
