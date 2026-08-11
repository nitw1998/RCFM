from pathlib import Path

from train_cfm_compare import parse_args_with_config as parse_cfm
from train_rcfm import parse_args_with_config as parse_rcfm
from train_rddm_compare import parse_args_with_config as parse_rddm


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "mmecg"
SPLIT_HASH = "e26fc81121cfd3b0e457608e37a7aa496ac7c48a0e0e7583a465bd069cc9da9f"


def _assert_shared(args):
    assert args.task == "rcg2ecg"
    assert args.datasets == "mmECG"
    assert args.normalization_id == "window_minmax_neg1_1_v1"
    assert args.split_hash == SPLIT_HASH
    assert args.epochs == 500
    assert args.batch_size == 128
    assert args.save_every == 25
    assert args.wandb_mode == "online"


def test_flow_configs_freeze_comparable_mmecg_protocol():
    cfm = parse_cfm(["--config", str(CONFIG_ROOT / "cfm_window_minmax_no_ot_seed31.yaml")])
    rcfm = parse_rcfm(["--config", str(CONFIG_ROOT / "rcfm_window_minmax_no_ot_seed31.yaml")])
    ot = parse_rcfm(["--config", str(CONFIG_ROOT / "rcfm_window_minmax_exact_ot_seed31.yaml")])
    for args in (cfm, rcfm, ot):
        _assert_shared(args)
        assert args.heldout_role == "upstream_test_final_only"
    assert cfm.region_weight == 0.0 and not cfm.use_minibatch_ot
    assert rcfm.region_weight == 0.01 and not rcfm.use_minibatch_ot
    assert ot.region_weight == 0.01 and ot.use_minibatch_ot
    assert ot.ot_method == "exact" and ot.ot_sampling_strategy == "assignment"


def test_rddm_config_uses_same_mmecg_split_and_schedule():
    args = parse_rddm(["--config", str(CONFIG_ROOT / "rddm_adapted_window_minmax_seed31.yaml")])
    _assert_shared(args)
    assert args.expected_train_windows == 9590
    assert args.expected_test_windows == 2877
    assert args.reproduction_label == "RDDM-RCG (adapted)"
