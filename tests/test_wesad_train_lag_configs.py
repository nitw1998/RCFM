from pathlib import Path
import subprocess

from train_cfm_compare import parse_args_with_config as parse_cfm
from train_cfm_ot import parse_args_with_config as parse_cfm_ot
from train_rcfm import parse_args_with_config as parse_rcfm
from train_rddm_compare import parse_args_with_config as parse_rddm


ROOT = Path(__file__).resolve().parents[1] / "configs" / "wesad"
VERSION = "wesad-subject-fold1-train-fixed-lag-aligned-v2"
ALIGNMENT = "train_subjects_peak_median_fixed_lag_crop_before_window_subject_fold1_v2"


def _assert_v2(args):
    assert args.dataset_version == VERSION
    assert args.alignment_id == ALIGNMENT
    assert args.datasets == "WESAD"
    assert args.normalization_id == "window_minmax_neg1_1_v1"


def test_aligned_flow_configs_share_one_data_contract():
    cfm = parse_cfm(["--config", str(ROOT / "cfm_train_fixed_lag_v2_seed31.yaml")])
    cfm_ot = parse_cfm_ot(["--config", str(ROOT / "cfm_ot_train_fixed_lag_v2_seed31.yaml")])
    rcfm = parse_rcfm(["--config", str(ROOT / "rcfm_train_fixed_lag_v2_seed31.yaml")])
    rcfm_ot = parse_rcfm(["--config", str(ROOT / "rcfm_ot_train_fixed_lag_v2_seed31.yaml")])
    for args in (cfm, cfm_ot, rcfm, rcfm_ot):
        _assert_v2(args)
    assert cfm.region_weight == cfm_ot.region_weight == 0.0
    assert rcfm.region_weight == rcfm_ot.region_weight == 0.01
    assert not cfm.use_minibatch_ot and not rcfm.use_minibatch_ot
    assert cfm_ot.use_minibatch_ot and rcfm_ot.use_minibatch_ot


def test_aligned_rddm_config_keeps_subject_split_and_official_sampler_contract():
    args = parse_rddm(["--config", str(ROOT / "rddm_train_fixed_lag_v2_seed31.yaml")])
    _assert_v2(args)
    assert (args.expected_train_windows, args.expected_test_windows) == (17494, 4213)
    assert args.nT == 10
    assert args.reproduction_label == "RDDM-PPG (matched-protocol reproduction)"


def test_launcher_uses_gpu_then_variant_interface():
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "train_wesad_fixed_lag_v2.sh"
    result = subprocess.run(
        ["bash", str(launcher)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "GPU_INDEX {cfm|rcfm|rcfm_ot|rddm}" in result.stderr
