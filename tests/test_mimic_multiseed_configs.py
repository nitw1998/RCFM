from pathlib import Path

import pytest

from train_cfm_compare import parse_args_with_config as parse_cfm
from train_cfm_ot import parse_args_with_config as parse_cfm_ot
from train_direct_cnn import parse_args_with_config as parse_direct_cnn
from train_rcfm import parse_args_with_config as parse_rcfm
from train_rddm_compare import parse_args_with_config as parse_rddm


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "mimic_afib"
EXPECTED_VERSION = "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1"
EXPECTED_SPLIT = "a7e388293adaa7b48d3493efc505dd8750520730cae9fd7649157866efa86a51"


def _assert_common(args, seed: int) -> None:
    assert args.datasets == "MIMIC-AFib"
    assert args.dataset_version == EXPECTED_VERSION
    assert args.split_hash == EXPECTED_SPLIT
    assert args.normalization_id == "rddm_window_minmax_neg1_1_v1"
    assert args.alignment_id == "paired_array_row_rddm_contract_zero_ppg_qc_v1"
    assert args.epochs == 500
    assert args.batch_size == 128
    assert args.save_every == 25
    assert args.seed == seed
    assert args.wandb_mode == "online"


@pytest.mark.parametrize("seed", (32, 33))
def test_mimic_flow_multiseed_variants_freeze_data_and_factors(seed: int) -> None:
    cfm = parse_cfm(
        [
            "--config", str(CONFIG_ROOT / "cfm_rddm_minmax_zero_qc_no_ot_seed31.yaml"),
            "--seed", str(seed), "--region_weight", "0", "--no-use_minibatch_ot",
        ]
    )
    cfm_ot = parse_cfm_ot(
        [
            "--config", str(CONFIG_ROOT / "cfm_ot_minmax_zero_qc_seed31.yaml"),
            "--seed", str(seed), "--region_weight", "0", "--use_minibatch_ot",
            "--ot_method", "exact", "--ot_sampling_strategy", "assignment",
        ]
    )
    rcfm = parse_rcfm(
        [
            "--config", str(CONFIG_ROOT / "rcfm_rddm_minmax_zero_qc_no_ot_seed31.yaml"),
            "--seed", str(seed), "--no-use_minibatch_ot",
        ]
    )
    rcfm_ot = parse_rcfm(
        [
            "--config", str(CONFIG_ROOT / "rcfm_rddm_minmax_zero_qc_exact_ot_seed31.yaml"),
            "--seed", str(seed), "--use_minibatch_ot", "--ot_method", "exact",
            "--ot_sampling_strategy", "assignment",
        ]
    )
    for args in (cfm, cfm_ot, rcfm, rcfm_ot):
        _assert_common(args, seed)
        assert args.validation_interval_epochs == 500
        assert args.heldout_role == "upstream_test_final_only"
        assert args.flow_matcher == "conditional"
        assert args.sigma == 0.0
    assert cfm.region_weight == 0 and cfm.use_minibatch_ot is False
    assert cfm_ot.region_weight == 0 and cfm_ot.use_minibatch_ot is True
    assert rcfm.region_weight == pytest.approx(0.01) and rcfm.use_minibatch_ot is False
    assert rcfm_ot.region_weight == pytest.approx(0.01) and rcfm_ot.use_minibatch_ot is True


def test_mimic_diagmask_pair_changes_only_inference_irrelevant_training_factors() -> None:
    config = str(CONFIG_ROOT / "rcfm_ot_xresnet_gradcam_l5_seed31.yaml")
    no_ot = parse_rcfm(["--config", config, "--seed", "32", "--no-use_minibatch_ot"])
    exact_ot = parse_rcfm(
        [
            "--config", config, "--seed", "32", "--use_minibatch_ot",
            "--ot_method", "exact", "--ot_sampling_strategy", "assignment",
        ]
    )
    for args in (no_ot, exact_ot):
        _assert_common(args, 32)
        assert args.mask_method == "xresnet1d101_afib_gradcam_l5_sample_center_v1"
        assert args.region_weight == pytest.approx(0.01)
        assert args.validation_interval_epochs == 500
    assert no_ot.use_minibatch_ot is False
    assert exact_ot.use_minibatch_ot is True


def test_mimic_rddm_and_direct_cnn_additional_seed_contracts() -> None:
    rddm = parse_rddm(
        [
            "--config", str(CONFIG_ROOT / "rddm_reproduced_minmax_zero_qc_seed31.yaml"),
            "--seed", "33",
        ]
    )
    direct = parse_direct_cnn(
        [
            "--config", str(CONFIG_ROOT / "direct_cnn_regression_minmax_zero_qc_seed31.yaml"),
            "--seed", "33",
        ]
    )
    for args in (rddm, direct):
        _assert_common(args, 33)
    assert rddm.expected_test_windows == 1800
    assert rddm.nT == 10
    assert rddm.reproduction_label == "RDDM (reproduced)"
    assert direct.expected_heldout_windows == 1800
    assert direct.heldout_role == "upstream_test_final_only"
