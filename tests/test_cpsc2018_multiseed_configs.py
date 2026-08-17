from pathlib import Path

import pytest

from train_cfm_compare import parse_args_with_config as parse_cfm
from train_cfm_ot import parse_args_with_config as parse_cfm_ot
from train_rcfm import parse_args_with_config as parse_rcfm


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "cpsc2018"
EXPECTED_VERSION = "cpsc2018-source-derived-all12lead-qc-v3-record-minmax-neg1-1"
EXPECTED_SPLIT = "35e0a796a60e4d6979b1f050048495fdf1826eedda7d40e47a56d4dcd5874223"


def _assert_common(args, seed: int) -> None:
    assert args.datasets == "CPSC2018"
    assert args.dataset_version == EXPECTED_VERSION
    assert args.split_hash == EXPECTED_SPLIT
    assert args.normalization_id == "record_minmax_neg1_1_v1"
    assert args.alignment_id == "same_record_simultaneous_channels_first_4s_lead_II_to_other_11_v3"
    assert args.condition_lead_index == 1
    assert args.target_lead_indices == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert args.epochs == 500
    assert args.batch_size == 128
    assert args.save_every == 25
    assert args.validation_interval_epochs == 1
    assert args.validation_fixed_noise_seed == 2025
    assert args.seed == seed
    assert args.wandb_mode == "online"
    assert args.flow_matcher == "conditional"
    assert args.sigma == 0.0


@pytest.mark.parametrize("seed", (32, 33))
def test_cpsc_core_factorial_additional_seed_contract(seed: int) -> None:
    cfm = parse_cfm(
        [
            "--config", str(CONFIG_ROOT / "cfm_compare_record_minmax_neg1_1_no_ot_seed31.yaml"),
            "--seed", str(seed), "--region_weight", "0", "--no-use_minibatch_ot",
            "--validation_interval_epochs", "1", "--wandb_mode", "online",
        ]
    )
    cfm_ot = parse_cfm_ot(
        [
            "--config", str(CONFIG_ROOT / "cfm_ot_record_minmax_neg1_1_seed31.yaml"),
            "--seed", str(seed), "--region_weight", "0", "--use_minibatch_ot",
            "--ot_method", "exact", "--ot_sampling_strategy", "assignment",
            "--validation_interval_epochs", "1", "--wandb_mode", "online",
        ]
    )
    rcfm = parse_rcfm(
        [
            "--config", str(CONFIG_ROOT / "rcfm_record_minmax_neg1_1_no_ot_seed31.yaml"),
            "--seed", str(seed), "--no-use_minibatch_ot",
            "--validation_interval_epochs", "1", "--wandb_mode", "online",
        ]
    )
    rcfm_ot = parse_rcfm(
        [
            "--config", str(CONFIG_ROOT / "rcfm_record_minmax_neg1_1_exact_ot_seed31.yaml"),
            "--seed", str(seed), "--use_minibatch_ot", "--ot_method", "exact",
            "--ot_sampling_strategy", "assignment", "--validation_interval_epochs", "1",
            "--wandb_mode", "online",
        ]
    )
    for args in (cfm, cfm_ot, rcfm, rcfm_ot):
        _assert_common(args, seed)
    assert cfm.region_weight == 0 and cfm.use_minibatch_ot is False
    assert cfm_ot.region_weight == 0 and cfm_ot.use_minibatch_ot is True
    assert rcfm.region_weight == pytest.approx(0.01) and rcfm.use_minibatch_ot is False
    assert rcfm_ot.region_weight == pytest.approx(0.01) and rcfm_ot.use_minibatch_ot is True
    assert cfm_ot.ot_sampling_strategy == "assignment"
    assert rcfm_ot.ot_sampling_strategy == "assignment"


def test_public_cpsc_launchers_are_path_configurable() -> None:
    repository = Path(__file__).resolve().parents[1]
    launcher = (repository / "scripts/launch_cpsc2018_factorial_multiseed.sh").read_text()
    worker = (repository / "scripts/run_cpsc2018_factorial_multiseed_worker.sh").read_text()
    for source in (launcher, worker):
        assert "/data/user" not in source
        assert "/home/user" not in source
        assert "CPSC2018_DATA_ROOT" in source
        assert "RCFM_RUNS_ROOT" in source
    assert "SEEDS=(31 32 33)" in launcher
    assert "--validation_interval_epochs 1" in worker
    assert "region_mask_path" not in worker
