from pathlib import Path

import pytest

from train_rddm_compare import checkpoint_epochs, parse_args_with_config


def _config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "mimic_afib"
        / "rddm_reproduced_minmax_zero_qc_seed31.yaml"
    )


def test_rddm_compare_config_locks_upstream_and_matched_data_protocol():
    args = parse_args_with_config(["--config", str(_config_path())])

    assert args.task == "ppg2ecg"
    assert args.datasets == "MIMIC-AFib"
    assert args.normalization_id == "rddm_window_minmax_neg1_1_v1"
    assert args.expected_train_windows == 8400
    assert args.expected_test_windows == 1800
    assert args.epochs == 500
    assert args.batch_size == 128
    assert args.save_every == 25
    assert args.nT == 10
    assert args.beta_start == pytest.approx(1e-4)
    assert args.beta_end == pytest.approx(0.2)
    assert args.ddpm_loss_weight == pytest.approx(100.0)
    assert args.region_loss_weight == pytest.approx(1.0)
    assert args.scheduler_t_max == 1000
    assert args.wandb_mode == "online"


def test_rddm_checkpoint_schedule_is_exactly_every_twenty_five_epochs():
    assert checkpoint_epochs(500, 25) == list(range(25, 501, 25))

    with pytest.raises(ValueError, match="divisible"):
        checkpoint_epochs(501, 25)


def test_rddm_debug_caps_must_be_positive():
    with pytest.raises(ValueError, match="max_batches must be positive"):
        parse_args_with_config(["--config", str(_config_path()), "--max_batches", "0"])


def test_rddm_compare_rejects_nonpaper_diffusion_steps():
    with pytest.raises(ValueError, match="nT=10"):
        parse_args_with_config(["--config", str(_config_path()), "--nT", "50"])


def test_ptbxl_rddm_config_locks_multilead_adaptation_protocol():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "ptbxl"
        / "rddm_adapted_record_minmax_neg1_1_seed31.yaml"
    )
    args = parse_args_with_config(["--config", str(config_path)])

    assert args.task == "ecg2ecg"
    assert args.datasets == "PTBXL"
    assert args.normalization_id == "record_minmax_neg1_1_v1"
    assert args.condition_lead_index == 1
    assert args.target_lead_indices == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert args.heldout_split == "val"
    assert args.expected_train_windows == 17440
    assert args.expected_test_windows == 2193
    assert args.epochs == 500
    assert args.batch_size == 128
    assert args.save_every == 25
    assert args.wandb_mode == "online"
    assert args.reproduction_label == "RDDM-ECG (adapted)"


def test_cpsc2018_rddm_config_locks_multilead_adaptation_protocol():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "cpsc2018"
        / "rddm_adapted_record_minmax_neg1_1_seed31.yaml"
    )
    args = parse_args_with_config(["--config", str(config_path)])

    assert args.task == "ecg2ecg"
    assert args.datasets == "CPSC2018"
    assert args.dataset_version == (
        "cpsc2018-source-derived-all12lead-qc-v3-record-minmax-neg1-1"
    )
    assert args.split_hash == (
        "35e0a796a60e4d6979b1f050048495fdf1826eedda7d40e47a56d4dcd5874223"
    )
    assert args.normalization_id == "record_minmax_neg1_1_v1"
    assert args.condition_lead_index == 1
    assert args.target_lead_indices == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert args.heldout_split == "val"
    assert args.expected_train_windows == 5487
    assert args.expected_test_windows == 686
    assert args.epochs == 500
    assert args.batch_size == 128
    assert args.save_every == 25
    assert args.wandb_mode == "online"
    assert args.reproduction_label == "RDDM-ECG (adapted)"


def test_cpsc2018_rddm_rejects_a_different_lead_contract():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "cpsc2018"
        / "rddm_adapted_record_minmax_neg1_1_seed31.yaml"
    )

    with pytest.raises(ValueError, match="Lead II"):
        parse_args_with_config(
            ["--config", str(config_path), "--condition_lead_index", "0"]
        )
