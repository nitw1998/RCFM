from pathlib import Path

import pytest

from src.rcfm.training import run_training
from train_rcfm import parse_args_with_config


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "mimic_afib"


@pytest.mark.parametrize(
    ("filename", "path_type", "uses_ot"),
    (
        ("rcfm_path_ablation_vp_sigma01_seed31.yaml", "vp", False),
        ("rcfm_path_ablation_target_sigma01_seed31.yaml", "target", False),
        ("rcfm_path_ablation_sb_exact_multinomial_sigma01_seed31.yaml", "sb", True),
    ),
)
def test_mimic_path_ablation_configs_lock_matched_protocol(filename, path_type, uses_ot):
    args = parse_args_with_config(["--config", str(CONFIG_ROOT / filename)])

    assert args.task == "ppg2ecg"
    assert args.datasets == "MIMIC-AFib"
    assert args.dataset_version == "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1"
    assert args.split_hash == "a7e388293adaa7b48d3493efc505dd8750520730cae9fd7649157866efa86a51"
    assert args.normalization_id == "rddm_window_minmax_neg1_1_v1"
    assert args.experiment_role == "path_ablation"
    assert args.flow_matcher == path_type
    assert args.sigma == pytest.approx(0.1)
    assert args.region_weight == pytest.approx(0.01)
    assert args.use_minibatch_ot is uses_ot
    assert args.epochs == 500 and args.batch_size == 128 and args.save_every == 25
    assert args.validation_interval_epochs == 500
    assert args.validation_fixed_noise_seed == 2025
    assert args.wandb_mode == "online"


def test_sb_ablation_config_uses_one_instrumented_exact_coupling():
    args = parse_args_with_config(
        ["--config", str(CONFIG_ROOT / "rcfm_path_ablation_sb_exact_multinomial_sigma01_seed31.yaml")]
    )
    assert args.ot_method == "exact"
    assert args.ot_sampling_strategy == "multinomial"
    assert args.ot_diagnostics and args.ot_strict_mode and args.association_debug


@pytest.mark.parametrize(
    ("filename", "updates", "message"),
    (
        (
            "rcfm_path_ablation_vp_sigma01_seed31.yaml",
            {"use_minibatch_ot": True},
            "must disable OT",
        ),
        (
            "rcfm_path_ablation_target_sigma01_seed31.yaml",
            {"use_minibatch_ot": True},
            "must disable OT",
        ),
        (
            "rcfm_path_ablation_sb_exact_multinomial_sigma01_seed31.yaml",
            {"use_minibatch_ot": False},
            "requires one external OT coupling",
        ),
        (
            "rcfm_path_ablation_sb_exact_multinomial_sigma01_seed31.yaml",
            {"ot_sampling_strategy": "assignment"},
            "requires exact OT with multinomial sampling",
        ),
    ),
)
def test_training_rejects_invalid_path_coupling_protocol(filename, updates, message):
    args = parse_args_with_config(["--config", str(CONFIG_ROOT / filename)])
    for key, value in updates.items():
        setattr(args, key, value)

    def dataset_builder_must_not_run(*_args, **_kwargs):
        raise AssertionError("invalid protocol reached dataset loading")

    with pytest.raises(ValueError, match=message):
        run_training(args, dataset_builder_must_not_run)
