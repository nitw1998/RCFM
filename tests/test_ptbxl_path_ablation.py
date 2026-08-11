import json
from pathlib import Path

import pytest

from src.rcfm.training import run_training
from train_rcfm import parse_args_with_config


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "ptbxl"
BASE_CONFIG = CONFIG_ROOT / "rcfm_record_minmax_neg1_1_no_ot_seed31.yaml"
EXPECTED_SPLIT_HASH = "7784c98a2c8daccc23fc7cb0d47dc1933eeee77f120c05f8a24ac149cd8474f7"


@pytest.mark.parametrize(
    ("filename", "path_type", "uses_ot"),
    (
        ("rcfm_path_ablation_vp_sigma01_seed31.yaml", "vp", False),
        ("rcfm_path_ablation_target_sigma01_seed31.yaml", "target", False),
        ("rcfm_path_ablation_sb_exact_multinomial_sigma01_seed31.yaml", "sb", True),
    ),
)
def test_ptbxl_path_configs_lock_waveform_protocol(filename, path_type, uses_ot):
    args = parse_args_with_config(["--config", str(CONFIG_ROOT / filename)])

    assert args.task == "ecg2ecg"
    assert args.datasets == "PTBXL"
    assert args.dataset_version == "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1"
    assert args.split_hash == EXPECTED_SPLIT_HASH
    assert args.normalization_id == "record_minmax_neg1_1_v1"
    assert args.condition_lead_index == 1
    assert args.target_lead_indices == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert args.experiment_role == "path_ablation"
    assert args.flow_matcher == path_type
    assert args.sigma == pytest.approx(0.1)
    assert args.region_weight == pytest.approx(0.01)
    assert args.use_minibatch_ot is uses_ot
    assert args.epochs == 500 and args.batch_size == 128 and args.save_every == 25
    assert args.validation_interval_epochs == 25
    assert args.heldout_role == "validation"
    assert args.validation_fixed_noise_seed == 2025
    assert args.wandb_mode == "online"


def test_ptbxl_path_configs_change_only_declared_path_coupling_fields():
    baseline = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    allowed = {
        "experiment_role",
        "flow_matcher",
        "sigma",
        "use_minibatch_ot",
        "ot_sampling_strategy",
        "association_debug",
        "wandb_group",
        "wandb_run_name",
    }
    baseline_shared = {key: value for key, value in baseline.items() if key not in allowed}
    for path in sorted(CONFIG_ROOT.glob("rcfm_path_ablation_*_sigma01_seed31.yaml")):
        config = json.loads(path.read_text(encoding="utf-8"))
        assert {key: value for key, value in config.items() if key not in allowed} == baseline_shared


def test_ptbxl_sb_uses_one_observable_exact_multinomial_coupling():
    args = parse_args_with_config(
        ["--config", str(CONFIG_ROOT / "rcfm_path_ablation_sb_exact_multinomial_sigma01_seed31.yaml")]
    )
    assert args.ot_method == "exact"
    assert args.ot_sampling_strategy == "multinomial"
    assert args.ot_diagnostics and args.ot_strict_mode and args.association_debug


@pytest.mark.parametrize(
    ("filename", "updates", "message"),
    (
        ("rcfm_path_ablation_vp_sigma01_seed31.yaml", {"use_minibatch_ot": True}, "must disable OT"),
        ("rcfm_path_ablation_target_sigma01_seed31.yaml", {"use_minibatch_ot": True}, "must disable OT"),
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
def test_ptbxl_invalid_path_protocol_fails_before_data_loading(filename, updates, message):
    args = parse_args_with_config(["--config", str(CONFIG_ROOT / filename)])
    for key, value in updates.items():
        setattr(args, key, value)

    with pytest.raises(ValueError, match=message):
        run_training(
            args,
            lambda *_args, **_kwargs: pytest.fail("invalid protocol reached dataset loading"),
        )
