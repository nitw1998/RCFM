import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from rcfm import RegionAwareConditionalFlowMatching
from src.rcfm.checkpoint import load_checkpoint
from src.rcfm.training import run_training
from train_cfm_compare import parse_args_with_config
from train_rcfm import parse_args_with_config as parse_rcfm_args_with_config


class ZeroFlow(nn.Module):
    def forward(self, value, conditions, time):
        del conditions, time
        return torch.zeros_like(value)


class TinyFlow(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.projection = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, value, conditions, time):
        del conditions
        return self.projection(value) + time.view(-1, 1, 1) * 0.0


class TinyCondition(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, value):
        feature = self.projection(value)
        return {"down_conditions": [feature], "up_conditions": [feature]}


class TinyPairedDataset(Dataset):
    normalization_metadata = {
        "method": "record_zscore",
        "stats_scope": "per_record_per_lead",
        "stats_source": "dataset_sidecar",
        "inverse_transform": "x=x_z*record_scale+record_mean",
        "generated_inverse_policy": "ground_truth_target_scaler_is_oracle_only",
    }

    def __init__(self):
        self.target = torch.linspace(-1.0, 1.0, 32).reshape(4, 1, 8)
        self.condition = torch.flip(self.target, dims=(-1,))
        self.mask = torch.zeros_like(self.target)
        self.mask[:, :, 2:5] = 1.0

    def __getitem__(self, index):
        return self.target[index], self.condition[index], self.mask[index]

    def __len__(self):
        return len(self.target)


def _config_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1] / "configs" / "cpsc2018"
    return (
        root / "cfm_compare_record_zscore_no_ot_seed31.yaml",
        root / "rcfm_record_zscore_no_ot_seed31.yaml",
    )


def test_cfm_compare_config_matches_rcfm_except_declared_factor_and_labels():
    cfm_path, rcfm_path = _config_paths()
    cfm = json.loads(cfm_path.read_text(encoding="utf-8"))
    rcfm = json.loads(rcfm_path.read_text(encoding="utf-8"))
    allowed_differences = {"region_weight", "save_every", "wandb_group"}

    assert {
        key: value for key, value in cfm.items() if key not in allowed_differences
    } == {
        key: value for key, value in rcfm.items() if key not in allowed_differences
    }
    parsed = parse_args_with_config(["--config", str(cfm_path)])
    assert parsed.model_family == "CFM"
    assert parsed.region_weight == 0.0
    assert parsed.use_minibatch_ot is False
    assert parsed.batch_size == 128
    assert parsed.seed == 31


def test_minmax_configs_match_and_freeze_long_schedule():
    root = Path(__file__).resolve().parents[1] / "configs" / "cpsc2018"
    cfm_path = root / "cfm_compare_record_minmax_neg1_1_no_ot_seed31.yaml"
    rcfm_path = root / "rcfm_record_minmax_neg1_1_no_ot_seed31.yaml"
    cfm = json.loads(cfm_path.read_text(encoding="utf-8"))
    rcfm = json.loads(rcfm_path.read_text(encoding="utf-8"))
    allowed_differences = {"region_weight", "wandb_group"}

    assert {
        key: value for key, value in cfm.items() if key not in allowed_differences
    } == {
        key: value for key, value in rcfm.items() if key not in allowed_differences
    }
    assert cfm["normalization_id"] == "record_minmax_neg1_1_v1"
    assert cfm["batch_size"] == 128
    assert cfm["epochs"] == 500
    assert cfm["save_every"] == 25


def test_minmax_rcfm_exact_ot_config_changes_only_coupling_and_label():
    root = Path(__file__).resolve().parents[1] / "configs" / "cpsc2018"
    no_ot_path = root / "rcfm_record_minmax_neg1_1_no_ot_seed31.yaml"
    exact_ot_path = root / "rcfm_record_minmax_neg1_1_exact_ot_seed31.yaml"
    no_ot = json.loads(no_ot_path.read_text(encoding="utf-8"))
    exact_ot = json.loads(exact_ot_path.read_text(encoding="utf-8"))
    allowed_differences = {
        "use_minibatch_ot",
        "ot_sampling_strategy",
        "wandb_group",
    }

    assert {
        key: value for key, value in no_ot.items() if key not in allowed_differences
    } == {
        key: value for key, value in exact_ot.items() if key not in allowed_differences
    }
    parsed = parse_rcfm_args_with_config(["--config", str(exact_ot_path)])
    assert parsed.normalization_id == "record_minmax_neg1_1_v1"
    assert parsed.use_minibatch_ot is True
    assert parsed.ot_method == "exact"
    assert parsed.ot_sampling_strategy == "assignment"
    assert parsed.ot_strict_mode is True
    assert parsed.batch_size == 128
    assert parsed.epochs == 500
    assert parsed.seed == 31


def test_ptbxl_configs_freeze_official_waveform_only_protocol():
    root = Path(__file__).resolve().parents[1] / "configs" / "ptbxl"
    cfm_path = root / "cfm_compare_record_minmax_neg1_1_no_ot_seed31.yaml"
    no_ot_path = root / "rcfm_record_minmax_neg1_1_no_ot_seed31.yaml"
    exact_ot_path = root / "rcfm_record_minmax_neg1_1_exact_ot_seed31.yaml"
    configs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (cfm_path, no_ot_path, exact_ot_path)
    ]

    allowed_differences = {
        "region_weight", "use_minibatch_ot", "ot_sampling_strategy", "wandb_group"
    }
    shared = [
        {key: value for key, value in config.items() if key not in allowed_differences}
        for config in configs
    ]
    assert shared[0] == shared[1] == shared[2]
    for config in configs:
        assert config["datasets"] == "PTBXL"
        assert config["dataset_version"] == (
            "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1"
        )
        assert config["split_hash"] == (
            "7784c98a2c8daccc23fc7cb0d47dc1933eeee77f120c05f8a24ac149cd8474f7"
        )
        assert config["normalization_id"] == "record_minmax_neg1_1_v1"
        assert config["condition_lead_index"] == 1
        assert config["target_lead_indices"] == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        assert config["epochs"] == 500
        assert config["batch_size"] == 128
        assert config["save_every"] == 25
        assert config["validation_interval_epochs"] == 25
        assert config["heldout_role"] == "validation"
        assert config["wandb_mode"] == "online"
        assert all("semantic" not in key.lower() for key in config)
        assert all("ptb-xl-plus" not in str(value).lower() for value in config.values())

    parsed_cfm = parse_args_with_config(["--config", str(cfm_path)])
    parsed_no_ot = parse_rcfm_args_with_config(["--config", str(no_ot_path)])
    parsed_exact_ot = parse_rcfm_args_with_config(["--config", str(exact_ot_path)])
    assert parsed_cfm.model_family == "CFM"
    assert parsed_cfm.region_weight == 0.0
    assert parsed_cfm.use_minibatch_ot is False
    assert parsed_no_ot.region_weight == pytest.approx(0.01)
    assert parsed_no_ot.use_minibatch_ot is False
    assert parsed_exact_ot.region_weight == pytest.approx(0.01)
    assert parsed_exact_ot.use_minibatch_ot is True
    assert parsed_exact_ot.ot_sampling_strategy == "assignment"


def test_ptbxl_diagmask_pair_changes_only_exact_coupling_fields():
    root = Path(__file__).resolve().parents[1] / "configs" / "ptbxl"
    no_ot_path = root / "rcfm_diagmask_record_minmax_no_ot_seed31.yaml"
    exact_ot_path = root / "rcfm_diagmask_record_minmax_exact_ot_seed31.yaml"
    no_ot = json.loads(no_ot_path.read_text(encoding="utf-8"))
    exact_ot = json.loads(exact_ot_path.read_text(encoding="utf-8"))
    allowed = {"use_minibatch_ot", "ot_sampling_strategy"}

    assert {key: value for key, value in no_ot.items() if key not in allowed} == {
        key: value for key, value in exact_ot.items() if key not in allowed
    }
    assert no_ot["mask_method"] == (
        "xresnet1d101_all_positive_statements_gradcam_l5_sample_center_v1"
    )
    assert no_ot["use_minibatch_ot"] is False
    assert exact_ot["use_minibatch_ot"] is True
    assert exact_ot["ot_method"] == "exact"
    assert exact_ot["ot_sampling_strategy"] == "assignment"
    for config in (no_ot, exact_ot):
        assert config["epochs"] == 500
        assert config["batch_size"] == 128
        assert config["save_every"] == 25
        assert config["target_lead_indices"] == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_cfm_compare_rejects_region_weight_and_minibatch_ot():
    cfm_path, _ = _config_paths()
    with pytest.raises(ValueError, match="region_weight=0"):
        parse_args_with_config(["--config", str(cfm_path), "--region_weight", "0.01"])
    with pytest.raises(ValueError, match="minibatch OT"):
        parse_args_with_config(["--config", str(cfm_path), "--use_minibatch_ot"])


def test_cfm_compare_keeps_roi_metrics_diagnostic_only():
    model = RegionAwareConditionalFlowMatching(
        ZeroFlow(),
        region_weight=0.0,
        use_minibatch_ot=False,
    )
    target = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    source = torch.zeros_like(target)
    mask = torch.tensor([[[0.0, 1.0, 1.0, 0.0]]])
    conditions = {
        "down_conditions": [torch.zeros_like(target)],
        "up_conditions": [torch.zeros_like(target)],
    }

    output = model(
        target=target,
        conditions=conditions,
        source=source,
        region_mask=mask,
    )

    torch.testing.assert_close(output["loss"], output["train/velocity_mse"])
    assert float(output["train/mask_occupancy"]) == pytest.approx(0.5)
    assert float(output["train/effective_mean_weight"]) == pytest.approx(1.0)
    assert torch.isfinite(output["train/roi_mse"])
    assert torch.isfinite(output["train/non_roi_mse"])


def test_cfm_compare_runs_shared_metric_and_checkpoint_interfaces(tmp_path, monkeypatch):
    import src.rcfm.training as training_module

    monkeypatch.setattr(training_module, "ConditionNet", TinyCondition)
    monkeypatch.setattr(training_module, "DiffusionUNetCrossAttention", TinyFlow)
    cfm_path, _ = _config_paths()
    args = parse_args_with_config(["--config", str(cfm_path)])
    args.output_dir = str(tmp_path)
    args.data_root = "/synthetic/not-used"
    args.run_id = "synthetic-cfm"
    args.target_lead = "V5"
    args.target_lead_index = 10
    args.target_lead_indices = [10]
    args.window_size = 0.0625
    args.epochs = 1
    args.batch_size = 2
    args.num_workers = 0
    args.warmup_epochs = 0
    args.save_every = 1
    args.checkpoint_policy = "latest_only"
    args.inference_steps = 2
    args.validation_interval_epochs = 1
    args.validation_max_batches = 1
    args.max_batches = 1
    args.device = "cpu"
    args.wandb_mode = "disabled"

    def dataset_builder(*unused_args, **unused_kwargs):
        return TinyPairedDataset(), TinyPairedDataset()

    run_training(args, dataset_builder)

    run_dir = tmp_path / "ecg2ecg" / "CPSC2018" / "synthetic-cfm"
    resolved = json.loads((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(run_dir / "checkpoint_latest.pt", map_location="cpu")
    epoch_text = (run_dir / "epoch_metrics.csv").read_text(encoding="utf-8")
    validation_text = (run_dir / "validation_metrics.csv").read_text(encoding="utf-8")

    assert resolved["model_family"] == "CFM"
    assert resolved["comparison_role"] == "preprocessing_matched_cfm_control"
    assert resolved["mask_method"] == "cached_target_ecg_r_peak_roi_diagnostics_only"
    assert resolved["mask_usage"] == "training_diagnostics_only"
    assert checkpoint["kind"] == "canonical_multistep_cfm"
    assert "train/total_loss" in epoch_text
    assert "train/velocity_mse" in epoch_text
    assert "train/roi_mse" in epoch_text
    assert "val/rmse" in validation_text
    assert "val/velocity_mse" in validation_text
    assert "val/waveform_fd" in validation_text


def test_mimic_flow_configs_are_matched_except_declared_region_and_ot_factors():
    root = Path(__file__).resolve().parents[1] / "configs" / "mimic_afib"
    paths = {
        "cfm": root / "cfm_rddm_minmax_zero_qc_no_ot_seed31.yaml",
        "rcfm": root / "rcfm_rddm_minmax_zero_qc_no_ot_seed31.yaml",
        "rcfm_ot": root / "rcfm_rddm_minmax_zero_qc_exact_ot_seed31.yaml",
    }
    configs = {
        name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()
    }
    shared_fields = {
        "task", "datasets", "dataset_version", "split_hash", "normalization_id",
        "alignment_id", "window_size", "epochs", "batch_size", "save_every", "seed",
        "validation_interval_epochs", "heldout_role",
    }
    reference = {field: configs["cfm"][field] for field in shared_fields}
    assert all(
        {field: config[field] for field in shared_fields} == reference
        for config in configs.values()
    )
    assert reference["epochs"] == 500
    assert reference["batch_size"] == 128
    assert reference["save_every"] == 25
    assert configs["cfm"]["region_weight"] == 0.0
    assert configs["cfm"]["use_minibatch_ot"] is False
    assert configs["rcfm"]["region_weight"] == pytest.approx(0.01)
    assert configs["rcfm"]["use_minibatch_ot"] is False
    assert configs["rcfm_ot"]["region_weight"] == pytest.approx(0.01)
    assert configs["rcfm_ot"]["use_minibatch_ot"] is True
    assert configs["rcfm_ot"]["ot_method"] == "exact"
    assert configs["rcfm_ot"]["ot_sampling_strategy"] == "assignment"

    cfm_args = parse_args_with_config(["--config", str(paths["cfm"])])
    ot_args = parse_rcfm_args_with_config(["--config", str(paths["rcfm_ot"])])
    assert cfm_args.model_family == "CFM"
    assert ot_args.use_minibatch_ot is True
