import csv
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from src.rcfm.experiment import LOCAL_FILE_HEADERS, RunArtifacts, WandbLogger
from src.rcfm.training import _build_scheduler, _build_training_loader, run_training
from train_rcfm import parse_args_with_config


def test_run_artifacts_create_required_local_schema(tmp_path: Path):
    artifacts = RunArtifacts(
        tmp_path,
        {"task": "synthetic", "use_minibatch_ot": True},
        {"status": "running"},
        "python=test",
        "commit=deadbeef",
    )
    required = {
        "resolved_config.yaml",
        "run_metadata.json",
        "epoch_metrics.csv",
        "ot_diagnostics.csv",
        "validation_metrics.csv",
        "clinical_metrics.csv",
        "checkpoint_manifest.json",
        "environment.txt",
        "git_state.txt",
    }
    assert required == {path.name for path in tmp_path.iterdir()}
    assert json.loads((tmp_path / "resolved_config.yaml").read_text())["task"] == "synthetic"

    artifacts.append_metrics(
        "ot_diagnostics.csv",
        epoch=1,
        global_step=2,
        metrics={"ot/plan_mass": 1.0},
    )
    with (tmp_path / "ot_diagnostics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"epoch": "1", "global_step": "2", "metric": "ot/plan_mass", "value": "1.0"}
    ]
    assert list(rows[0]) == LOCAL_FILE_HEADERS["ot_diagnostics.csv"]


def test_wandb_disabled_does_not_create_wandb_files(tmp_path: Path):
    logger = WandbLogger(
        mode="disabled",
        run_dir=tmp_path,
        config={"task": "synthetic"},
        project="RCFM-test",
        group=None,
        job_type="test",
        run_name="disabled",
    )
    logger.log({"train/total_loss": 1.0}, step=0)
    logger.finish({"status": "completed"})
    assert logger.run is None
    assert not (tmp_path / "wandb").exists()


def test_wandb_offline_mode_requires_no_login(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_SILENT", "true")
    calls = {}

    class FakeRun:
        def __init__(self):
            self.summary = {}

        def log(self, metrics, step):
            calls["log"] = (metrics, step)

        def finish(self, exit_code):
            calls["exit_code"] = exit_code

    def fake_init(**kwargs):
        calls["init"] = kwargs
        return FakeRun()

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=fake_init, Histogram=lambda values: values),
    )
    logger = WandbLogger(
        mode="offline",
        run_dir=tmp_path,
        config={"task": "synthetic", "data_root": "/private/data"},
        project="RCFM-test",
        group="synthetic",
        job_type="test",
        run_name="offline",
    )
    logger.log({"train/total_loss": 1.0}, step=0)
    logger.finish({"status": "completed"})
    assert logger.run is not None
    assert calls["init"]["mode"] == "offline"
    assert "data_root" not in calls["init"]["config"]
    assert calls["exit_code"] == 0


def test_five_ot_templates_are_controlled_and_cli_overridable():
    config_root = Path(__file__).resolve().parents[1] / "configs" / "ot_experiments"
    paths = sorted(config_root.glob("*.yaml"))
    assert len(paths) == 5
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    experimental_factors = {
        "region_weight", "use_minibatch_ot", "ot_method", "ot_sampling_strategy"
    }
    shared = [
        {key: value for key, value in config.items() if key not in experimental_factors}
        for config in configs
    ]
    assert all(config == shared[0] for config in shared[1:])
    assert configs[1]["ot_method"] == "exact"
    assert configs[1]["ot_sampling_strategy"] == "assignment"
    assert configs[4]["ot_method"] == "exact"
    assert configs[4]["ot_sampling_strategy"] == "assignment"
    assert configs[2]["ot_method"] == "sinkhorn"

    parsed = parse_args_with_config(["--config", str(paths[0]), "--seed", "47"])
    assert parsed.seed == 47
    assert parsed.sigma == 0.0
    assert parsed.flow_matcher == "conditional"


def test_cpsc2018_full_config_is_uncapped_and_wandb_online():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "cpsc2018"
        / "rcfm_record_zscore_no_ot_seed31.yaml"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parsed = parse_args_with_config(["--config", str(config_path)])

    assert "max_batches" not in config
    assert "max_train_records" not in config
    assert "max_heldout_records" not in config
    assert parsed.max_batches is None
    assert parsed.max_train_records is None
    assert parsed.max_heldout_records is None
    assert parsed.checkpoint_policy == "full"
    assert parsed.wandb_mode == "online"
    assert parsed.wandb_project == "RCFM"
    assert parsed.normalization_id == "record_zscore_v1"
    assert parsed.seed == 31
    assert parsed.use_minibatch_ot is False
    assert parsed.region_weight == pytest.approx(0.01)


def test_all_cpsc_flow_training_configs_use_lead_ii_to_joint_other_eleven_protocol():
    config_root = Path(__file__).resolve().parents[1] / "configs" / "cpsc2018"
    expected_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    expected_leads = [
        "I", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
    ]
    paths = sorted(path for path in config_root.glob("*.yaml") if not path.name.startswith("rddm_"))

    assert len(paths) == 6
    for path in paths:
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["condition_lead"] == "II"
        assert config["condition_lead_index"] == 1
        assert config["target_lead_index"] is None
        assert config["target_lead_indices"] == expected_indices
        assert config["target_lead"].split(",") == expected_leads
        assert "lead_II_to_other_11" in config["alignment_id"]
        assert "source-derived-all12lead-qc-v3" in config["dataset_version"]
        assert config["split_hash"] == (
            "35e0a796a60e4d6979b1f050048495fdf1826eedda7d40e47a56d4dcd5874223"
        )


def test_mimic_afib_rddm_no_ot_config_is_frozen_for_final_test_only():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "mimic_afib"
        / "rcfm_rddm_minmax_zero_qc_no_ot_seed31.yaml"
    )
    parsed = parse_args_with_config(["--config", str(config_path)])

    assert parsed.datasets == "MIMIC-AFib"
    assert parsed.normalization_id == "rddm_window_minmax_neg1_1_v1"
    assert parsed.batch_size == 128
    assert parsed.epochs == 500
    assert parsed.validation_interval_epochs == 500
    assert parsed.heldout_role == "upstream_test_final_only"
    assert parsed.use_minibatch_ot is False
    assert parsed.region_weight == pytest.approx(0.01)
    assert parsed.wandb_mode == "online"


def test_upstream_test_role_rejects_intermediate_evaluation():
    args = SimpleNamespace(
        flow_matcher="conditional",
        sigma=0.0,
        region_weight=0.01,
        use_minibatch_ot=False,
        validation_interval_epochs=10,
        inference_steps=50,
        heldout_role="upstream_test_final_only",
        epochs=500,
    )

    with pytest.raises(ValueError, match="requires validation_interval_epochs to equal epochs"):
        run_training(args, lambda *_args, **_kwargs: pytest.fail("must fail before loading data"))


def test_zero_warmup_does_not_reduce_first_step_learning_rate():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)

    _build_scheduler(optimizer, epochs=1, warmup_epochs=0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_scheduler_rejects_invalid_epoch_configuration():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)

    with pytest.raises(ValueError, match="epochs must be positive"):
        _build_scheduler(optimizer, epochs=0, warmup_epochs=0)


def test_exact_ot_training_loader_keeps_final_incomplete_batch():
    dataset = TensorDataset(torch.arange(5))
    loader = _build_training_loader(
        dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )

    assert sorted(len(batch[0]) for batch in loader) == [1, 4]
