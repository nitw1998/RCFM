from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "facm_ptbxl.yaml": ("PTBXL", 17440, 2193),
    "facm_cpsc2018.yaml": ("CPSC2018", 5487, 686),
    "facm_mimic.yaml": ("MIMIC-AFib", 8400, 1800),
    "facm_wesad.yaml": ("WESAD", 17494, 4213),
    "facm_mmecg.yaml": ("mmECG", 9590, 2877),
}
CFM50_CONFIGS = {
    "facm_cfm50_ptbxl.yaml": ("PTBXL", 34939, 8735),
    "facm_cfm50_cpsc2018.yaml": ("CPSC2018", 19364, 4842),
    "facm_cfm50_mimic.yaml": ("MIMIC-AFib", 8160, 2040),
    "facm_cfm50_wesad.yaml": ("WESAD", 17365, 4342),
    "facm_cfm50_mmecg.yaml": ("mmECG", 9973, 2494),
}


def _entry_module():
    path = ROOT / "scripts" / "train_facm_acceleration.py"
    spec = importlib.util.spec_from_file_location("facm_entry_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_five_dataset_configs_are_complete_and_validate() -> None:
    module = _entry_module()
    for name, (dataset, train_count, heldout_count) in CONFIGS.items():
        path = ROOT / "configs" / "one_step" / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["datasets"] == dataset
        assert payload["model_name"] == "RCFM-OneStep"
        assert payload["expected_train_windows"] == train_count
        assert payload["expected_heldout_windows"] == heldout_count
        assert len(payload["expected_base_checkpoint_sha256"]) == 64
        args = module.parse_args(
            [
                "--config", str(path),
                "--teacher_checkpoint", "teacher.pt",
                "--data_root", "data",
                "--output_dir", "runs",
            ]
        )
        module.validate_args(args)


def test_cfm50_five_dataset_configs_are_complete_and_validate() -> None:
    module = _entry_module()
    for name, (dataset, train_count, heldout_count) in CFM50_CONFIGS.items():
        path = ROOT / "configs" / "one_step" / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["datasets"] == dataset
        assert payload["base_model_family"] == "cfm_nfe50"
        assert payload["expected_train_windows"] == train_count
        assert payload["expected_heldout_windows"] == heldout_count
        assert len(payload["expected_base_checkpoint_sha256"]) == 64
        args = module.parse_args(
            [
                "--config", str(path),
                "--teacher_checkpoint", "teacher.pt",
                "--data_root", "data",
                "--output_dir", "runs",
            ]
        )
        module.validate_args(args)


def test_aggregate_launcher_lists_all_five_datasets() -> None:
    launcher = (ROOT / "scripts" / "launch_rcfm_onestep_five_dataset.sh").read_text(
        encoding="utf-8"
    )
    worker = (ROOT / "scripts" / "run_rcfm_onestep_five_dataset_worker.sh").read_text(
        encoding="utf-8"
    )
    for key in ("ptbxl", "cpsc2018", "mimic_afib", "wesad", "mmecg"):
        assert key in launcher
        assert key in worker
    assert "DATASETS=(ptbxl cpsc2018 mimic_afib wesad mmecg)" in launcher
    assert "rcfm_onestep_cfm50_v2" in launcher
    for name in CFM50_CONFIGS:
        assert name in worker
