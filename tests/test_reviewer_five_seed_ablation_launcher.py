from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "launch_reviewer_five_seed_ablation.sh"
DATASETS = ("ptbxl", "cpsc2018", "mimic-afib", "wesad", "mmecg")
MODELS = ("cfm", "rcfm", "rcfm-ot", "rddm")


def dry_run(tmp_path: Path, dataset: str, model: str, seeds: str = "all") -> str:
    env = os.environ.copy()
    env.update(
        {
            "RCFM_DRY_RUN": "1",
            "RCFM_DATA_ROOT": str(tmp_path / "data"),
            "RCFM_RUNS_ROOT": str(tmp_path / "runs"),
            "RCFM_PYTHON": "python",
            "WANDB_MODE": "disabled",
        }
    )
    result = subprocess.run(
        [str(LAUNCHER), "0", dataset, model, seeds],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("model", MODELS)
def test_full_matrix_has_five_locked_seeds_and_epochs(
    tmp_path: Path, dataset: str, model: str
) -> None:
    output = dry_run(tmp_path, dataset, model)
    lines = [line for line in output.splitlines() if line.startswith("DRY-RUN")]
    assert len(lines) == 5
    expected_epochs = 400 if model == "rddm" else 200
    for seed, line in zip(range(31, 36), lines, strict=True):
        assert f"dataset={dataset} model={model} seed={seed} epochs={expected_epochs}" in line
        assert f"--epochs {expected_epochs}" in line
        assert f"--seed {seed}" in line
        assert f"_s{seed}_e{expected_epochs}_reviewer_5seed_v1" in line
        config_match = re.search(r"--config ([^ ]+)", line)
        assert config_match is not None
        assert (REPO_ROOT / config_match.group(1)).is_file()


def test_model_ablation_contracts_are_explicit(tmp_path: Path) -> None:
    cfm = dry_run(tmp_path, "ptbxl", "cfm", "31")
    rcfm = dry_run(tmp_path, "ptbxl", "rcfm", "31")
    rcfm_ot = dry_run(tmp_path, "ptbxl", "rcfm-ot", "31")

    assert "train_cfm_compare.py" in cfm
    assert "--region_weight 0" in cfm
    assert "--no-use_minibatch_ot" in cfm
    assert "--no-use_minibatch_ot" in rcfm
    assert "--use_minibatch_ot" in rcfm_ot
    assert "--ot_method exact" in rcfm_ot
    assert "--ot_sampling_strategy assignment" in rcfm_ot


def test_rejects_seed_outside_prespecified_set(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "RCFM_DRY_RUN": "1",
            "RCFM_DATA_ROOT": str(tmp_path / "data"),
            "RCFM_RUNS_ROOT": str(tmp_path / "runs"),
        }
    )
    result = subprocess.run(
        [str(LAUNCHER), "0", "wesad", "rcfm", "36"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "locked to 31,32,33,34,35" in result.stderr


def test_ecg_to_ecg_path_has_no_phase_correction_option(tmp_path: Path) -> None:
    for dataset in ("ptbxl", "cpsc2018"):
        output = dry_run(tmp_path, dataset, "rcfm-ot", "31")
        assert "phase" not in output.lower()
