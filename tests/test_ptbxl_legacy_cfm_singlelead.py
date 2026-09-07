import json
from pathlib import Path

import numpy as np

from train_ptbxl_legacy_cfm_singlelead import (
    CFM_SOURCE_SHA256,
    CONDITION_INDEX,
    LEGACY_COMMIT,
    MODEL_SOURCE_SHA256,
    TARGET_INDEX,
    _load_config,
    _prepare_split,
    _sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ptbxl/cfm_legacy_singlelead_iii_to_v5_official_seed42.yaml"


def test_config_freezes_unchanged_legacy_architecture_and_singlelead_task():
    config = _load_config(CONFIG)
    assert config["architecture_source_commit"] == LEGACY_COMMIT
    assert config["condition_lead"] == "III" and config["condition_lead_index"] == CONDITION_INDEX
    assert config["target_lead"] == "V5" and config["target_lead_index"] == TARGET_INDEX
    assert config["epochs"] == 1000 and config["batch_size"] == 256
    assert config["scheduler"] == "none" and config["amp"] is False


def test_legacy_source_files_match_pinned_commit_hashes():
    assert _sha256(ROOT / "model.py") == MODEL_SOURCE_SHA256
    assert _sha256(ROOT / "train_cfm_basic.py") == CFM_SOURCE_SHA256


def test_split_preparation_selects_iii_to_v5_and_scales_each_record(tmp_path):
    values = np.zeros((2, 520, 12), dtype=np.float32)
    values[0, :512, CONDITION_INDEX] = np.linspace(10, 20, 512)
    values[1, :512, CONDITION_INDEX] = np.linspace(-4, 4, 512)
    values[0, :512, TARGET_INDEX] = np.linspace(100, 120, 512)
    values[1, :512, TARGET_INDEX] = np.linspace(-8, 2, 512)
    path = tmp_path / "split.npy"
    np.save(path, values)
    target, condition = _prepare_split(path)
    assert target.shape == condition.shape == (2, 1, 512)
    np.testing.assert_allclose(condition[:, :, [0, -1]], [[[-1, 1]], [[-1, 1]]])
    np.testing.assert_allclose(target[:, :, [0, -1]], [[[-1, 1]], [[-1, 1]]])


def test_config_is_plain_json_for_reproducible_launcher_use():
    payload = json.loads(CONFIG.read_text())
    assert payload["architecture_id"].startswith("legacy_")
