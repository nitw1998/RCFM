import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.prepare_ptbxl_semantic_resunet_masks import (
    METHOD,
    SPLIT_HASH,
    _validate_checkpoint,
    semantic_soft_mask,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_semantic_soft_mask_is_maximum_p_qrs_t_probability():
    probabilities = torch.tensor(
        [[[0.1, 0.8], [0.7, 0.2], [0.3, 0.4]]], dtype=torch.float32
    )
    logits = torch.logit(probabilities)
    masks = semantic_soft_mask(logits)

    assert masks.shape == (1, 1, 2)
    torch.testing.assert_close(masks, torch.tensor([[[0.7, 0.8]]]))


def test_semantic_soft_mask_rejects_wrong_region_channels():
    with pytest.raises(ValueError, match="batch, 3, samples"):
        semantic_soft_mask(torch.zeros(2, 2, 16))


def test_semantic_checkpoint_contract_binds_fold9_selection_and_sidecar(tmp_path: Path):
    sidecar_path = tmp_path / "dataset_manifest.json"
    sidecar = {
        "waveform_split_hash": SPLIT_HASH,
        "split_and_eligibility_hash": "eligibility-hash",
    }
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    checkpoint = {
        "kind": "ptbxl_plus_delineation_resunet1d",
        "epoch": 13,
        "config": {
            "sample_rate_hz": 128,
            "window_samples": 512,
            "waveform_split_hash": SPLIT_HASH,
        },
        "output_spec": {"region_classes": ["p", "qrs", "t"]},
        "provenance": {
            "dataset_manifest_sha256": _sha256(sidecar_path),
            "split_and_eligibility_hash": "eligibility-hash",
        },
        "model_state": {"weight": torch.ones(1)},
    }

    config, loaded_sidecar = _validate_checkpoint(
        checkpoint, tmp_path / "checkpoint.pt", sidecar_path, expected_epoch=13
    )
    assert config["window_samples"] == 512
    assert loaded_sidecar == sidecar

    checkpoint["epoch"] = 50
    with pytest.raises(ValueError, match="fold-9 selection"):
        _validate_checkpoint(
            checkpoint, tmp_path / "checkpoint.pt", sidecar_path, expected_epoch=13
        )


def test_semanticmask_training_config_is_a_single_factor_no_ot_ablation():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/ptbxl/rcfm_semanticmask_record_minmax_no_ot_seed31.yaml"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["mask_method"] == METHOD
    assert config["region_weight"] == 0.01
    assert config["use_minibatch_ot"] is False
    assert config["flow_matcher"] == "conditional"
    assert config["normalization_id"] == "record_minmax_neg1_1_v1"
    assert config["target_lead_indices"] == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert config["batch_size"] == 128
    assert config["epochs"] == 500
    assert config["save_every"] == 25
