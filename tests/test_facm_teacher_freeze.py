from __future__ import annotations

import torch
from torch import nn

from src.rcfm.one_step.facm_adapter import (
    freeze_module,
    validate_facm_base,
    validate_mimic_facm_base,
)


def test_freeze_module_disables_training_and_gradients() -> None:
    module = nn.Sequential(nn.Linear(3, 4), nn.Dropout())
    frozen = freeze_module(module)
    assert not frozen.training
    assert all(not parameter.requires_grad for parameter in frozen.parameters())


def test_mimic_base_contract_rejects_ot_teacher() -> None:
    checkpoint = {
        "kind": "canonical_multistep_rcfm",
        "epoch": 500,
        "config": {
            "task": "ppg2ecg",
            "datasets": ["MIMIC-AFib"],
            "dataset_version": "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1",
            "split_hash": "a7e388293adaa7b48d3493efc505dd8750520730cae9fd7649157866efa86a51",
            "normalization_id": "rddm_window_minmax_neg1_1_v1",
            "flow_matcher": "conditional",
            "sigma": 0.0,
            "region_weight": 0.01,
            "use_minibatch_ot": True,
        },
        "output_spec": {"channels": 1, "length": 512},
    }
    try:
        validate_mimic_facm_base(checkpoint)
    except ValueError as error:
        assert "use_minibatch_ot" in str(error)
    else:
        raise AssertionError("OT-trained teacher must not pass the principal FACM contract")


def test_multilead_facm_base_accepts_matching_canonical_teacher() -> None:
    target_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    checkpoint = {
        "kind": "canonical_multistep_rcfm",
        "epoch": 500,
        "config": {
            "task": "ecg2ecg",
            "datasets": ["PTBXL"],
            "dataset_version": "version",
            "split_hash": "split",
            "normalization_id": "record_minmax_neg1_1_v1",
            "alignment_id": "alignment",
            "flow_matcher": "conditional",
            "sigma": 0.0,
            "region_weight": 0.01,
            "use_minibatch_ot": False,
        },
        "output_spec": {"channels": 11, "length": 512},
    }
    validate_facm_base(
        checkpoint,
        {
            "task": "ecg2ecg",
            "datasets": "PTBXL",
            "dataset_version": "version",
            "split_hash": "split",
            "normalization_id": "record_minmax_neg1_1_v1",
            "alignment_id": "alignment",
            "target_lead_indices": target_indices,
        },
    )
