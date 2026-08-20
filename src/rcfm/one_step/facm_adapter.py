"""Checkpoint/model adapter for FACM acceleration of the existing RCFM U-Net."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from model import ConditionNet, DiffusionUNetCrossAttention
from src.rcfm.checkpoint import load_checkpoint


FACM_UPSTREAM_COMMIT = "8d80d4c65101f814095984a91329ce4aa37be79b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _flow_state(model_state: Mapping[str, Any]) -> dict[str, Any]:
    prefix = "flow_model."
    selected = {
        key[len(prefix) :]: value for key, value in model_state.items() if key.startswith(prefix)
    }
    if not selected or len(selected) != len(model_state):
        raise ValueError("base RCFM model_state must contain only flow_model.* parameters")
    return selected


def validate_facm_base(
    checkpoint: Mapping[str, Any], expected_protocol: Mapping[str, Any]
) -> None:
    """Validate that a FACM teacher is the declared canonical RCFM run."""

    config = checkpoint["config"]
    expected = {
        key: expected_protocol[key]
        for key in (
            "task",
            "dataset_version",
            "split_hash",
            "normalization_id",
            "alignment_id",
        )
    }
    expected.update({
        "flow_matcher": "conditional",
        "sigma": 0.0,
        "region_weight": 0.01,
        "use_minibatch_ot": False,
    })
    mismatched = [key for key, value in expected.items() if config.get(key) != value]
    if checkpoint.get("kind") != "canonical_multistep_rcfm":
        mismatched.append("kind")
    dataset = expected_protocol["datasets"]
    if config.get("datasets") not in ([dataset], dataset):
        mismatched.append("datasets")
    if int(checkpoint.get("epoch", -1)) != int(expected_protocol.get("base_epoch", 500)):
        mismatched.append("epoch")
    output = checkpoint["output_spec"]
    expected_channels = len(expected_protocol.get("target_lead_indices") or [0])
    if output.get("channels") != expected_channels or output.get("length") != 512:
        mismatched.append("output_spec")
    if mismatched:
        raise ValueError(
            "FACM base checkpoint violates the frozen dataset contract: "
            + ", ".join(mismatched)
        )


def validate_mimic_facm_base(checkpoint: Mapping[str, Any]) -> None:
    """Backward-compatible validator for the original MIMIC-only entry point."""

    validate_facm_base(
        checkpoint,
        {
            "task": "ppg2ecg",
            "datasets": "MIMIC-AFib",
            "dataset_version": "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1",
            "split_hash": "a7e388293adaa7b48d3493efc505dd8750520730cae9fd7649157866efa86a51",
            "normalization_id": "rddm_window_minmax_neg1_1_v1",
            "alignment_id": "paired_array_row_rddm_contract_zero_ppg_qc_v1",
            "base_epoch": 500,
        },
    )


@dataclass
class FACMTrainingModels:
    student_flow: nn.Module
    student_condition: nn.Module
    teacher_flow: nn.Module
    teacher_condition: nn.Module
    base_checkpoint: dict[str, Any]
    base_checkpoint_sha256: str

    def trainable_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.student_flow.parameters()) + list(self.student_condition.parameters())
        if not parameters or any(not parameter.requires_grad for parameter in parameters):
            raise RuntimeError("all FACM student parameters must be trainable")
        return parameters


def build_facm_training_models(
    checkpoint_path: Path,
    *,
    device: torch.device,
    expected_sha256: str | None = None,
    expected_protocol: Mapping[str, Any] | None = None,
) -> FACMTrainingModels:
    """Initialize student and frozen teacher from one audited base checkpoint."""

    actual_sha256 = sha256_file(checkpoint_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("base RCFM checkpoint SHA-256 does not match the frozen config")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if expected_protocol is None:
        validate_mimic_facm_base(checkpoint)
    else:
        validate_facm_base(checkpoint, expected_protocol)
    output_spec = checkpoint["output_spec"]
    config = checkpoint["config"]

    def new_flow() -> nn.Module:
        model = DiffusionUNetCrossAttention(
            int(output_spec["length"]),
            int(output_spec["channels"]),
            str(device),
            num_heads=int(config["attention_heads"]),
        )
        model.load_state_dict(_flow_state(checkpoint["model_state"]), strict=True)
        return model.to(device)

    def new_condition() -> nn.Module:
        model = ConditionNet()
        model.load_state_dict(checkpoint["condition_state"], strict=True)
        return model.to(device)

    return FACMTrainingModels(
        student_flow=new_flow(),
        student_condition=new_condition(),
        teacher_flow=freeze_module(new_flow()),
        teacher_condition=freeze_module(new_condition()),
        base_checkpoint=checkpoint,
        base_checkpoint_sha256=actual_sha256,
    )
