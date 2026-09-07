"""Structured canonical RCFM checkpoints with validation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REQUIRED_CONFIG_FIELDS = {
    "task",
    "datasets",
    "dataset_version",
    "seed",
    "split_hash",
    "flow_matcher",
    "sigma",
    "region_weight",
    "use_minibatch_ot",
    "ot_method",
    "ot_reg",
    "normalization_id",
    "condition_unit",
    "target_unit",
    "alignment_id",
    "condition_lead",
    "target_lead",
    "condition_lead_index",
    "target_lead_index",
    "window_size",
    "attention_heads",
}

CHECKPOINT_KINDS = {
    "canonical_multistep_cfm": "CFM",
    "canonical_multistep_cfm_ot": "CFM",
    "canonical_multistep_rcfm": "RCFM",
    "path_ablation_rcfm": "RCFM",
}


def validate_checkpoint(payload: Mapping[str, Any]) -> None:
    schema_version = payload.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError("checkpoint schema_version must be 1 or 2")
    kind = payload.get("kind")
    if kind not in CHECKPOINT_KINDS:
        raise ValueError(
            "checkpoint kind must be canonical_multistep_cfm, canonical_multistep_cfm_ot, "
            "canonical_multistep_rcfm, or path_ablation_rcfm"
        )
    for field in (
        "epoch",
        "model_state",
        "condition_state",
        "optimizer_state",
        "scheduler_state",
        "config",
        "normalization",
        "output_spec",
        "provenance",
    ):
        if field not in payload:
            raise ValueError(f"checkpoint is missing {field}")
    config = payload["config"]
    missing = sorted(REQUIRED_CONFIG_FIELDS - set(config))
    if missing:
        raise ValueError(f"checkpoint config is missing: {', '.join(missing)}")
    if kind == "path_ablation_rcfm":
        if config.get("experiment_role") != "path_ablation":
            raise ValueError("path-ablation checkpoint requires experiment_role=path_ablation")
        path_type = config["flow_matcher"]
        if path_type not in {"vp", "target", "sb"} or float(config["sigma"]) != 0.1:
            raise ValueError("path-ablation checkpoint has the wrong path or sigma")
        if path_type in {"vp", "target"} and bool(config["use_minibatch_ot"]):
            raise ValueError("VP/target path-ablation checkpoint must not use OT")
        if path_type == "sb" and (
            not bool(config["use_minibatch_ot"])
            or config["ot_method"] != "exact"
            or config.get("ot_sampling_strategy") != "multinomial"
        ):
            raise ValueError("SB path-ablation checkpoint requires external exact multinomial OT")
    elif config["flow_matcher"] != "conditional" or float(config["sigma"]) != 0.0:
        raise ValueError("checkpoint does not use the canonical linear multistep path")
    model_family = str(config.get("model_family", "RCFM")).upper()
    if model_family != CHECKPOINT_KINDS[kind]:
        raise ValueError("checkpoint kind and config model_family disagree")
    if model_family == "CFM" and float(config["region_weight"]) != 0.0:
        raise ValueError("canonical CFM checkpoint requires region_weight=0")
    if kind == "canonical_multistep_cfm" and bool(config["use_minibatch_ot"]):
        raise ValueError("canonical CFM checkpoint requires minibatch OT disabled")
    if kind == "canonical_multistep_cfm_ot" and not bool(config["use_minibatch_ot"]):
        raise ValueError("CFM-OT checkpoint requires minibatch OT enabled")
    if config["task"] == "ecg2ecg":
        target_indices = config.get("target_lead_indices")
        if target_indices is None and config.get("target_lead_index") is not None:
            target_indices = [config["target_lead_index"]]
        if config["condition_lead_index"] is None or not target_indices:
            raise ValueError("ecg2ecg checkpoint requires condition and target lead indices")
        indices = [int(config["condition_lead_index"]), *(int(i) for i in target_indices)]
        if min(indices) < 0 or len(set(indices)) != len(indices):
            raise ValueError("ecg2ecg checkpoint lead indices must be nonnegative and unique")
    normalization = payload["normalization"]
    normalization_fields = {"method", "normalization_id", "condition_unit", "target_unit"}
    if not isinstance(normalization, Mapping):
        raise ValueError("checkpoint normalization must be a mapping")
    missing_normalization = sorted(normalization_fields - set(normalization))
    if missing_normalization:
        raise ValueError(
            "checkpoint normalization is missing: " + ", ".join(missing_normalization)
        )
    if normalization["method"] == "training_global_zscore":
        scalar_fields = {
            "target_mean", "target_scale", "condition_mean", "condition_scale"
        }
        missing_scalars = sorted(scalar_fields - set(normalization))
        if missing_scalars:
            raise ValueError(
                "checkpoint global normalization is missing: " + ", ".join(missing_scalars)
            )
        if (
            float(normalization["target_scale"]) <= 0
            or float(normalization["condition_scale"]) <= 0
        ):
            raise ValueError("checkpoint normalization scales must be positive")
    elif normalization["method"] == "record_zscore":
        expected = {
            "stats_scope": "per_record_per_lead",
            "stats_source": "dataset_sidecar",
            "inverse_transform": "x=x_z*record_scale+record_mean",
            "generated_inverse_policy": "ground_truth_target_scaler_is_oracle_only",
        }
        mismatched = [key for key, value in expected.items() if normalization.get(key) != value]
        if mismatched:
            raise ValueError(
                "checkpoint record normalization has invalid fields: " + ", ".join(mismatched)
            )
    elif normalization["method"] == "record_minmax_neg1_1":
        expected = {
            "stats_scope": "per_record_per_lead",
            "stats_source": "selected_waveform_record",
            "feature_range": [-1.0, 1.0],
            "inverse_transform": "x=(x_scaled+1)*record_range/2+record_min",
            "generated_inverse_policy": "ground_truth_target_scaler_is_oracle_only",
        }
        mismatched = [key for key, value in expected.items() if normalization.get(key) != value]
        if mismatched:
            raise ValueError(
                "checkpoint record min-max normalization has invalid fields: "
                + ", ".join(mismatched)
            )
    elif normalization["method"] == "record_joint12_minmax_neg1_1":
        expected = {
            "stats_scope": "per_record_shared_all_12_leads",
            "stats_source": "dataset_sidecar_fixed_first_model_window",
            "feature_range": [-1.0, 1.0],
            "inverse_transform": "x=(x_scaled+1)*record_joint_range/2+record_joint_min",
            "preserves_interlead_relative_amplitudes_and_offsets": True,
            "heldout_target_statistics_used": True,
            "deployment_scope": "paired_benchmark_only_not_lead_II_only_inference",
            "generated_inverse_policy": "ground_truth_joint_12lead_scaler_is_oracle_only",
        }
        mismatched = [key for key, value in expected.items() if normalization.get(key) != value]
        if mismatched:
            raise ValueError(
                "checkpoint joint-12-lead min-max normalization has invalid fields: "
                + ", ".join(mismatched)
            )
    elif normalization["method"] == "source_record_joint12_minmax_neg1_1":
        expected = {
            "stats_scope": "per_source_record_shared_full_10s_all_12_leads",
            "stats_source": "dataset_sidecar_full_source_record",
            "feature_range": [-1.0, 1.0],
            "inverse_transform": "x=(x_scaled+1)*source_record_joint_range/2+source_record_joint_min",
            "preserves_interlead_relative_amplitudes_and_offsets": True,
            "preserves_within_record_interwindow_scale": True,
            "heldout_target_statistics_used": True,
            "deployment_scope": "paired_benchmark_only_not_lead_II_only_inference",
            "generated_inverse_policy": "ground_truth_full_record_joint_12lead_scaler_is_oracle_only",
        }
        mismatched = [key for key, value in expected.items() if normalization.get(key) != value]
        if mismatched:
            raise ValueError(
                "checkpoint source-record joint-12-lead min-max normalization has invalid fields: "
                + ", ".join(mismatched)
            )
    elif normalization["method"] == "rddm_window_minmax_neg1_1":
        expected = {
            "stats_scope": "per_window_per_modality",
            "stats_source": "selected_ecg_or_ppg_window",
            "feature_range": [-1.0, 1.0],
            "preprocessing_order": "nan_to_num_float32_then_minmax_then_neurokit_clean",
            "inverse_transform": "unavailable_after_per_window_scaling_and_neurokit_cleaning",
            "generated_inverse_policy": "normalized_domain_only",
        }
        mismatched = [key for key, value in expected.items() if normalization.get(key) != value]
        if mismatched:
            raise ValueError(
                "checkpoint RDDM window min-max normalization has invalid fields: "
                + ", ".join(mismatched)
            )
    elif normalization["method"] == "window_minmax_neg1_1":
        expected = {
            "stats_scope": "per_window_per_modality",
            "stats_source": "selected_ecg_or_condition_window",
            "feature_range": [-1.0, 1.0],
            "preprocessing_order": "finite_float32_then_minmax_no_signal_cleaning",
            "inverse_transform": "unavailable_without_saved_per_window_minima_and_ranges",
            "generated_inverse_policy": "normalized_domain_only",
        }
        mismatched = [key for key, value in expected.items() if normalization.get(key) != value]
        if mismatched:
            raise ValueError(
                "checkpoint window min-max normalization has invalid fields: "
                + ", ".join(mismatched)
            )
    elif normalization["method"] == "source_record_minmax_neg1_1":
        expected = {
            "stats_scope": "per_source_continuous_record_per_modality",
            "stats_source": "dataset_sidecars_computed_before_window_split",
            "feature_range": [-1.0, 1.0],
            "inverse_transform": "x=(x_scaled+1)*source_record_range/2+source_record_min",
            "generated_inverse_policy": "ground_truth_target_scaler_is_oracle_only",
            "cross_modality_scaler_shared": False,
        }
        mismatched = [key for key, value in expected.items() if normalization.get(key) != value]
        if mismatched:
            raise ValueError(
                "checkpoint source-record min-max normalization has invalid fields: "
                + ", ".join(mismatched)
            )
    else:
        raise ValueError("checkpoint normalization method is unsupported")
    expected_normalization_id = {
        "training_global_zscore": "training_global_zscore_v1",
        "record_zscore": "record_zscore_v1",
        "record_minmax_neg1_1": "record_minmax_neg1_1_v1",
        "record_joint12_minmax_neg1_1": "record_joint12_minmax_neg1_1_v1",
        "source_record_joint12_minmax_neg1_1": "source_record_joint12_minmax_neg1_1_v1",
        "rddm_window_minmax_neg1_1": "rddm_window_minmax_neg1_1_v1",
        "window_minmax_neg1_1": "window_minmax_neg1_1_v1",
        "source_record_minmax_neg1_1": "source_record_minmax_neg1_1_v1",
    }[normalization["method"]]
    if normalization["normalization_id"] != expected_normalization_id:
        raise ValueError("checkpoint normalization method and normalization_id disagree")
    for field in ("normalization_id", "condition_unit", "target_unit"):
        if normalization[field] != config[field]:
            raise ValueError(f"checkpoint normalization and config disagree on {field}")
    output_spec = payload["output_spec"]
    if int(output_spec.get("channels", 0)) <= 0 or int(output_spec.get("length", 0)) <= 0:
        raise ValueError("checkpoint output_spec requires positive channels and length")
    if float(output_spec.get("sampling_rate_hz", 0)) <= 0 or not output_spec.get("target_lead"):
        raise ValueError("checkpoint output_spec requires sampling_rate_hz and target_lead")
    if output_spec["target_lead"] != config["target_lead"]:
        raise ValueError("checkpoint output_spec and config disagree on target_lead")
    if config["task"] == "ecg2ecg":
        target_indices = config.get("target_lead_indices")
        if target_indices is None:
            target_indices = [config["target_lead_index"]]
        target_leads = output_spec.get("target_leads", [output_spec["target_lead"]])
        if len(target_indices) != int(output_spec["channels"]) or len(target_leads) != int(
            output_spec["channels"]
        ):
            raise ValueError("checkpoint target lead metadata disagrees with output channels")
        if output_spec.get("target_lead_indices", target_indices) != target_indices:
            raise ValueError("checkpoint output_spec and config disagree on target lead indices")
    expected_length = float(config["window_size"]) * float(output_spec["sampling_rate_hz"])
    if float(output_spec["length"]) != expected_length:
        raise ValueError("checkpoint output length disagrees with window size and sampling rate")
    provenance = payload["provenance"]
    for field in ("git_commit", "git_dirty", "command"):
        if field not in provenance:
            raise ValueError(f"checkpoint provenance is missing {field}")
    if schema_version == 2:
        if "ot_normalize_cost" not in config:
            raise ValueError("schema-2 checkpoint config is missing ot_normalize_cost")
        for field in ("global_step", "best_metrics", "rng_states"):
            if field not in payload:
                raise ValueError(f"schema-2 checkpoint is missing {field}")
        if int(payload["global_step"]) < 0:
            raise ValueError("checkpoint global_step must be nonnegative")
        if not isinstance(payload["best_metrics"], Mapping):
            raise ValueError("checkpoint best_metrics must be a mapping")
        rng_states = payload["rng_states"]
        required_rng = {"python", "numpy", "torch_cpu", "torch_cuda"}
        if not isinstance(rng_states, Mapping) or required_rng - set(rng_states):
            raise ValueError("schema-2 checkpoint has incomplete RNG states")


def capture_rng_states() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_states(states: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(states)
    if missing:
        raise ValueError(f"RNG states are missing: {sorted(missing)}")
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.random.set_rng_state(states["torch_cpu"])
    if torch.cuda.is_available() and states["torch_cuda"]:
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def save_checkpoint(payload: Mapping[str, Any], path: Path) -> None:
    validate_checkpoint(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)


def load_checkpoint(path: Path, map_location: torch.device | str) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    validate_checkpoint(payload)
    return dict(payload)
