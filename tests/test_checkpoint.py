from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.rcfm.checkpoint import (
    capture_rng_states,
    load_checkpoint,
    restore_rng_states,
    save_checkpoint,
    validate_checkpoint,
)


def _payload():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return {
        "schema_version": 1,
        "kind": "canonical_multistep_rcfm",
        "epoch": 3,
        "model_state": model.state_dict(),
        "condition_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": {
            "task": "ecg2ecg",
            "datasets": ["synthetic"],
            "dataset_version": "synthetic-v1",
            "seed": 31,
            "split_hash": "abc",
            "flow_matcher": "conditional",
            "sigma": 0.0,
            "region_weight": 0.01,
            "use_minibatch_ot": True,
            "ot_method": "sinkhorn",
            "ot_reg": 0.05,
            "ot_normalize_cost": False,
            "normalization_id": "training_global_zscore_v1",
            "condition_unit": "mV",
            "target_unit": "mV",
            "alignment_id": "native-synchronized-v1",
            "condition_lead": "source",
            "target_lead": "V5",
            "condition_lead_index": 2,
            "target_lead_index": 10,
            "window_size": 4,
            "attention_heads": 8,
        },
        "normalization": {
            "method": "training_global_zscore",
            "target_mean": 0.0,
            "target_scale": 1.0,
            "condition_mean": 0.0,
            "condition_scale": 1.0,
            "normalization_id": "training_global_zscore_v1",
            "condition_unit": "mV",
            "target_unit": "mV",
        },
        "output_spec": {
            "channels": 1,
            "length": 512,
            "sampling_rate_hz": 128,
            "target_lead": "V5",
        },
        "provenance": {
            "git_commit": "deadbeef",
            "git_dirty": False,
            "command": "synthetic",
        },
    }


def test_checkpoint_roundtrip_and_exact_config(tmp_path: Path):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(_payload(), path)
    restored = load_checkpoint(path, map_location="cpu")

    assert restored["config"]["flow_matcher"] == "conditional"
    assert restored["config"]["sigma"] == 0.0
    assert restored["output_spec"] == {
        "channels": 1,
        "length": 512,
        "sampling_rate_hz": 128,
        "target_lead": "V5",
    }


def test_checkpoint_rejects_noncanonical_or_incomplete_config():
    payload = _payload()
    payload["config"]["flow_matcher"] = "vp"
    with pytest.raises(ValueError, match="canonical linear"):
        validate_checkpoint(payload)

    payload = _payload()
    del payload["config"]["split_hash"]
    with pytest.raises(ValueError, match="split_hash"):
        validate_checkpoint(payload)


def test_checkpoint_accepts_cfm_compare_kind_and_rejects_mislabelling():
    payload = _payload()
    payload["kind"] = "canonical_multistep_cfm"
    payload["config"].update(
        {
            "model_family": "CFM",
            "region_weight": 0.0,
            "use_minibatch_ot": False,
        }
    )
    validate_checkpoint(payload)

    payload["config"]["region_weight"] = 0.01
    with pytest.raises(ValueError, match="region_weight=0"):
        validate_checkpoint(payload)

    payload = _payload()
    payload["kind"] = "canonical_multistep_cfm"
    with pytest.raises(ValueError, match="model_family disagree"):
        validate_checkpoint(payload)


@pytest.mark.parametrize("path_type", ("vp", "target"))
def test_checkpoint_accepts_non_ot_path_ablation_and_rejects_ot(path_type):
    payload = _payload()
    payload["kind"] = "path_ablation_rcfm"
    payload["config"].update(
        {
            "experiment_role": "path_ablation",
            "flow_matcher": path_type,
            "sigma": 0.1,
            "use_minibatch_ot": False,
            "ot_sampling_strategy": "multinomial",
        }
    )
    validate_checkpoint(payload)

    payload["config"]["use_minibatch_ot"] = True
    with pytest.raises(ValueError, match="must not use OT"):
        validate_checkpoint(payload)


def test_checkpoint_accepts_sb_path_ablation_only_with_external_exact_multinomial_ot():
    payload = _payload()
    payload["kind"] = "path_ablation_rcfm"
    payload["config"].update(
        {
            "experiment_role": "path_ablation",
            "flow_matcher": "sb",
            "sigma": 0.1,
            "use_minibatch_ot": True,
            "ot_method": "exact",
            "ot_sampling_strategy": "multinomial",
        }
    )
    validate_checkpoint(payload)

    for field, bad_value in (
        ("use_minibatch_ot", False),
        ("ot_method", "sinkhorn"),
        ("ot_sampling_strategy", "assignment"),
    ):
        invalid = _payload()
        invalid["kind"] = "path_ablation_rcfm"
        invalid["config"].update(payload["config"])
        invalid["config"][field] = bad_value
        with pytest.raises(ValueError, match="external exact multinomial OT"):
            validate_checkpoint(invalid)


def test_path_ablation_checkpoint_cannot_be_labelled_canonical():
    payload = _payload()
    payload["kind"] = "path_ablation_rcfm"
    with pytest.raises(ValueError, match="experiment_role=path_ablation"):
        validate_checkpoint(payload)


def test_checkpoint_rejects_inconsistent_restoration_metadata():
    payload = _payload()
    payload["normalization"]["target_unit"] = "uV"
    with pytest.raises(ValueError, match="target_unit"):
        validate_checkpoint(payload)

    payload = _payload()
    payload["output_spec"]["target_lead"] = "III"
    with pytest.raises(ValueError, match="target_lead"):
        validate_checkpoint(payload)


def test_checkpoint_accepts_joint_eleven_lead_output_and_rejects_bad_metadata():
    payload = _payload()
    indices = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    leads = [
        "I", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
    ]
    payload["config"].update(
        {
            "condition_lead": "II",
            "condition_lead_index": 1,
            "target_lead": ",".join(leads),
            "target_lead_index": None,
            "target_lead_indices": indices,
        }
    )
    payload["output_spec"].update(
        {
            "channels": 11,
            "target_lead": ",".join(leads),
            "target_leads": leads,
            "target_lead_indices": indices,
        }
    )

    validate_checkpoint(payload)

    payload["output_spec"]["target_leads"] = leads[:-1]
    with pytest.raises(ValueError, match="metadata disagrees with output channels"):
        validate_checkpoint(payload)


def test_checkpoint_accepts_record_zscore_sidecar_protocol():
    payload = _payload()
    payload["config"]["normalization_id"] = "record_zscore_v1"
    payload["normalization"] = {
        "method": "record_zscore",
        "stats_scope": "per_record_per_lead",
        "stats_source": "dataset_sidecar",
        "inverse_transform": "x=x_z*record_scale+record_mean",
        "generated_inverse_policy": "ground_truth_target_scaler_is_oracle_only",
        "normalization_id": "record_zscore_v1",
        "condition_unit": "mV",
        "target_unit": "mV",
    }

    validate_checkpoint(payload)

    payload["normalization"]["stats_source"] = "ground_truth_target_at_inference"
    with pytest.raises(ValueError, match="record normalization"):
        validate_checkpoint(payload)


def test_checkpoint_accepts_record_minmax_neg1_1_protocol():
    payload = _payload()
    payload["config"]["normalization_id"] = "record_minmax_neg1_1_v1"
    payload["normalization"] = {
        "method": "record_minmax_neg1_1",
        "stats_scope": "per_record_per_lead",
        "stats_source": "selected_waveform_record",
        "feature_range": [-1.0, 1.0],
        "inverse_transform": "x=(x_scaled+1)*record_range/2+record_min",
        "generated_inverse_policy": "ground_truth_target_scaler_is_oracle_only",
        "normalization_id": "record_minmax_neg1_1_v1",
        "condition_unit": "mV",
        "target_unit": "mV",
    }

    validate_checkpoint(payload)

    payload["normalization"]["feature_range"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="record min-max normalization"):
        validate_checkpoint(payload)

    payload = _payload()
    payload["normalization"]["normalization_id"] = "record_zscore_v1"
    payload["config"]["normalization_id"] = "record_zscore_v1"
    with pytest.raises(ValueError, match="method and normalization_id"):
        validate_checkpoint(payload)


def test_checkpoint_accepts_rddm_window_minmax_protocol():
    payload = _payload()
    payload["config"]["normalization_id"] = "rddm_window_minmax_neg1_1_v1"
    payload["normalization"] = {
        "method": "rddm_window_minmax_neg1_1",
        "stats_scope": "per_window_per_modality",
        "stats_source": "selected_ecg_or_ppg_window",
        "feature_range": [-1.0, 1.0],
        "preprocessing_order": "nan_to_num_float32_then_minmax_then_neurokit_clean",
        "inverse_transform": "unavailable_after_per_window_scaling_and_neurokit_cleaning",
        "generated_inverse_policy": "normalized_domain_only",
        "normalization_id": "rddm_window_minmax_neg1_1_v1",
        "condition_unit": "mV",
        "target_unit": "mV",
    }

    validate_checkpoint(payload)

    payload["normalization"]["preprocessing_order"] = "clean_then_minmax"
    with pytest.raises(ValueError, match="RDDM window min-max normalization"):
        validate_checkpoint(payload)


def test_checkpoint_accepts_generic_window_minmax_protocol():
    payload = _payload()
    payload["config"]["normalization_id"] = "window_minmax_neg1_1_v1"
    payload["normalization"] = {
        "method": "window_minmax_neg1_1",
        "stats_scope": "per_window_per_modality",
        "stats_source": "selected_ecg_or_condition_window",
        "feature_range": [-1.0, 1.0],
        "preprocessing_order": "finite_float32_then_minmax_no_signal_cleaning",
        "inverse_transform": "unavailable_without_saved_per_window_minima_and_ranges",
        "generated_inverse_policy": "normalized_domain_only",
        "normalization_id": "window_minmax_neg1_1_v1",
        "condition_unit": payload["config"]["condition_unit"],
        "target_unit": payload["config"]["target_unit"],
    }

    validate_checkpoint(payload)

    payload["normalization"]["preprocessing_order"] = "clean_then_minmax"
    with pytest.raises(ValueError, match="checkpoint window min-max normalization"):
        validate_checkpoint(payload)


def test_schema_two_checkpoint_requires_selection_and_rng_metadata(tmp_path: Path):
    payload = _payload()
    payload.update(
        {
            "schema_version": 2,
            "global_step": 17,
            "best_metrics": {"val/rmse": 0.5},
            "rng_states": capture_rng_states(),
        }
    )
    path = tmp_path / "schema_two.pt"
    save_checkpoint(payload, path)
    restored = load_checkpoint(path, map_location="cpu")
    assert restored["global_step"] == 17
    assert restored["best_metrics"] == {"val/rmse": 0.5}

    del payload["rng_states"]
    with pytest.raises(ValueError, match="rng_states"):
        validate_checkpoint(payload)


def test_rng_state_capture_and_restore_roundtrip():
    torch.manual_seed(123)
    states = capture_rng_states()
    expected = torch.rand(3)
    torch.rand(7)
    restore_rng_states(states)
    torch.testing.assert_close(torch.rand(3), expected)
