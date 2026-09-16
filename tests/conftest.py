"""Public-release test routing for intentionally omitted experiment configs."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# Only tests whose declared input is an omitted config are skipped. Other tests
# in the same modules continue to run in the public suite.
INTERNAL_CONFIG_TESTS = {
    "test_catransformer.py::test_cat_configs_freeze_reproduction_and_adaptation_labels",
    "test_catransformer.py::test_cat_ecg_resume_rejects_superseded_pointwise_adapter_checkpoint",
    "test_catransformer.py::test_resume_contract_rejects_architecture_change",
    "test_cfm_compare.py::test_cfm_compare_config_matches_rcfm_except_declared_factor_and_labels",
    "test_cfm_compare.py::test_cfm_compare_rejects_region_weight_and_minibatch_ot",
    "test_cfm_compare.py::test_cfm_compare_runs_shared_metric_and_checkpoint_interfaces",
    "test_cfm_compare.py::test_mimic_flow_configs_are_matched_except_declared_region_and_ot_factors",
    "test_cfm_compare.py::test_minmax_configs_match_and_freeze_long_schedule",
    "test_cfm_compare.py::test_minmax_rcfm_exact_ot_config_changes_only_coupling_and_label",
    "test_cfm_compare.py::test_ptbxl_configs_freeze_official_waveform_only_protocol",
    "test_cfm_compare.py::test_ptbxl_diagmask_pair_changes_only_exact_coupling_fields",
    "test_cfm_ot.py::test_cfm_ot_configs_freeze_the_factorial_contract",
    "test_cfm_ot.py::test_cfm_ot_rejects_region_loss_and_disabled_ot",
    "test_cpsc2018_multiseed_configs.py::test_cpsc_core_factorial_additional_seed_contract",
    "test_cpsc2018_multiseed_configs.py::test_cpsc_rddm_additional_seed_contract",
    "test_cpsc2018_random_window_split.py::test_cpsc_random_window_cfm_config_and_launcher_are_frozen",
    "test_delineation_unet.py::test_delineation_config_is_full_online_training",
    "test_direct_cnn.py::test_frozen_direct_cnn_configs",
    "test_direct_cnn_random_window.py::test_random_window_direct_cnn_config_contract",
    "test_direct_cnn_random_window.py::test_random_window_direct_cnn_rejects_wrong_split_hash",
    "test_experiment.py::test_cpsc2018_full_config_is_uncapped_and_wandb_online",
    "test_experiment.py::test_five_ot_templates_are_controlled_and_cli_overridable",
    "test_experiment.py::test_mimic_afib_rddm_no_ot_config_is_frozen_for_final_test_only",
    "test_facm_five_dataset_entry.py::test_cfm50_five_dataset_configs_are_complete_and_validate",
    "test_facm_five_dataset_entry.py::test_five_dataset_configs_are_complete_and_validate",
    "test_mimic_afib_random_window_split.py::test_mimic_random_window_cfm_config_and_launcher_are_frozen",
    "test_mimic_multiseed_configs.py::test_mimic_flow_multiseed_variants_freeze_data_and_factors",
    "test_mimic_multiseed_configs.py::test_mimic_rddm_and_direct_cnn_additional_seed_contracts",
    "test_mimic_path_ablation.py::test_mimic_path_ablation_configs_lock_matched_protocol",
    "test_mimic_path_ablation.py::test_training_rejects_invalid_path_coupling_protocol",
    "test_mmecg_configs.py::test_flow_configs_freeze_comparable_mmecg_protocol",
    "test_mmecg_configs.py::test_rddm_config_uses_same_mmecg_split_and_schedule",
    "test_mmecg_random_window_split.py::test_mmecg_random_window_cfm_config_and_launcher_are_frozen",
    "test_ptbxl_ecgmamba_taskhead_masks.py::test_taskhead_mask_methods_are_distinct_and_configs_match",
    "test_ptbxl_factorial_multiseed.py::test_ptbxl_rddm_additional_seed_contract",
    "test_ptbxl_joint12_minmax.py::test_joint12_lead2_training_config_and_launcher_contract",
    "test_ptbxl_legacy_cfm_singlelead.py::test_config_freezes_unchanged_legacy_architecture_and_singlelead_task",
    "test_ptbxl_legacy_cfm_singlelead.py::test_config_is_plain_json_for_reproducible_launcher_use",
    "test_ptbxl_path_ablation.py::test_ptbxl_invalid_path_protocol_fails_before_data_loading",
    "test_ptbxl_path_ablation.py::test_ptbxl_path_configs_change_only_declared_path_coupling_fields",
    "test_ptbxl_path_ablation.py::test_ptbxl_path_configs_lock_waveform_protocol",
    "test_ptbxl_random_window_split.py::test_random_window_cfm_config_is_frozen_to_200_epochs_and_non_grouped_artifact",
    "test_ptbxl_semantic_masks.py::test_semanticmask_training_config_is_a_single_factor_no_ot_ablation",
    "test_random_window_comparator_training.py::test_flow_comparator_configs_match_random_window_cfm",
    "test_random_window_comparator_training.py::test_rddm_random_window_configs_are_strictly_accepted",
    "test_rddm_compare.py::test_cpsc2018_rddm_config_locks_multilead_adaptation_protocol",
    "test_rddm_compare.py::test_cpsc2018_rddm_rejects_a_different_lead_contract",
    "test_rddm_compare.py::test_ptbxl_rddm_config_locks_multilead_adaptation_protocol",
    "test_rddm_compare.py::test_rddm_compare_config_locks_upstream_and_matched_data_protocol",
    "test_rddm_compare.py::test_rddm_compare_rejects_nonpaper_diffusion_steps",
    "test_rddm_compare.py::test_rddm_debug_caps_must_be_positive",
    "test_rddm_joint_clean.py::test_joint_config_freezes_upstream_derived_training_contract",
    "test_reviewer_five_seed_ablation_launcher.py::test_full_matrix_has_five_locked_seeds_and_epochs",
    "test_wesad_configs.py::test_flow_configs_freeze_comparable_wesad_protocol",
    "test_wesad_configs.py::test_rddm_config_uses_same_wesad_split_and_schedule",
    "test_wesad_mmecg_multiseed.py::test_all_wesad_mmecg_additional_seed_configs",
    "test_wesad_random_window_record_minmax.py::test_record_minmax_cfm_config_and_launcher_are_frozen",
    "test_wesad_random_window_split.py::test_wesad_random_window_cfm_config_and_launcher_are_frozen",
    "test_wesad_train_lag_configs.py::test_aligned_flow_configs_share_one_data_contract",
    "test_wesad_train_lag_configs.py::test_aligned_rddm_config_keeps_subject_split_and_official_sampler_contract",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if os.environ.get("RCFM_INCLUDE_INTERNAL_CONFIG_TESTS") == "1":
        return
    reason = (
        "internal experiment matrix is outside the public RCFM-OT config scope; "
        "set RCFM_INCLUDE_INTERNAL_CONFIG_TESTS=1 after restoring private configs"
    )
    marker = pytest.mark.skip(reason=reason)
    for item in items:
        function_name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
        key = f"{Path(str(item.fspath)).name}::{function_name}"
        if key in INTERNAL_CONFIG_TESTS:
            item.add_marker(marker)
