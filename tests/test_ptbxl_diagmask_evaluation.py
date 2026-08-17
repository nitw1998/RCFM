import numpy as np
import pytest

from scripts.assemble_ptbxl_diagmask_sixway import (
    _extra_comparisons,
    _named_directories,
    _paired_test,
)
from scripts.analyze_ptbxl_diagmask_clinical_significance import (
    _paired_complete_case,
    _resolve_comparisons,
)
from scripts.evaluate_ptbxl_flow_checkpoint import _validate_checkpoint


def _contract(region_weight=0.01, use_ot=False, mask=True, kind="canonical_multistep_rcfm"):
    method = "xresnet1d101_all_positive_statements_gradcam_l5_sample_center_v1"
    config = {
        "task": "ecg2ecg", "datasets": ["PTBXL"],
        "dataset_version": "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1",
        "split_hash": "7784c98a2c8daccc23fc7cb0d47dc1933eeee77f120c05f8a24ac149cd8474f7",
        "normalization_id": "record_minmax_neg1_1_v1",
        "alignment_id": "ptbxl_official_folds_first4s_same_record_lead_II_to_other11_v1",
        "condition_lead_index": 1, "target_lead_indices": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "window_size": 4, "flow_matcher": "conditional", "sigma": 0.0, "seed": 31,
        "region_weight": region_weight, "use_minibatch_ot": use_ot, "ot_method": "exact",
        "mask_method": method if mask else None,
        "region_mask_path": "mask.npy" if mask else None,
        "region_mask_manifest": "manifest.json" if mask else None,
        "region_mask_provenance": {"method": method, "test_mask_generated": False} if mask else {},
    }
    return {
        "kind": kind, "config": config,
        "normalization": {"normalization_id": "record_minmax_neg1_1_v1"},
        "output_spec": {"channels": 11, "length": 512, "target_leads": ["I", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]},
    }


def test_diagmask_contract_requires_training_only_diagnostic_mask():
    _validate_checkpoint(_contract(), "diag")
    bad = _contract()
    bad["config"]["region_mask_provenance"]["test_mask_generated"] = True
    with pytest.raises(ValueError, match="test_mask_generated"):
        _validate_checkpoint(bad, "diag")


def test_semanticmask_contract_requires_resunet_training_only_mask():
    checkpoint = _contract()
    method = "ptbxl_plus_resunet_p_qrs_t_lead_ii_soft_max_epoch13_v1"
    checkpoint["config"]["mask_method"] = method
    checkpoint["config"]["region_mask_provenance"]["method"] = method
    _validate_checkpoint(checkpoint, "semantic")

    bad = _contract()
    bad["config"]["region_mask_provenance"]["test_mask_generated"] = True
    bad["config"]["mask_method"] = method
    bad["config"]["region_mask_provenance"]["method"] = method
    with pytest.raises(ValueError, match="test_mask_generated"):
        _validate_checkpoint(bad, "semantic")


@pytest.mark.parametrize(
    ("model", "method"),
    [
        (
            "ecgmamba_diag",
            "ecgmamba_fca_mgda_s42_positive_diagnostic_head_gradcam_fullres_v1",
        ),
        (
            "ecgmamba_semantic",
            "ecgmamba_fca_mgda_s42_p_qrs_t_semantic_head_gradcam_fullres_v1",
        ),
    ],
)
def test_ecgmamba_taskhead_contracts_require_exact_training_only_mask(model, method):
    checkpoint = _contract()
    checkpoint["config"]["mask_method"] = method
    checkpoint["config"]["region_mask_provenance"]["method"] = method
    _validate_checkpoint(checkpoint, model)

    bad = _contract()
    bad["config"]["mask_method"] = method
    bad["config"]["region_mask_provenance"]["method"] = method
    bad["config"]["region_mask_provenance"]["test_mask_generated"] = True
    with pytest.raises(ValueError, match="test_mask_generated"):
        _validate_checkpoint(bad, model)


def test_cfm_ot_contract_has_no_mask_and_exact_ot():
    _validate_checkpoint(
        _contract(region_weight=0.0, use_ot=True, mask=False, kind="canonical_multistep_cfm_ot"),
        "cfm_ot",
    )


@pytest.mark.parametrize(
    ("model", "weight"),
    [("diag_l003", 0.03), ("diag_l010", 0.1), ("diag_l030", 0.3)],
)
def test_lambda_contracts_are_explicit_and_training_only(model, weight):
    _validate_checkpoint(_contract(region_weight=weight), model)
    with pytest.raises(ValueError, match="region_weight"):
        _validate_checkpoint(_contract(region_weight=weight + 0.01), model)


def test_optional_models_and_comparisons_are_validated():
    directories = _named_directories(["diag_l003=/tmp/l003", "diag_l010=/tmp/l010"])
    assert list(directories) == ["diag_l003", "diag_l010"]
    models = ("cfm", "cfm_ot", "pan", "pan_ot", "diag", "diag_ot", *directories)
    assert _extra_comparisons(["diag_l003:diag", "diag_l010:diag"], models) == (
        ("diag_l003", "diag"),
        ("diag_l010", "diag"),
    )
    with pytest.raises(ValueError, match="invalid extra comparison"):
        _extra_comparisons(["missing:diag"], models)


def test_clinical_comparisons_follow_source_protocol():
    configured = [["diag", "cfm"], ["diag_l010", "diag"]]
    assert _resolve_comparisons({"analysis_comparisons": configured}) == (
        ("diag", "cfm"),
        ("diag_l010", "diag"),
    )


def test_patient_test_averages_records_before_paired_inference():
    patient_ids = np.array([1, 1, 2, 3])
    reference = np.array([1.0, 3.0, 2.0, 4.0])
    comparison = np.array([2.0, 4.0, 1.0, 5.0])
    result = _paired_test(comparison, reference, patient_ids, seed=7, replicates=200)
    assert result["patients"] == 3
    assert result["mean_difference"] == pytest.approx(1 / 3)
    assert result["difference_definition"].startswith("comparison_minus_reference")


def test_clinical_comparison_uses_same_record_lead_complete_cases():
    errors = {
        "diag": {
            ("1", "p1", "V3"): {"qrs_ms": 2.0},
            ("2", "p1", "V3"): {},
            ("3", "p2", "V3"): {"qrs_ms": 4.0},
        },
        "pan": {
            ("1", "p1", "V3"): {"qrs_ms": 3.0},
            ("2", "p1", "V3"): {"qrs_ms": 1.0},
            ("3", "p2", "V3"): {"qrs_ms": 5.0},
        },
    }
    comparison, reference, patients, cells = _paired_complete_case(
        errors, "diag", "pan", "qrs_ms"
    )
    np.testing.assert_allclose(comparison, [2.0, 4.0])
    np.testing.assert_allclose(reference, [3.0, 5.0])
    np.testing.assert_array_equal(patients, ["p1", "p2"])
    assert cells == 2
