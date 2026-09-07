from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_ptbxl_factorial_multiseed import (
    _exact_seed_sign_flip_p_value,
    _hierarchical_paired_test,
    _parse_prediction_specs,
)
from scripts.evaluate_ptbxl_flow_checkpoint import _validate_checkpoint
from train_rddm_compare import parse_args_with_config as parse_rddm


def _cfm_contract(seed: int):
    return {
        "kind": "canonical_multistep_cfm",
        "config": {
            "task": "ecg2ecg",
            "datasets": ["PTBXL"],
            "dataset_version": "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1",
            "split_hash": "7784c98a2c8daccc23fc7cb0d47dc1933eeee77f120c05f8a24ac149cd8474f7",
            "normalization_id": "record_minmax_neg1_1_v1",
            "alignment_id": "ptbxl_official_folds_first4s_same_record_lead_II_to_other11_v1",
            "condition_lead_index": 1,
            "target_lead_indices": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "window_size": 4,
            "flow_matcher": "conditional",
            "sigma": 0.0,
            "seed": seed,
            "region_weight": 0.0,
            "use_minibatch_ot": False,
            "ot_method": "exact",
        },
        "normalization": {"normalization_id": "record_minmax_neg1_1_v1"},
        "output_spec": {
            "channels": 11,
            "length": 512,
            "target_leads": [
                "I", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
            ],
        },
    }


def test_prediction_specs_require_complete_four_by_three_factorial():
    values = [
        f"{model}:{seed}=/tmp/{model}_s{seed}"
        for model in ("cfm", "cfm_ot", "diag", "diag_ot")
        for seed in (31, 32, 33)
    ]
    parsed = _parse_prediction_specs(values)
    assert parsed[("diag_ot", 33)] == Path("/tmp/diag_ot_s33")
    with pytest.raises(ValueError, match="missing model/seed"):
        _parse_prediction_specs(values[:-1])


def test_checkpoint_contract_accepts_explicit_training_seed_and_plain_cfm():
    cfm = _cfm_contract(32)
    _validate_checkpoint(cfm, "cfm", expected_training_seed=32)
    with pytest.raises(ValueError, match="seed"):
        _validate_checkpoint(cfm, "cfm", expected_training_seed=33)


@pytest.mark.parametrize("seed", (32, 33))
def test_ptbxl_rddm_additional_seed_contract(seed: int):
    config = Path(__file__).resolve().parents[1] / "configs/ptbxl/rddm_adapted_record_minmax_neg1_1_seed31.yaml"
    args = parse_rddm(["--config", str(config), "--seed", str(seed)])
    assert args.datasets == "PTBXL"
    assert args.seed == seed
    assert args.nT == 10
    assert args.expected_train_windows == 17440
    assert args.expected_test_windows == 2193
    assert args.reproduction_label == "RDDM-ECG (adapted)"


def test_three_seed_exact_sign_flip_has_minimum_two_sided_resolution_quarter():
    assert _exact_seed_sign_flip_p_value(np.array([-1.0, -2.0, -3.0])) == pytest.approx(0.25)


def test_hierarchical_test_averages_records_before_seed_and_patient_inference():
    comparison = [
        {"p1": 2.0, "p2": 4.0, "p3": 8.0},
        {"p1": 3.0, "p2": 5.0, "p3": 9.0},
        {"p1": 4.0, "p2": 6.0, "p3": 10.0},
    ]
    reference = [
        {"p1": 1.0, "p2": 3.0, "p3": 7.0},
        {"p1": 2.0, "p2": 4.0, "p3": 8.0},
        {"p1": 3.0, "p2": 5.0, "p3": 9.0},
    ]
    result = _hierarchical_paired_test(
        comparison,
        reference,
        bootstrap_seed=7,
        bootstrap_replicates=200,
    )
    assert result["training_seeds"] == 3
    assert result["patients"] == 3
    assert result["mean_difference"] == pytest.approx(1.0)
    assert result["seed_level_differences"] == pytest.approx([1.0, 1.0, 1.0])
    assert result["exact_seed_sign_flip_p_value"] == pytest.approx(0.25)
