import numpy as np
import pytest

from scripts.evaluate_cpsc_minmax_threeway import (
    _clinical_threeway_comparison,
    _include_all_p_wave_applicability,
    _validate_threeway_contracts,
)


def _contract(kind, family, region_weight, use_ot, sampling="multinomial"):
    return {
        "kind": kind,
        "config": {
            "task": "ecg2ecg",
            "datasets": ["CPSC2018"],
            "dataset_version": "synthetic-v1",
            "split_hash": "split-123",
            "normalization_id": "record_minmax_neg1_1_v1",
            "condition_unit": "unknown_source_unit",
            "target_unit": "unknown_source_unit",
            "alignment_id": "aligned",
            "condition_lead": "lead-2",
            "target_lead": "lead-10",
            "condition_lead_index": 2,
            "target_lead_index": 10,
            "window_size": 4,
            "attention_heads": 8,
            "flow_matcher": "conditional",
            "sigma": 0.0,
            "seed": 31,
            "region_weight": region_weight,
            "use_minibatch_ot": use_ot,
            "ot_method": "exact",
            "ot_sampling_strategy": sampling,
            "model_family": family,
        },
        "normalization": {
            "method": "record_minmax_neg1_1",
            "normalization_id": "record_minmax_neg1_1_v1",
        },
        "output_spec": {"channels": 1, "length": 512},
    }


def _contracts():
    return {
        "cfm": _contract("canonical_multistep_cfm", "CFM", 0.0, False),
        "rcfm": _contract("canonical_multistep_rcfm", "RCFM", 0.01, False),
        "rcfm_ot": _contract(
            "canonical_multistep_rcfm", "RCFM", 0.01, True, "assignment"
        ),
    }


def test_threeway_contract_accepts_only_expected_region_and_ot_factors():
    contracts = _contracts()
    _validate_threeway_contracts(contracts)

    contracts["rcfm_ot"]["config"]["split_hash"] = "wrong"
    with pytest.raises(ValueError, match="split_hash"):
        _validate_threeway_contracts(contracts)


def test_historical_threeway_clinical_evaluator_rejects_multilead_checkpoints():
    contracts = _contracts()
    for contract in contracts.values():
        contract["output_spec"]["channels"] = 11

    with pytest.raises(ValueError, match="only single-target-lead"):
        _validate_threeway_contracts(contracts)


def test_include_all_p_wave_policy_does_not_consult_af_labels():
    applicable, policy = _include_all_p_wave_applicability(4)

    np.testing.assert_array_equal(applicable, np.ones(4, dtype=bool))
    assert policy["af_labels_consulted"] is False
    assert policy["applicable_records"] == 4


def _clinical(real, generated):
    return {
        "unit_summaries": {
            "real": {
                "a": {"rr_ms": real[0]},
                "b": {"rr_ms": real[1]},
            },
            "generated": {
                "a": {"rr_ms": generated[0]},
                "b": {"rr_ms": generated[1]},
            },
        }
    }


def test_threeway_clinical_comparison_uses_exact_common_records():
    results = {
        "cfm": _clinical([1000.0, 900.0], [1020.0, 930.0]),
        "rcfm": _clinical([1000.0, 900.0], [1010.0, 920.0]),
        "rcfm_ot": _clinical([1000.0, 900.0], [1005.0, 910.0]),
    }

    comparison = _clinical_threeway_comparison(results)["parameters"]["rr_ms"]

    assert comparison["n"] == 2
    assert comparison["models"]["cfm"]["mae"] == pytest.approx(25.0)
    assert comparison["models"]["rcfm"]["mae"] == pytest.approx(15.0)
    assert comparison["models"]["rcfm_ot"]["mae"] == pytest.approx(7.5)
    assert comparison["models"]["rcfm_ot"]["bland_altman"]["bias"] == pytest.approx(7.5)
