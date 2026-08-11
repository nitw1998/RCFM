from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_ptbxl_path_ablation import _read_per_record
from scripts.evaluate_ptbxl_path_ablation import MODEL_ORDER, _validate_contracts


def _contract(name: str) -> dict[str, object]:
    config = {
        "task": "ecg2ecg",
        "datasets": ["PTBXL"],
        "dataset_version": "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1",
        "split_hash": "7784c98a2c8daccc23fc7cb0d47dc1933eeee77f120c05f8a24ac149cd8474f7",
        "normalization_id": "record_minmax_neg1_1_v1",
        "alignment_id": "ptbxl_official_folds_first4s_same_record_lead_II_to_other11_v1",
        "condition_lead_index": 1,
        "target_lead_indices": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "window_size": 4,
        "attention_heads": 8,
        "region_weight": 0.01,
        "seed": 31,
        "ot_method": "exact",
    }
    factors = {
        "conditional_ot": ("canonical_multistep_rcfm", None, "conditional", 0.0, True, "assignment"),
        "vp": ("path_ablation_rcfm", "path_ablation", "vp", 0.1, False, "multinomial"),
        "target": ("path_ablation_rcfm", "path_ablation", "target", 0.1, False, "multinomial"),
        "sb": ("path_ablation_rcfm", "path_ablation", "sb", 0.1, True, "multinomial"),
    }
    kind, role, matcher, sigma, use_ot, sampling = factors[name]
    config.update(
        experiment_role=role,
        flow_matcher=matcher,
        sigma=sigma,
        use_minibatch_ot=use_ot,
        ot_sampling_strategy=sampling,
    )
    return {
        "kind": kind,
        "epoch": 500,
        "global_step": 68500,
        "config": config,
        "normalization": {"method": "record_minmax_neg1_1"},
        "output_spec": {
            "channels": 11,
            "length": 512,
            "target_leads": ["I", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        },
    }


def test_path_ablation_contract_accepts_frozen_factor_matrix():
    contracts = {name: _contract(name) for name in MODEL_ORDER}
    _validate_contracts(contracts)


def test_path_ablation_contract_rejects_wrong_endpoint():
    contracts = {name: _contract(name) for name in MODEL_ORDER}
    contracts = deepcopy(contracts)
    contracts["sb"]["epoch"] = 475
    with pytest.raises(ValueError, match="epoch-500"):
        _validate_contracts(contracts)


def test_path_ablation_contract_rejects_calling_target_an_ot_path():
    contracts = {name: _contract(name) for name in MODEL_ORDER}
    contracts = deepcopy(contracts)
    contracts["target"]["config"]["use_minibatch_ot"] = True
    with pytest.raises(ValueError, match="target path/coupling factors"):
        _validate_contracts(contracts)


def test_path_ablation_per_record_reader_preserves_model_metric_mapping(tmp_path: Path):
    fields = ["record_id", "patient_id"] + [
        f"{model}_{metric}" for model in MODEL_ORDER for metric in ("rmse", "mae")
    ]
    values = ["1", "10"] + [str(index / 10) for index in range(8)]
    path = tmp_path / "metrics.csv"
    path.write_text(",".join(fields) + "\n" + ",".join(values) + "\n", encoding="utf-8")
    result = _read_per_record(path)
    np.testing.assert_allclose(result["conditional_ot"]["rmse"], [0.0])
    np.testing.assert_allclose(result["sb"]["mae"], [0.7])
