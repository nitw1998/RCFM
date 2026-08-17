from collections import OrderedDict

import pytest
import torch

from scripts.export_jetson_onnx import (
    ExportMultiheadAttention,
    _attention_heads,
    _checkpoint_contract,
)


def test_rcfm_checkpoint_extracts_flow_prefix():
    tensor = torch.ones(1)
    payload = {
        "model_state": OrderedDict((('flow_model.weight', tensor),)),
        "condition_state": OrderedDict((('weight', tensor),)),
        "config": {"attention_heads": 8},
    }
    states, config = _checkpoint_contract(payload, "rcfm")
    assert list(states["flow"]) == ["weight"]
    assert list(states["condition"]) == ["weight"]
    assert config["attention_heads"] == 8


def test_rddm_checkpoint_extracts_two_denoisers():
    tensor = torch.ones(1)
    payload = {
        "rddm_state": OrderedDict((
            ('sqrtab', tensor), ('region_model.weight', tensor), ('eps_model.weight', tensor),
        )),
        "condition_1_state": {"weight": tensor},
        "condition_2_state": {"weight": tensor},
    }
    states, _ = _checkpoint_contract(payload, "rddm")
    assert list(states["region"]) == ["weight"]
    assert list(states["epsilon"]) == ["weight"]


def test_export_contract_rejects_incompatible_attention_heads():
    with pytest.raises(ValueError, match="divide"):
        _attention_heads({}, 7)


def test_export_attention_matches_pytorch_attention():
    torch.manual_seed(7)
    original = torch.nn.MultiheadAttention(32, 4, batch_first=True).eval()
    exported = ExportMultiheadAttention(original).eval()
    query = torch.randn(2, 9, 32)
    context = torch.randn(2, 5, 32)
    with torch.inference_mode():
        expected, _ = original(query, context, context, need_weights=False)
        actual, _ = exported(query, context, context)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
