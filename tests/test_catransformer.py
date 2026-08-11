from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.train_cat import (
    _resolved_config,
    _validate_resume_contract,
    parse_args_with_config,
)
from src.rcfm.baselines.cat_checkpoint import load_cat_checkpoint, save_cat_checkpoint
from src.rcfm.baselines.cat_cycle import CycleViewBuilder, SourceCycleExtractor
from src.rcfm.baselines.catransformer import (
    CATLoss,
    CATransformer,
    CycleAwareTransformerBlock,
)
from src.rcfm.checkpoint import capture_rng_states
from src.rcfm.runtime import exception_summary, gradients_are_finite


def _small_model(dropout: float = 0.0) -> CATransformer:
    return CATransformer(
        input_length=64,
        output_channels=1,
        cat_layers=2,
        top_k=2,
        patch_width=8,
        d_model=16,
        n_heads=4,
        encoder_layers=1,
        ff_dim=32,
        dropout=dropout,
    )


def test_source_fft_selects_known_cycle_without_target_input():
    time = torch.arange(64, dtype=torch.float32)
    source = torch.sin(2 * torch.pi * 4 * time / 64)[None, None]
    selection = SourceCycleExtractor(top_k=2)(source)
    assert selection.frequencies[0, 0].item() == 4
    assert selection.periods[0, 0].item() == 16
    assert not selection.fallback.any()
    assert "target" not in inspect.signature(SourceCycleExtractor.forward).parameters


def test_all_zero_source_has_deterministic_fallback():
    source = torch.zeros(2, 1, 32)
    first = SourceCycleExtractor(top_k=2)(source)
    second = SourceCycleExtractor(top_k=2)(source)
    assert first.fallback.tolist() == [True, True]
    torch.testing.assert_close(first.frequencies, torch.tensor([[1, 2], [1, 2]]))
    torch.testing.assert_close(first.weights, torch.full((2, 2), 0.5))
    torch.testing.assert_close(first.frequencies, second.frequencies)


def test_variable_prefix_masks_produce_explicit_padding_masks():
    source = torch.arange(2 * 32, dtype=torch.float32).reshape(2, 1, 32)
    source_mask = torch.arange(32)[None, :] < torch.tensor([32, 20])[:, None]
    selection = SourceCycleExtractor(top_k=2)(source, source_mask)
    views, masks = CycleViewBuilder(patch_width=5)(source, selection)
    assert selection.valid_lengths.tolist() == [32, 20]
    for branch, (view, mask) in enumerate(zip(views, masks)):
        assert view.shape[:2] == mask.shape
        torch.testing.assert_close(mask.sum(dim=1), selection.periods[:, branch])
    bad_mask = source_mask.clone()
    bad_mask[1, 3] = False
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        SourceCycleExtractor(top_k=2)(source, bad_mask)


def test_grouped_cycle_view_matches_equations_three_and_four():
    source = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8)
    selection = SourceCycleExtractor(top_k=1)(source)
    selection = type(selection)(
        frequencies=torch.tensor([[2], [4]]),
        periods=torch.tensor([[4], [2]]),
        amplitudes=torch.ones(2, 1),
        weights=torch.ones(2, 1),
        valid_lengths=torch.tensor([8, 8]),
        fallback=torch.zeros(2, dtype=torch.bool),
    )
    views, masks = CycleViewBuilder(patch_width=3)(source, selection)
    expected_first = torch.tensor([[0, 1, 1], [2, 3, 3], [4, 5, 5], [6, 7, 7]])
    expected_second = torch.tensor([[8, 9, 10], [12, 13, 14]])
    torch.testing.assert_close(views[0][0], expected_first.float())
    torch.testing.assert_close(views[0][1, :2], expected_second.float())
    assert masks[0].tolist() == [[True] * 4, [True, True, False, False]]


def test_grouped_cycle_view_preserves_source_gradient():
    source = torch.randn(3, 1, 32, requires_grad=True)
    selection = SourceCycleExtractor(top_k=2)(source)
    views, _ = CycleViewBuilder(patch_width=8)(source, selection)
    sum(view.square().mean() for view in views).backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    assert torch.count_nonzero(source.grad) > 0


def test_tensorized_restore_matches_flatten_truncate_and_tail_fill():
    block = CycleAwareTransformerBlock(6, 1, 2, 2, 1, 1, 4, 0.0)
    encoded = torch.tensor(
        [[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0], [99.0, 99.0]]]
    )
    mask = torch.tensor([[True, True, True], [True, True, False]])
    restored, fallback = block._restore(encoded, mask, torch.tensor([5, 4]))
    torch.testing.assert_close(
        restored[:, 0],
        torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 4.0], [6.0, 7.0, 8.0, 9.0, 9.0, 9.0]]),
    )
    assert fallback.tolist() == [False, False]


def test_cat_output_shape_determinism_diagnostics_and_leakage_guard():
    torch.manual_seed(7)
    model = _small_model(dropout=0.1).eval()
    source = torch.randn(3, 1, 64)
    first, diagnostics = model(source, return_diagnostics=True)
    second = model(source)
    assert first.shape == (3, 1, 64)
    torch.testing.assert_close(first, second)
    assert len([key for key in diagnostics if key.endswith("cycle_fallback")]) == 2
    with pytest.raises(TypeError):
        model(source, target=torch.randn_like(source))
    with pytest.raises(ValueError, match="separately labelled adaptation"):
        CATransformer(input_length=64, output_channels=2)
    assert not any(key.startswith("projection.") for key in model.state_dict())
    assert model.nfe == 1


def test_cat_loss_is_finite_and_backpropagates():
    model = _small_model().train()
    source = torch.randn(2, 1, 64)
    target = torch.randn(2, 1, 64)
    prediction = model(source)
    loss, components = CATLoss(kl_weight=1.0)(prediction, target)
    assert torch.isfinite(loss)
    assert set(components) == {"total_loss", "mse_loss", "kl_loss"}
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_runtime_guards_distinguish_amp_overflow_and_empty_interrupt_message():
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([1.0, 2.0])
    assert gradients_are_finite([parameter])
    parameter.grad[1] = float("inf")
    assert not gradients_are_finite([parameter])
    assert exception_summary(KeyboardInterrupt()) == "KeyboardInterrupt"
    assert exception_summary(ValueError("first line\nprivate detail")) == "first line"


def test_cat_checkpoint_roundtrip_retains_recovery_state(tmp_path: Path):
    model = _small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    generator = torch.Generator().manual_seed(31)
    payload = {
        "schema_version": 1,
        "kind": "independent_catransformer_reproduction",
        "epoch": 25,
        "global_step": 100,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": {},
        "config": {"reproduction_label": "CAT-PPG (reproduced)", "cycle_source": "source_ppg_fft_only", "nfe": 1},
        "normalization": {"method": "window_minmax"},
        "output_spec": {"channels": 1, "length": 512},
        "rng_states": capture_rng_states(),
        "data_loader_generator_state": generator.get_state(),
        "provenance": {
            "git_commit": "synthetic",
            "command": "pytest",
            "paper_doi": "10.1109/JBHI.2024.3482853",
            "implementation_status": "independent_paper_based_reproduction",
        },
    }
    path = tmp_path / "checkpoint.pt"
    save_cat_checkpoint(payload, path)
    restored = load_cat_checkpoint(path, "cpu")
    assert restored["epoch"] == 25
    torch.testing.assert_close(
        restored["data_loader_generator_state"], payload["data_loader_generator_state"]
    )


def test_resume_contract_rejects_architecture_change():
    config = Path(__file__).resolve().parents[1] / "configs/baselines/cat_ppg_mimic_afib_seed31.yaml"
    args = parse_args_with_config(["--config", str(config)])
    checkpoint = {"config": _resolved_config(args), "epoch": 25}
    _validate_resume_contract(args, checkpoint)
    checkpoint["config"] = dict(checkpoint["config"])
    checkpoint["config"]["d_model"] = 64
    with pytest.raises(ValueError, match="d_model"):
        _validate_resume_contract(args, checkpoint)
