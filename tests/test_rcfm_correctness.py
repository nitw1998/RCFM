import inspect

import torch
import torch.nn as nn

from conditional_flow_matcher import (
    ConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from rcfm import RegionAwareConditionalFlowMatching


class ConstantVelocity(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, x, conditions, t):
        del conditions, t
        self.calls += 1
        return torch.full_like(x, self.value)


def _conditions(values: torch.Tensor):
    return {
        "down_conditions": [values.clone()],
        "up_conditions": [values.clone()],
    }


def test_linear_path_and_velocity_follow_canonical_direction():
    matcher = ConditionalFlowMatcher(sigma=0.0)
    noise = torch.tensor([[[1.0, -2.0]]])
    target = torch.tensor([[[5.0, 4.0]]])
    time = torch.tensor([0.25])
    epsilon = torch.zeros_like(noise)

    path = matcher.sample_xt(noise, target, time, epsilon)
    velocity = matcher.compute_conditional_flow(noise, target, time, path)

    torch.testing.assert_close(path, 0.75 * noise + 0.25 * target)
    torch.testing.assert_close(velocity, target - noise)


def test_linear_path_velocity_matches_finite_difference():
    matcher = ConditionalFlowMatcher(sigma=0.0)
    source = torch.tensor([[[1.0, -2.0]]])
    target = torch.tensor([[[5.0, 4.0]]])
    time = torch.tensor([0.4])
    epsilon = torch.zeros_like(source)
    delta = 1e-4
    before = matcher.sample_xt(source, target, time - delta, epsilon)
    after = matcher.sample_xt(source, target, time + delta, epsilon)
    derivative = (after - before) / (2 * delta)
    path = matcher.sample_xt(source, target, time, epsilon)
    analytical = matcher.compute_conditional_flow(source, target, time, path)
    torch.testing.assert_close(derivative, analytical, rtol=1e-3, atol=1e-3)


def test_noncanonical_matcher_velocities_match_their_analytical_paths():
    source = torch.tensor([[[1.0, -0.5]]], dtype=torch.float64)
    target = torch.tensor([[[2.0, 3.0]]], dtype=torch.float64)
    epsilon = torch.tensor([[[0.2, -0.3]]], dtype=torch.float64)
    time = torch.tensor([0.4], dtype=torch.float64)
    delta = 1e-6
    matchers = [
        TargetConditionalFlowMatcher(sigma=0.1),
        VariancePreservingConditionalFlowMatcher(sigma=0.0),
        SchrodingerBridgeConditionalFlowMatcher(sigma=0.2),
    ]
    for matcher in matchers:
        before = matcher.sample_xt(source, target, time - delta, epsilon)
        after = matcher.sample_xt(source, target, time + delta, epsilon)
        derivative = (after - before) / (2 * delta)
        path = matcher.sample_xt(source, target, time, epsilon)
        analytical = matcher.compute_conditional_flow(source, target, time, path)
        torch.testing.assert_close(derivative, analytical, rtol=1e-5, atol=1e-5)


def test_forward_euler_sampling_integrates_from_zero_to_one():
    flow = ConstantVelocity(value=2.0)
    model = RegionAwareConditionalFlowMatching(
        flow_model=flow,
        use_minibatch_ot=False,
    )
    conditions = _conditions(torch.zeros(2, 1, 8))

    torch.manual_seed(13)
    initial = torch.randn(2, 1, 8)
    torch.manual_seed(13)
    result = model.sample(conditions=conditions, shape=(2, 1, 8), steps=4)

    torch.testing.assert_close(result, initial + 2.0)
    assert flow.calls == 4


def test_sampling_accepts_fixed_initial_noise_without_consuming_rng():
    flow = ConstantVelocity(value=1.0)
    model = RegionAwareConditionalFlowMatching(flow_model=flow, use_minibatch_ot=False)
    conditions = _conditions(torch.zeros(2, 1, 4))
    initial = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
    state_before = torch.random.get_rng_state()

    result = model.sample(
        conditions=conditions,
        shape=initial.shape,
        steps=2,
        initial_noise=initial,
    )

    torch.testing.assert_close(result, initial + 1.0)
    torch.testing.assert_close(torch.random.get_rng_state(), state_before)
    torch.testing.assert_close(initial, torch.arange(8).reshape(2, 1, 4).float())


def test_ot_reindexes_target_condition_and_mask_together(monkeypatch):
    model = RegionAwareConditionalFlowMatching(
        flow_model=ConstantVelocity(value=0.0),
        use_minibatch_ot=True,
    )
    source = torch.tensor([[[10.0]], [[20.0]], [[30.0]]])
    target = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    condition_values = torch.tensor([[[101.0]], [[102.0]], [[103.0]]])
    mask = torch.tensor([[[201.0]], [[202.0]], [[203.0]]])
    source_idx = torch.tensor([2, 0, 1])
    target_idx = torch.tensor([1, 2, 0])
    monkeypatch.setattr(model, "_sample_ot_indices", lambda *_: (source_idx, target_idx))

    coupled = model._apply_minibatch_ot(
        source,
        target,
        _conditions(condition_values),
        mask,
    )

    torch.testing.assert_close(coupled[0], source[source_idx])
    torch.testing.assert_close(coupled[1], target[target_idx])
    torch.testing.assert_close(coupled[2]["down_conditions"][0], condition_values[target_idx])
    torch.testing.assert_close(coupled[2]["up_conditions"][0], condition_values[target_idx])
    torch.testing.assert_close(coupled[3], mask[target_idx])


def test_ot_metadata_guard_detects_misaligned_condition_or_mask(monkeypatch):
    model = RegionAwareConditionalFlowMatching(
        flow_model=ConstantVelocity(value=0.0),
        use_minibatch_ot=True,
    )
    source = torch.zeros(3, 1, 1)
    target = torch.ones(3, 1, 1)
    index = torch.tensor([2, 0, 1])
    monkeypatch.setattr(model, "_sample_ot_indices", lambda *_: (index, index))
    metadata = {
        "target_sample_id": ["a", "b", "c"],
        "condition_target_id": ["a", "wrong", "c"],
        "mask_target_id": ["a", "b", "c"],
    }

    try:
        model._apply_minibatch_ot(
            source,
            target,
            _conditions(target),
            torch.ones_like(target),
            sample_metadata=metadata,
        )
    except ValueError as error:
        assert "associations diverged" in str(error)
    else:
        raise AssertionError("Expected metadata association mismatch to be rejected")


def test_association_debug_requires_ids_and_checks_batch_sizes():
    model = RegionAwareConditionalFlowMatching(
        flow_model=ConstantVelocity(value=0.0),
        use_minibatch_ot=False,
        association_debug=True,
    )
    target = torch.ones(2, 1, 2)
    conditions = _conditions(target)
    try:
        model(target=target, conditions=conditions)
    except ValueError as error:
        assert "requires sample_metadata" in str(error)
    else:
        raise AssertionError("Expected debug mode without sample IDs to be rejected")

    bad_conditions = _conditions(torch.ones(1, 1, 2))
    try:
        model._apply_minibatch_ot(target, target, bad_conditions, None)
    except ValueError as error:
        assert "condition batch sizes" in str(error)
    else:
        raise AssertionError("Expected inconsistent condition batch size to be rejected")


def test_ot_index_guard_rejects_wrong_dtype_and_out_of_range(monkeypatch):
    model = RegionAwareConditionalFlowMatching(
        flow_model=ConstantVelocity(value=0.0),
        use_minibatch_ot=True,
    )
    source = torch.zeros(2, 1, 1)
    target = torch.ones(2, 1, 1)

    monkeypatch.setattr(
        model,
        "_sample_ot_indices",
        lambda *_: (torch.tensor([0.0, 1.0]), torch.tensor([0, 1])),
    )
    try:
        model._apply_minibatch_ot(source, target, _conditions(target), None)
    except TypeError as error:
        assert "torch.long" in str(error)
    else:
        raise AssertionError("Expected floating point indices to be rejected")

    monkeypatch.setattr(
        model,
        "_sample_ot_indices",
        lambda *_: (torch.tensor([0, 1]), torch.tensor([0, 2])),
    )
    try:
        model._apply_minibatch_ot(source, target, _conditions(target), None)
    except IndexError as error:
        assert "target index" in str(error)
    else:
        raise AssertionError("Expected out-of-range indices to be rejected")


def test_mask_is_training_only_and_sb_requires_observable_outer_ot():
    assert "region_mask" not in inspect.signature(RegionAwareConditionalFlowMatching.sample).parameters

    try:
        RegionAwareConditionalFlowMatching(
            flow_model=ConstantVelocity(value=0.0),
            flow_matcher_type="sb",
            sigma=0.1,
            use_minibatch_ot=False,
        )
    except ValueError as error:
        assert "observable outer" in str(error)
    else:
        raise AssertionError("Expected untracked internal SB coupling to be rejected")


def test_sb_ablation_uses_one_observable_outer_coupling_for_all_paired_tensors(monkeypatch):
    model = RegionAwareConditionalFlowMatching(
        flow_model=ConstantVelocity(value=0.0),
        flow_matcher_type="sb",
        sigma=0.1,
        use_minibatch_ot=True,
        ot_method="exact",
        ot_sampling_strategy="multinomial",
    )
    assert model.flow_matcher.ot_sampler is None
    source = torch.tensor([[[10.0]], [[20.0]], [[30.0]]])
    target = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    condition = torch.tensor([[[101.0]], [[102.0]], [[103.0]]])
    mask = torch.tensor([[[201.0]], [[202.0]], [[203.0]]])
    source_idx = torch.tensor([2, 0, 1])
    target_idx = torch.tensor([1, 2, 0])
    monkeypatch.setattr(model, "_sample_ot_indices", lambda *_: (source_idx, target_idx))

    coupled = model._apply_minibatch_ot(source, target, _conditions(condition), mask)

    torch.testing.assert_close(coupled[0], source[source_idx])
    torch.testing.assert_close(coupled[1], target[target_idx])
    torch.testing.assert_close(coupled[2]["down_conditions"][0], condition[target_idx])
    torch.testing.assert_close(coupled[3], mask[target_idx])


def test_canonical_rcfm_rejects_nonzero_sigma():
    try:
        RegionAwareConditionalFlowMatching(
            flow_model=ConstantVelocity(value=0.0),
            flow_matcher_type="conditional",
            sigma=0.1,
            use_minibatch_ot=False,
        )
    except ValueError as error:
        assert "requires sigma=0" in str(error)
    else:
        raise AssertionError("Expected nonzero sigma to be rejected for canonical RCFM")


def test_noncanonical_path_with_outer_ot_requires_explicit_opt_in():
    for path_type in ("target", "vp"):
        try:
            RegionAwareConditionalFlowMatching(
                flow_model=ConstantVelocity(value=0.0),
                flow_matcher_type=path_type,
                sigma=0.1,
                use_minibatch_ot=True,
            )
        except ValueError as error:
            assert "allow_noncanonical_ot_path=True" in str(error)
        else:
            raise AssertionError(f"Expected ambiguous {path_type}-plus-OT path to be rejected")

    RegionAwareConditionalFlowMatching(
        flow_model=ConstantVelocity(value=0.0),
        flow_matcher_type="target",
        sigma=0.1,
        use_minibatch_ot=True,
        allow_noncanonical_ot_path=True,
    )


def test_loss_and_path_decomposition_are_finite_for_empty_regions():
    model = RegionAwareConditionalFlowMatching(
        flow_model=ConstantVelocity(value=0.0),
        region_weight=2.0,
        use_minibatch_ot=False,
    )
    target = torch.ones(2, 1, 4)
    source = torch.zeros_like(target)
    conditions = _conditions(target)

    no_roi = model(
        target=target,
        source=source,
        conditions=conditions,
        region_mask=torch.zeros_like(target),
    )
    all_roi = model(
        target=target,
        source=source,
        conditions=conditions,
        region_mask=torch.ones_like(target),
    )

    tensor_values = [value for value in no_roi.values() if torch.is_tensor(value)]
    tensor_values += [value for value in all_roi.values() if torch.is_tensor(value)]
    assert all(torch.isfinite(value).all() for value in tensor_values)
    assert no_roi["train/roi_mse"].item() == 0.0
    assert no_roi["train/non_roi_mse"].item() == 1.0
    assert all_roi["train/roi_mse"].item() == 1.0
    assert all_roi["train/non_roi_mse"].item() == 0.0
    assert all_roi["train/effective_mean_weight"].item() == 3.0
    assert all_roi["flow/path_type"] == "conditional"
