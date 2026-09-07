import numpy as np

from scripts.analyze_rcfm_ot_random_window_multiseed_clinical import (
    SEEDS,
    _group_common_pairs,
    _holm,
    _paired_test,
    _seed_metrics,
)


def test_group_common_pairs_intersects_rows_and_averages_within_group():
    rows = {}
    for seed in SEEDS:
        rows[seed] = {
            ("0", "I"): {"__group__": ("p1", "p1"), "qrs_ms": (90.0, 91.0 + seed - 31)},
            ("1", "I"): {"__group__": ("p1", "p1"), "qrs_ms": (110.0, 109.0 + seed - 31)},
            ("2", "I"): {"__group__": ("p2", "p2"), "qrs_ms": (100.0, 102.0 + seed - 31)},
        }
    rows[31][("extra", "I")] = {"__group__": ("p3", "p3"), "qrs_ms": (80.0, 80.0)}
    groups, reference, generated, counts = _group_common_pairs(rows, "qrs_ms")
    assert groups.tolist() == ["p1", "p2"]
    np.testing.assert_allclose(reference, [100.0, 100.0])
    np.testing.assert_allclose(generated[31], [100.0, 102.0])
    np.testing.assert_array_equal(counts, [2, 1])


def test_seed_metrics_and_paired_test_do_not_treat_seeds_as_groups():
    reference = np.arange(10.0)
    generated = reference + np.linspace(0.1, 1.0, 10)
    metrics = _seed_metrics(reference, generated)
    assert metrics["mae"] == np.mean(generated - reference)
    result = _paired_test(reference, generated, 31, draws=1000) if False else _paired_test(reference, generated, 31)
    assert result["n_groups"] == 10
    assert result["difference_definition"] == "three_seed_mean_generated_minus_reference"
    assert 0 <= result["raw_p_value"] <= 1


def test_holm_is_monotone_in_sorted_raw_p_values():
    rows = [
        {"status": "ok", "raw_p_value": 0.04},
        {"status": "ok", "raw_p_value": 0.01},
        {"status": "ok", "raw_p_value": 0.03},
    ]
    _holm(rows)
    ordered = sorted(rows, key=lambda row: row["raw_p_value"])
    adjusted = [row["holm_adjusted_p_value_dataset_11_parameters"] for row in ordered]
    assert adjusted == sorted(adjusted)
    assert all(row["holm_adjusted_p_value_dataset_11_parameters"] >= row["raw_p_value"] for row in rows)


def test_large_wilcoxon_p_value_never_serializes_as_zero():
    reference = np.zeros(6000)
    generated = np.linspace(-0.1, 1.0, 6000)
    result = _paired_test(reference, generated, 31)
    assert result["wilcoxon_method"] == "approx"
    assert result["raw_p_value"] > 0
    assert result["raw_log10_p_value"] < 0
