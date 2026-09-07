import numpy as np

from scripts.analyze_rcfm_ot_random_window_multiseed_waveform import (
    DATASET_SPECS,
    SEEDS,
    aggregate_seed_rows,
    calculate_batch_wfd,
    extract_metrics,
)


def test_extracts_nested_mimic_metrics():
    payload = {"waveform": {"oracle_aligned_fixed_support_480": {
        "rmse": 1.0, "mae": 2.0, "waveform_fd": 3.0,
        "per_record_pearson": {"median": 0.75},
    }}}
    assert extract_metrics(payload, DATASET_SPECS["mimic_afib"]) == {
        "rmse": 1.0, "mae": 2.0, "waveform_fd": 3.0,
        "pearson_r_median": 0.75,
    }


def test_aggregate_uses_sample_sd_across_three_seeds():
    rows = []
    for dataset, spec in DATASET_SPECS.items():
        for offset, seed in enumerate(SEEDS):
            rows.append({
                "dataset": dataset, "seed": seed, "phase_mode": spec["phase_mode"],
                "rmse": 1.0 + offset, "mae": 2.0 + offset,
                "waveform_fd": 3.0 + offset, "pearson_r_median": 0.5 + 0.1 * offset,
            })
    aggregated = aggregate_seed_rows(rows)
    assert len(aggregated) == 5
    assert aggregated[0]["rmse_mean"] == 2.0
    assert aggregated[0]["rmse_sample_sd"] == 1.0
    assert np.isclose(aggregated[0]["pearson_r_median_sample_sd"], 0.1)


def test_batch_wfd_uses_unweighted_batch_then_lead_macro_mean(tmp_path):
    source = tmp_path / "evaluation/ptbxl_random_window_rcfm_ot_e200_s31_raw_v1"
    source.mkdir(parents=True)
    reference = np.arange(6 * 2 * 4, dtype=np.float32).reshape(6, 2, 4) / 20
    generated = reference.copy()
    generated[:4, 0] += 0.1
    generated[4:, 1] += 0.3
    np.savez(source / "paired_reference.npz", targets=reference)
    np.save(source / "rcfm_ot_predictions.npy", generated)
    value, rows = calculate_batch_wfd(tmp_path, "ptbxl", 31, batch_size=4)
    assert len(rows) == 4  # two batches by two leads
    by_lead = []
    for lead in (0, 1):
        by_lead.append(np.mean([row["waveform_fd"] for row in rows if row["lead_index"] == lead]))
    assert value == np.mean(by_lead)
