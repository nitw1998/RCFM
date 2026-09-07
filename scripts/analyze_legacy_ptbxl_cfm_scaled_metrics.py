#!/usr/bin/env python3
"""Compute record-scaled mV RMSE and wFD for legacy PTB-XL CFM III-to-V5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.rcfm.metrics.waveform import waveform_frechet_distance


V5_INDEX = 10
SAMPLES = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inverse_minmax(normalized: np.ndarray, minima_mv: np.ndarray, ranges_mv: np.ndarray) -> np.ndarray:
    values = np.asarray(normalized, dtype=np.float64)
    minima = np.asarray(minima_mv, dtype=np.float64)
    ranges = np.asarray(ranges_mv, dtype=np.float64)
    if values.ndim != 2 or minima.shape != (len(values),) or ranges.shape != (len(values),):
        raise ValueError("expected normalized [records, samples] and one min/range per record")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(minima)):
        raise FloatingPointError("normalization inputs contain NaN or Inf")
    if not np.all(np.isfinite(ranges)) or np.any(ranges <= 0):
        raise ValueError("all record ranges must be finite and positive")
    return (values + 1.0) * ranges[:, None] / 2.0 + minima[:, None]


def scaled_errors(
    reference: np.ndarray, generated: np.ndarray, ranges_mv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    real = np.asarray(reference, dtype=np.float64)
    fake = np.asarray(generated, dtype=np.float64)
    ranges = np.asarray(ranges_mv, dtype=np.float64)
    if real.shape != fake.shape or real.ndim != 2 or ranges.shape != (len(real),):
        raise ValueError("reference/generated must match [records, samples] with one range per record")
    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(fake)):
        raise FloatingPointError("waveforms contain NaN or Inf")
    if not np.all(np.isfinite(ranges)) or np.any(ranges <= 0):
        raise ValueError("all record ranges must be finite and positive")
    scales = ranges / 2.0
    error_mv = (fake - real) * scales[:, None]
    record_rmse_mv = np.sqrt(np.mean(error_mv**2, axis=1))
    return scales, error_mv, record_rmse_mv


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _plot(record_rmse_mv: np.ndarray, scales: np.ndarray, path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Times New Roman", "Times"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    axes[0].hist(record_rmse_mv, bins=50, color="#2878b5", alpha=0.85, edgecolor="white", linewidth=0.25)
    axes[0].axvline(np.median(record_rmse_mv), color="#c43d4b", linewidth=1.0, label="Median")
    axes[0].set(xlabel="Per-record RMSE (mV)", ylabel="Record count", title="CFM III-to-V5 physical RMSE")
    axes[0].legend(frameon=False)
    axes[1].scatter(scales, record_rmse_mv, s=5, alpha=0.28, color="#3b8d5a", rasterized=True)
    axes[1].set(
        xlabel="V5 error scale = range/2 (mV/unit)",
        ylabel="Per-record RMSE (mV)",
        title="Effect of record-specific scaling",
    )
    for axis in axes:
        axis.grid(alpha=0.15, linewidth=0.4)
    figure.savefig(path, dpi=400)
    plt.close(figure)


def run(evaluation_dir: Path, preprocessed_dir: Path, output_dir: Path, epoch: int = 999) -> Path:
    evaluation_dir = evaluation_dir.resolve()
    preprocessed_dir = preprocessed_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = evaluation_dir / "paired_reference.npz"
    prediction_path = evaluation_dir / f"predictions_epoch_{epoch}.npy"
    minima_path = preprocessed_dir / "record_minima_test.npy"
    ranges_path = preprocessed_dir / "record_ranges_test.npy"
    waveform_path = preprocessed_dir / "X_test_resampled.npy"

    with np.load(reference_path, allow_pickle=False) as artifact:
        reference_3d = np.asarray(artifact["targets"], dtype=np.float64)
    generated_3d = np.asarray(np.load(prediction_path, allow_pickle=False), dtype=np.float64)
    minima_all = np.asarray(np.load(minima_path, allow_pickle=False), dtype=np.float64)
    ranges_all = np.asarray(np.load(ranges_path, allow_pickle=False), dtype=np.float64)
    stored = np.load(waveform_path, mmap_mode="r", allow_pickle=False)

    expected = (len(reference_3d), 1, SAMPLES)
    if reference_3d.shape != expected or generated_3d.shape != expected:
        raise ValueError(f"reference and prediction must both have shape {expected}")
    if minima_all.shape != (len(reference_3d), 12) or ranges_all.shape != minima_all.shape:
        raise ValueError("PTB-XL fold-10 min/range sidecars have the wrong shape")
    if stored.shape[0] != len(reference_3d) or stored.shape[1] < SAMPLES or stored.shape[2] != 12:
        raise ValueError("PTB-XL fold-10 waveform array has the wrong shape")

    reference = reference_3d[:, 0]
    generated = generated_3d[:, 0]
    minima_mv = minima_all[:, V5_INDEX]
    ranges_mv = ranges_all[:, V5_INDEX]
    scales, error_mv, record_rmse_mv = scaled_errors(reference, generated, ranges_mv)
    record_normalized_rmse = np.sqrt(np.mean((generated - reference) ** 2, axis=1))
    record_mae_mv = np.mean(np.abs(error_mv), axis=1)

    scale_only_reference_mv = reference * scales[:, None]
    scale_only_generated_mv = generated * scales[:, None]
    physical_reference_mv = inverse_minmax(reference, minima_mv, ranges_mv)
    physical_generated_mv = inverse_minmax(generated, minima_mv, ranges_mv)
    stored_reference_mv = np.asarray(stored[:, :SAMPLES, V5_INDEX], dtype=np.float64)
    reconstruction_error = float(np.max(np.abs(physical_reference_mv - stored_reference_mv)))
    if reconstruction_error > 2e-6:
        raise ValueError(f"target inverse reconstruction disagrees with stored V5: {reconstruction_error}")

    normalized_wfd = waveform_frechet_distance(reference, generated)
    scale_only_wfd = waveform_frechet_distance(scale_only_reference_mv, scale_only_generated_mv)
    physical_wfd = waveform_frechet_distance(physical_reference_mv, physical_generated_mv)

    csv_path = output_dir / "per_record_scaled_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        columns = [
            "dataset_row", "v5_min_mV", "v5_range_mV", "error_scale_mV_per_normalized_unit",
            "normalized_rmse", "rmse_mV", "mae_mV",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(len(reference)):
            writer.writerow(
                {
                    "dataset_row": index,
                    "v5_min_mV": minima_mv[index],
                    "v5_range_mV": ranges_mv[index],
                    "error_scale_mV_per_normalized_unit": scales[index],
                    "normalized_rmse": record_normalized_rmse[index],
                    "rmse_mV": record_rmse_mv[index],
                    "mae_mV": record_mae_mv[index],
                }
            )

    plot_path = output_dir / "per_record_rmse_mV_distribution.png"
    _plot(record_rmse_mv, scales, plot_path)
    result = {
        "schema_version": 1,
        "status": "completed",
        "dataset": "PTB-XL 1.0.1",
        "split": "official_fold_10",
        "model": "legacy CFM Lead III to V5",
        "records": int(len(reference)),
        "samples_per_record": SAMPLES,
        "normalization": "independent per-record/per-lead min-max to [-1,1] on the first 512 samples",
        "definitions": {
            "error_scale": "held-out real V5 range_mV / 2 for each record",
            "rmse_mV": "sqrt(mean(((generated_normalized-reference_normalized)*record_error_scale)^2))",
            "scale_only_wFD_mV2": "complete-fold raw-waveform Gaussian Frechet distance after multiplying each normalized real/generated record by its V5 range/2; record midpoint offsets remain removed",
            "full_oracle_inverse_wFD_mV2": "complete-fold raw-waveform Gaussian Frechet distance after applying each held-out real V5 min and range to both real and generated records",
        },
        "results": {
            "normalized_rmse_global": float(np.sqrt(np.mean((generated - reference) ** 2))),
            "rmse_mV_global": float(np.sqrt(np.mean(error_mv**2))),
            "mae_mV_global": float(np.mean(np.abs(error_mv))),
            "normalized_wFD": normalized_wfd,
            "scale_only_wFD_mV2": scale_only_wfd,
            "full_oracle_inverse_wFD_mV2": physical_wfd,
            "per_record_rmse_mV_distribution": _distribution(record_rmse_mv),
            "error_scale_mV_per_normalized_unit_distribution": _distribution(scales),
        },
        "verification": {
            "inverse_reference_vs_stored_V5_max_abs_error_mV": reconstruction_error,
            "rmse_translation_invariance_max_abs_error_mV": float(
                np.max(
                    np.abs(
                        np.sqrt(np.mean((physical_generated_mv - physical_reference_mv) ** 2, axis=1))
                        - record_rmse_mv
                    )
                )
            ),
        },
        "inputs": {
            "paired_reference_sha256": _sha256(reference_path),
            "prediction_sha256": _sha256(prediction_path),
            "record_minima_test_sha256": _sha256(minima_path),
            "record_ranges_test_sha256": _sha256(ranges_path),
            "X_test_resampled_sha256": _sha256(waveform_path),
        },
        "outputs": {
            "per_record_csv": csv_path.name,
            "distribution_plot": plot_path.name,
        },
        "claim_boundary": "All mV metrics use held-out real V5 record-specific scaling coefficients. They are oracle scale-assisted descriptive audits, not deployable inference metrics. wFD has squared-amplitude units (mV^2).",
    }
    summary_path = output_dir / "scaled_metric_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["results"], indent=2, sort_keys=True), flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation_dir", type=Path, required=True)
    parser.add_argument("--preprocessed_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=999)
    args = parser.parse_args()
    run(args.evaluation_dir, args.preprocessed_dir, args.output_dir, args.epoch)


if __name__ == "__main__":
    main()
