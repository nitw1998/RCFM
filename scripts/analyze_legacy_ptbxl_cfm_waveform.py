#!/usr/bin/env python3
"""Summarize waveform agreement for a legacy PTB-XL III-to-V5 CFM prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def waveform_summary(reference: np.ndarray, generated: np.ndarray) -> dict[str, object]:
    real = np.asarray(reference, dtype=np.float64)
    fake = np.asarray(generated, dtype=np.float64)
    if real.shape != fake.shape or real.ndim != 3 or real.shape[1] != 1:
        raise ValueError("reference and generated must share shape [records, 1, samples]")
    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(fake)):
        raise FloatingPointError("waveform inputs contain NaN or Inf")
    error = fake - real
    centered_real = real[:, 0] - np.mean(real[:, 0], axis=1, keepdims=True)
    centered_fake = fake[:, 0] - np.mean(fake[:, 0], axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(centered_real**2, axis=1) * np.sum(centered_fake**2, axis=1))
    record_pearson = np.divide(
        np.sum(centered_real * centered_fake, axis=1), denominator,
        out=np.full(len(real), np.nan), where=denominator > 0,
    )
    difference_sd = float(np.std(error, ddof=1))
    bias = float(np.mean(error))
    record_rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
    return {
        "records": int(len(real)), "samples_per_record": int(real.shape[2]),
        "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error))),
        "record_rmse_mean": float(np.mean(record_rmse)),
        "record_rmse_median": float(np.median(record_rmse)),
        "record_pearson_mean": float(np.nanmean(record_pearson)),
        "record_pearson_median": float(np.nanmedian(record_pearson)),
        "pointwise_bland_altman_descriptive_only": {
            "n_points": int(error.size), "difference_definition": "generated_minus_reference",
            "bias": bias, "difference_sd": difference_sd,
            "lower_limit": bias - 1.96 * difference_sd,
            "upper_limit": bias + 1.96 * difference_sd,
            "caveat": "samples within records and patients are correlated; limits are descriptive, not inferential",
        },
    }


def _plot(reference: np.ndarray, generated: np.ndarray, summary: dict[str, object], stem: Path) -> list[Path]:
    real = np.asarray(reference, dtype=np.float64).reshape(-1)
    fake = np.asarray(generated, dtype=np.float64).reshape(-1)
    stride = max(1, int(np.ceil(real.size / 100_000)))
    means = (real[::stride] + fake[::stride]) / 2
    differences = fake[::stride] - real[::stride]
    agreement = summary["pointwise_bland_altman_descriptive_only"]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "Times"],
                         "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axis = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    axis.scatter(means, differences, s=1.5, alpha=0.12, color="#2878b5", rasterized=True)
    axis.axhline(agreement["bias"], color="#c43d4b", linewidth=1.0, label="Bias")
    axis.axhline(agreement["lower_limit"], color="#333333", linestyle="--", linewidth=0.8, label="95% LoA")
    axis.axhline(agreement["upper_limit"], color="#333333", linestyle="--", linewidth=0.8)
    axis.set(title="CFM Lead III to V5 waveform agreement", xlabel="Pair mean (normalized)",
             ylabel="Generated - real (normalized)")
    axis.grid(alpha=0.15, linewidth=0.4); axis.legend(frameon=False)
    paths = []
    for suffix in ("png", "pdf"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=600 if suffix == "png" else None); paths.append(path)
    plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=999)
    args = parser.parse_args()
    reference_path = args.input_dir / "paired_reference.npz"
    prediction_path = args.input_dir / f"predictions_epoch_{args.epoch}.npy"
    with np.load(reference_path, allow_pickle=False) as artifact:
        reference = np.asarray(artifact["targets"], dtype=np.float32)
    generated = np.asarray(np.load(prediction_path, allow_pickle=False), dtype=np.float32)
    summary = waveform_summary(reference, generated)
    plots = _plot(reference, generated, summary, args.input_dir / "waveform_bland_altman")
    summary["provenance"] = {
        "reference_sha256": _sha256(reference_path), "prediction_sha256": _sha256(prediction_path),
        "plot_sha256": {path.name: _sha256(path) for path in plots},
    }
    (args.input_dir / "waveform_agreement.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
