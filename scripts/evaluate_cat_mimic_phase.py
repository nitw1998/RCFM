"""Evaluate CAT-PPG on MIMIC-AFib with frozen oracle phase sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _array_sha256, _sha256, _waveform_metrics
from scripts.evaluate_mimic_phase_corrected import _fixed_support_align
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic


def _metric_row(summary: dict[str, object], phase_mode: str) -> dict[str, object]:
    correlation = summary["per_record_pearson"]
    agreement = summary["pointwise_bland_altman_descriptive_only"]
    return {
        "model": "cat",
        "phase_mode": phase_mode,
        "records": 1800,
        "support_samples": 512 if phase_mode == "raw_full_window" else 480,
        "rmse": summary["rmse"],
        "mae": summary["mae"],
        "waveform_fd": summary["waveform_fd"],
        "per_record_pearson_mean": correlation["mean"],
        "per_record_pearson_median": correlation["median"],
        "pointwise_bland_altman_bias": agreement["bias"],
        "pointwise_bland_altman_lower_limit": agreement["lower_limit"],
        "pointwise_bland_altman_upper_limit": agreement["upper_limit"],
    }


def run(args: argparse.Namespace) -> Path:
    if args.max_lag_samples != 16:
        raise ValueError("the frozen CAT phase protocol requires max_lag_samples=16")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = args.predictions.resolve()
    with np.load(prediction_path, allow_pickle=False) as artifact:
        required = {"targets", "predictions", "sources"}
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"CAT prediction artifact is missing arrays: {sorted(missing)}")
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        predictions = np.asarray(artifact["predictions"], dtype=np.float32)
        sources = np.asarray(artifact["sources"], dtype=np.float32)
    expected_shape = (1800, 1, 512)
    if any(array.shape != expected_shape for array in (targets, predictions, sources)):
        raise ValueError("CAT phase arrays must have shape (1800,1,512)")
    if any(not np.all(np.isfinite(array)) for array in (targets, predictions, sources)):
        raise FloatingPointError("CAT phase arrays must be finite")

    lag_summary, lag_values = _lag_diagnostic(
        targets, predictions, args.max_lag_samples, args.sampling_rate
    )
    shifts = lag_values["best_lag_samples"].astype(np.int32)
    target_center, unshifted, aligned = _fixed_support_align(
        targets, predictions, shifts, args.max_lag_samples
    )
    raw_summary, raw_per_record = _waveform_metrics(targets, predictions)
    unshifted_summary, unshifted_per_record = _waveform_metrics(target_center, unshifted)
    aligned_summary, aligned_per_record = _waveform_metrics(target_center, aligned)
    rows = [
        _metric_row(raw_summary, "raw_full_window"),
        _metric_row(unshifted_summary, "unshifted_fixed_support"),
        _metric_row(aligned_summary, "oracle_aligned_fixed_support"),
    ]

    phase_path = output_dir / "cat_oracle_aligned_predictions_maxlag_16.npz"
    np.savez_compressed(
        phase_path,
        targets=target_center,
        cat_unshifted_predictions=unshifted,
        cat_oracle_aligned_predictions=aligned,
        cat_oracle_shifts=shifts,
    )
    csv_path = output_dir / "waveform_phase_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_record_path = output_dir / "per_record_phase_metrics.csv"
    with per_record_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "row", "oracle_shift_samples", "oracle_shift_ms", "raw_rmse", "raw_pearson_r",
            "unshifted_fixed_support_rmse", "unshifted_fixed_support_pearson_r",
            "oracle_aligned_fixed_support_rmse", "oracle_aligned_fixed_support_pearson_r",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, shift in enumerate(shifts):
            writer.writerow({
                "row": index,
                "oracle_shift_samples": int(shift),
                "oracle_shift_ms": float(shift) * 1000.0 / args.sampling_rate,
                "raw_rmse": raw_per_record["rmse"][index],
                "raw_pearson_r": raw_per_record["pearson_r"][index],
                "unshifted_fixed_support_rmse": unshifted_per_record["rmse"][index],
                "unshifted_fixed_support_pearson_r": unshifted_per_record["pearson_r"][index],
                "oracle_aligned_fixed_support_rmse": aligned_per_record["rmse"][index],
                "oracle_aligned_fixed_support_pearson_r": aligned_per_record["pearson_r"][index],
            })
    summary_path = output_dir / "waveform_phase_summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": 1,
        "model": "CAT-PPG (reproduced)",
        "raw_full_window": raw_summary,
        "unshifted_fixed_support": unshifted_summary,
        "oracle_aligned_fixed_support": aligned_summary,
        "oracle_lag_diagnostic": {
            **lag_summary,
            "fraction_at_search_boundary": float(np.mean(np.abs(shifts) == args.max_lag_samples)),
            "fraction_rmse_improved": float(np.mean(
                aligned_per_record["rmse"] < unshifted_per_record["rmse"]
            )),
        },
        "interpretation_boundary": (
            "Raw full-window metrics are primary. Oracle alignment selects each lag using its real "
            "test target and is a diagnostic sensitivity analysis, not deployable preprocessing."
        ),
    }, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    outputs = (phase_path, csv_path, per_record_path, summary_path)
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "input": {"path": str(prediction_path), "sha256": _sha256(prediction_path)},
        "protocol": {
            "records": 1800,
            "sampling_rate_hz": args.sampling_rate,
            "max_lag_samples": 16,
            "max_lag_ms": 125.0,
            "lag_objective": "maximize per-window Pearson using the real ECG target",
            "fixed_support_samples": 480,
            "targets_sha256": _array_sha256(targets),
        },
        "execution": {"python": platform.python_version(), "numpy": np.__version__},
        "outputs": {path.name: _sha256(path) for path in outputs},
    }, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sampling_rate", type=int, default=128)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
