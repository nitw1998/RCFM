"""Compare MIMIC generators under target-informed oracle phase correction."""

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
from typing import Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _array_sha256, _json, _sha256, _waveform_metrics
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic


MODEL_ORDER = ("cfm", "rcfm", "rcfm_ot", "rddm")
DISPLAY_NAMES = {"cfm": "CFM", "rcfm": "RCFM", "rcfm_ot": "RCFM-OT", "rddm": "RDDM"}
COLORS = {"cfm": "#2878b5", "rcfm": "#2f8f5b", "rcfm_ot": "#c43d4b", "rddm": "#d17a00"}


def _parse_max_lags(value: str) -> tuple[int, ...]:
    try:
        lags = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("max lags must be comma-separated integers") from error
    if not lags or any(lag <= 0 for lag in lags) or len(set(lags)) != len(lags):
        raise argparse.ArgumentTypeError("max lags must be unique positive integers")
    return tuple(sorted(lags))


def _load_inputs(flow_path: Path, rddm_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    with np.load(flow_path, allow_pickle=False) as flow, np.load(rddm_path, allow_pickle=False) as rddm:
        required_flow = {
            "targets", "conditions", "cfm_predictions", "rcfm_predictions", "rcfm_ot_predictions"
        }
        required_rddm = {"targets", "conditions", "rddm_predictions", "source_test_rows_before_zero_filter"}
        if required_flow - set(flow.files) or required_rddm - set(rddm.files):
            raise ValueError("prediction artifacts do not contain the required arrays")
        targets = np.asarray(flow["targets"], dtype=np.float32)
        conditions = np.asarray(flow["conditions"], dtype=np.float32)
        if not np.array_equal(targets, rddm["targets"]):
            raise ValueError("RDDM and flow targets are not row-aligned")
        if not np.array_equal(conditions, rddm["conditions"]):
            raise ValueError("RDDM and flow conditions are not row-aligned")
        predictions = {
            "cfm": np.asarray(flow["cfm_predictions"], dtype=np.float32),
            "rcfm": np.asarray(flow["rcfm_predictions"], dtype=np.float32),
            "rcfm_ot": np.asarray(flow["rcfm_ot_predictions"], dtype=np.float32),
            "rddm": np.asarray(rddm["rddm_predictions"], dtype=np.float32),
        }
        original_rows = np.asarray(rddm["source_test_rows_before_zero_filter"], dtype=np.int64)
    expected_shape = (1800, 1, 512)
    arrays = [targets, conditions, *predictions.values()]
    if any(array.shape != expected_shape for array in arrays):
        raise ValueError("all MIMIC arrays must have shape (1800,1,512)")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError("MIMIC phase-comparison arrays must be finite")
    if original_rows.shape != (1800,) or len(np.unique(original_rows)) != 1800:
        raise ValueError("original test-row mapping is invalid")
    return targets, conditions, predictions, original_rows


def _fixed_support_align(
    targets: np.ndarray,
    predictions: np.ndarray,
    shifts: np.ndarray,
    margin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if targets.shape != predictions.shape or targets.ndim != 3:
        raise ValueError("phase alignment requires matching 3D arrays")
    if shifts.shape != (len(targets),) or not np.issubdtype(shifts.dtype, np.integer):
        raise ValueError("phase shifts must be one integer per record")
    length = targets.shape[-1]
    if margin <= 0 or 2 * margin >= length or np.any(np.abs(shifts) > margin):
        raise ValueError("fixed-support margin must contain every requested shift")
    target_center = targets[:, :, margin : length - margin].copy()
    unshifted_center = predictions[:, :, margin : length - margin].copy()
    aligned = np.empty_like(target_center)
    for index, shift in enumerate(shifts):
        start = margin - int(shift)
        stop = length - margin - int(shift)
        aligned[index] = predictions[index, :, start:stop]
    return target_center, unshifted_center, aligned


def _metric_record(
    summary: Mapping[str, object],
    phase_mode: str,
    model: str,
    max_lag: int,
    support_samples: int,
) -> dict[str, object]:
    correlation = summary["per_record_pearson"]
    return {
        "model": model,
        "phase_mode": phase_mode,
        "max_lag_samples": max_lag,
        "support_samples": support_samples,
        "rmse": summary["rmse"],
        "mae": summary["mae"],
        "waveform_fd": summary["waveform_fd"],
        "per_record_pearson_mean": correlation["mean"],
        "per_record_pearson_median": correlation["median"],
    }


def _write_summary_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_per_window_csv(
    path: Path,
    original_rows: np.ndarray,
    per_window: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    sampling_rate: int,
) -> None:
    fields = [
        "test_row",
        "source_test_row_before_zero_filter",
        "model",
        "max_lag_samples",
        "max_lag_ms",
        "oracle_shift_samples",
        "oracle_shift_ms",
        "unshifted_fixed_support_rmse",
        "oracle_aligned_fixed_support_rmse",
        "rmse_change",
        "unshifted_fixed_support_mae",
        "oracle_aligned_fixed_support_mae",
        "unshifted_fixed_support_pearson_r",
        "oracle_aligned_fixed_support_pearson_r",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for max_lag, models in per_window.items():
            for model in MODEL_ORDER:
                values = models[model]
                for row in range(len(original_rows)):
                    shift = int(values["shifts"][row])
                    before_rmse = float(values["unshifted_rmse"][row])
                    after_rmse = float(values["aligned_rmse"][row])
                    writer.writerow(
                        {
                            "test_row": row,
                            "source_test_row_before_zero_filter": int(original_rows[row]),
                            "model": model,
                            "max_lag_samples": max_lag,
                            "max_lag_ms": max_lag * 1000.0 / sampling_rate,
                            "oracle_shift_samples": shift,
                            "oracle_shift_ms": shift * 1000.0 / sampling_rate,
                            "unshifted_fixed_support_rmse": before_rmse,
                            "oracle_aligned_fixed_support_rmse": after_rmse,
                            "rmse_change": after_rmse - before_rmse,
                            "unshifted_fixed_support_mae": values["unshifted_mae"][row],
                            "oracle_aligned_fixed_support_mae": values["aligned_mae"][row],
                            "unshifted_fixed_support_pearson_r": values["unshifted_pearson"][row],
                            "oracle_aligned_fixed_support_pearson_r": values["aligned_pearson"][row],
                        }
                    )


def _plot_comparison(path: Path, records: list[dict[str, object]], sampling_rate: int) -> None:
    metrics = ("rmse", "mae", "waveform_fd", "per_record_pearson_median")
    titles = ("RMSE", "MAE", "Waveform FD", "Median per-window Pearson")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    max_lags = sorted(
        {int(record["max_lag_samples"]) for record in records if int(record["max_lag_samples"]) > 0}
    )
    x_labels = [f"+/-{lag * 1000 / sampling_rate:.0f} ms" for lag in max_lags]
    x = np.arange(len(x_labels))
    for axis, metric, title in zip(axes.ravel(), metrics, titles):
        for model in MODEL_ORDER:
            unshifted = [
                next(
                    record[metric]
                    for record in records
                    if record["model"] == model
                    and record["phase_mode"] == "unshifted_fixed_support"
                    and record["max_lag_samples"] == lag
                )
                for lag in max_lags
            ]
            corrected = [
                next(
                    record[metric]
                    for record in records
                    if record["model"] == model
                    and record["phase_mode"] == "oracle_aligned_fixed_support"
                    and record["max_lag_samples"] == lag
                )
                for lag in max_lags
            ]
            axis.plot(
                x,
                unshifted,
                marker="o",
                linestyle="--",
                alpha=0.55,
                color=COLORS[model],
                label=f"{DISPLAY_NAMES[model]} unshifted",
            )
            axis.plot(
                x,
                corrected,
                marker="o",
                color=COLORS[model],
                label=f"{DISPLAY_NAMES[model]} oracle",
            )
        axis.set_title(title)
        axis.set_xticks(x, x_labels)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(ncol=2, fontsize=8)
    figure.suptitle(
        "MIMIC-AFib target-informed oracle phase sensitivity (matched fixed support)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_path = args.flow_predictions.resolve()
    rddm_path = args.rddm_predictions.resolve()
    targets, conditions, predictions, original_rows = _load_inputs(flow_path, rddm_path)

    records: list[dict[str, object]] = []
    raw_summaries = {}
    for model in MODEL_ORDER:
        summary, _ = _waveform_metrics(targets, predictions[model])
        raw_summaries[model] = summary
        records.append(_metric_record(summary, "raw_full_window", model, 0, 512))
    for baseline, values in (("zero", np.zeros_like(targets)), ("ppg_copy", conditions)):
        summary, _ = _waveform_metrics(targets, values)
        raw_summaries[baseline] = summary
        records.append(_metric_record(summary, "raw_full_window", baseline, 0, 512))

    sensitivity = {}
    fixed_support_baselines = {}
    per_window = {}
    aligned_artifact_names = []
    for max_lag in args.max_lags:
        lag_models = {}
        per_window[max_lag] = {}
        target_fixed = targets[:, :, max_lag : 512 - max_lag]
        condition_fixed = conditions[:, :, max_lag : 512 - max_lag]
        aligned_arrays = {"targets": target_fixed}
        fixed_support_baselines[str(max_lag)] = {}
        for baseline, values in (
            ("zero", np.zeros_like(target_fixed)),
            ("ppg_copy", condition_fixed),
        ):
            baseline_summary, _ = _waveform_metrics(target_fixed, values)
            fixed_support_baselines[str(max_lag)][baseline] = baseline_summary
            records.append(
                _metric_record(
                    baseline_summary,
                    "unshifted_fixed_support",
                    baseline,
                    max_lag,
                    target_fixed.shape[-1],
                )
            )
        for model in MODEL_ORDER:
            lag_summary, lag_values = _lag_diagnostic(
                targets, predictions[model], max_lag, args.sampling_rate
            )
            shifts = lag_values["best_lag_samples"].astype(np.int32)
            target_center, unshifted, aligned = _fixed_support_align(
                targets, predictions[model], shifts, max_lag
            )
            unshifted_summary, unshifted_per = _waveform_metrics(target_center, unshifted)
            aligned_summary, aligned_per = _waveform_metrics(target_center, aligned)
            records.extend(
                (
                    _metric_record(
                        unshifted_summary,
                        "unshifted_fixed_support",
                        model,
                        max_lag,
                        target_center.shape[-1],
                    ),
                    _metric_record(
                        aligned_summary,
                        "oracle_aligned_fixed_support",
                        model,
                        max_lag,
                        target_center.shape[-1],
                    ),
                )
            )
            lag_models[model] = {
                "lag_search": lag_summary,
                "shift_samples_mean": float(np.mean(shifts)),
                "shift_samples_median": float(np.median(shifts)),
                "absolute_shift_samples_median": float(np.median(np.abs(shifts))),
                "fraction_at_search_boundary": float(np.mean(np.abs(shifts) == max_lag)),
                "fraction_rmse_improved": float(
                    np.mean(aligned_per["rmse"] < unshifted_per["rmse"])
                ),
                "unshifted_fixed_support": unshifted_summary,
                "oracle_aligned_fixed_support": aligned_summary,
            }
            per_window[max_lag][model] = {
                "shifts": shifts,
                "unshifted_rmse": unshifted_per["rmse"],
                "aligned_rmse": aligned_per["rmse"],
                "unshifted_mae": unshifted_per["mae"],
                "aligned_mae": aligned_per["mae"],
                "unshifted_pearson": unshifted_per["pearson_r"],
                "aligned_pearson": aligned_per["pearson_r"],
            }
            aligned_arrays[f"{model}_oracle_shifts"] = shifts
            aligned_arrays[f"{model}_unshifted_predictions"] = unshifted
            aligned_arrays[f"{model}_oracle_aligned_predictions"] = aligned
        sensitivity[str(max_lag)] = lag_models
        artifact_name = f"oracle_aligned_predictions_maxlag_{max_lag}.npz"
        np.savez_compressed(output_dir / artifact_name, **aligned_arrays)
        aligned_artifact_names.append(artifact_name)

    _write_summary_csv(output_dir / "phase_corrected_summary.csv", records)
    _write_per_window_csv(
        output_dir / "per_window_phase_metrics.csv",
        original_rows,
        per_window,
        args.sampling_rate,
    )
    _plot_comparison(
        output_dir / "phase_corrected_comparison.png", records, args.sampling_rate
    )
    _json(
        output_dir / "phase_corrected_summary.json",
        {
            "raw_full_window": raw_summaries,
            "fixed_support_baselines": fixed_support_baselines,
            "oracle_phase_sensitivity": sensitivity,
            "interpretation_boundary": (
                "Each lag is selected using the generated waveform and its real ECG target. "
                "This is an oracle diagnostic, not a deployable preprocessing or inference method."
            ),
        },
    )

    artifact_names = [
        "phase_corrected_summary.csv",
        "per_window_phase_metrics.csv",
        "phase_corrected_summary.json",
        "phase_corrected_comparison.png",
        "phase_corrected_comparison.pdf",
        *aligned_artifact_names,
    ]
    upstream_path = args.upstream_preprocessing.resolve()
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "command": shlex.join(sys.argv),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "records": len(targets),
            "sampling_rate_hz": args.sampling_rate,
            "window_samples": targets.shape[-1],
            "max_lag_samples": list(args.max_lags),
            "max_lag_ms": [lag * 1000.0 / args.sampling_rate for lag in args.max_lags],
            "lag_objective": "maximize per-window Pearson over overlapping samples",
            "fixed_support_rule": (
                "target[max_lag:L-max_lag] versus generated[max_lag-shift:L-max_lag-shift]"
            ),
            "raw_metrics_remain_primary": True,
            "oracle_metrics_role": "phase_sensitivity_diagnostic_only",
            "targets_sha256": _array_sha256(targets),
            "conditions_sha256": _array_sha256(conditions),
        },
        "input_artifacts": {
            "flow_predictions": {"path": str(flow_path), "sha256": _sha256(flow_path)},
            "rddm_predictions": {"path": str(rddm_path), "sha256": _sha256(rddm_path)},
        },
        "source_alignment_audit": {
            "upstream_preprocessing_path": str(upstream_path),
            "upstream_preprocessing_sha256": _sha256(upstream_path),
            "finding": (
                "ECG and PPG from each WFDB record are independently resampled 125-to-128 Hz "
                "and sliced with identical start indices; no explicit lag estimation, peak "
                "matching, cross-correlation, or delay compensation is implemented."
            ),
            "training_alignment_label": "shared_clock_same_window_boundaries_without_delay_correction",
        },
        "execution": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "claim_boundary": (
            "Oracle target-informed correction cannot be used as the primary metric or claimed "
            "as the training/deployment alignment method. The four-second QC split also lacks "
            "subject IDs and recording continuity."
        ),
        "artifact_sha256": {name: _sha256(output_dir / name) for name in artifact_names},
        "outputs": [*artifact_names, "protocol.json"],
    }
    _json(output_dir / "protocol.json", protocol)
    print(json.dumps({"output_dir": str(output_dir), "status": "completed"}, sort_keys=True))
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow_predictions", type=Path, required=True)
    parser.add_argument("--rddm_predictions", type=Path, required=True)
    parser.add_argument("--upstream_preprocessing", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sampling_rate", type=int, default=128)
    parser.add_argument("--max_lags", type=_parse_max_lags, default=(16, 32, 64))
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
