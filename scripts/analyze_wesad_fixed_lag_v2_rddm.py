#!/usr/bin/env python3
"""Summarize and visualize the completed WESAD fixed-lag-v2 RDDM result."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_wesad_fixed_lag_v2_pair import subject_waveform_metrics
from scripts.plot_ptbxl_wesad_rmse_record_audit import (
    _limits,
    _per_record_pearson,
    _per_record_rmse,
    _select_extremes,
)


CLINICAL_PARAMETERS = (
    "heart_rate_bpm", "rr_ms", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms",
)


def _pairwise_clinical(path: Path) -> dict[str, dict[str, float | int]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    output: dict[str, dict[str, float | int]] = {}
    for parameter in CLINICAL_PARAMETERS:
        pairs = []
        for row in rows:
            if row["model"] != "rddm" or row["phase_mode"] != "unshifted":
                continue
            try:
                pairs.append((float(row[f"real_{parameter}"]), float(row[f"generated_{parameter}"])))
            except (TypeError, ValueError):
                continue
        values = np.asarray(pairs, dtype=np.float64)
        error = values[:, 1] - values[:, 0]
        output[parameter] = {
            "n": int(len(values)),
            "reference_mean": float(values[:, 0].mean()),
            "generated_mean": float(values[:, 1].mean()),
            "bias": float(error.mean()),
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error * error))),
            "pearson_r": float(np.corrcoef(values[:, 0], values[:, 1])[0, 1]),
        }
    return output


def _metrics(path: Path, model: str) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))["raw_full_window"][model]


def _write_selected(
    path: Path,
    selected: list[dict[str, float | int | str]],
    rmse: np.ndarray,
    pearson: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("selection", "dataset_row", "rank_one_based", "percentile", "rmse", "pearson"),
        )
        writer.writeheader()
        for item in selected:
            row = int(item["dataset_row"])
            writer.writerow(
                {
                    "selection": item["selection"],
                    "dataset_row": row,
                    "rank_one_based": int(item["rank_zero_based"]) + 1,
                    "percentile": item["percentile"],
                    "rmse": rmse[row],
                    "pearson": pearson[row],
                }
            )


def _plot_cases(
    target: np.ndarray,
    condition: np.ndarray,
    prediction: np.ndarray,
    selected: list[dict[str, float | int | str]],
    rmse: np.ndarray,
    pearson: np.ndarray,
) -> plt.Figure:
    time = np.arange(512) / 128.0
    figure, axes = plt.subplots(3, 3, figsize=(15, 8.2), constrained_layout=True)
    for column, item in enumerate(selected):
        row = int(item["dataset_row"])
        axes[0, column].plot(time, condition[row, 0], color="#666666", linewidth=0.9)
        axes[0, column].set_title(
            f"{str(item['selection']).upper()} | row {row} | "
            f"rank {int(item['rank_zero_based']) + 1}/{len(target)}\n"
            f"RMSE={rmse[row]:.4f}, Pearson r={pearson[row]:.4f}",
            fontsize=10,
        )
        axes[1, column].plot(time, target[row, 0], color="#111111", linewidth=1.0, label="Real ECG")
        axes[1, column].plot(time, prediction[row, 0], color="#D17A00", linewidth=0.9, label="RDDM-PPG")
        axes[1, column].set_ylim(*_limits([target[row], prediction[row]]))
        axes[2, column].axhline(0, color="#777777", linewidth=0.6)
        axes[2, column].plot(time, prediction[row, 0] - target[row, 0], color="#C44E52", linewidth=0.8)
        axes[2, column].set_xlabel("Time (s)")
        for axis in axes[:, column]:
            axis.set_xlim(time[0], time[-1]); axis.grid(True, alpha=0.18, linewidth=0.45)
            axis.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("BVP condition")
    axes[1, 0].set_ylabel("ECG overlay")
    axes[2, 0].set_ylabel("Generated - real")
    axes[1, 0].legend(fontsize=7, loc="upper right")
    figure.suptitle("WESAD fixed-lag-v2 RDDM: exact per-record RMSE ranking", fontsize=14)
    return figure


def _plot_distribution(rmse: np.ndarray, selected: list[dict[str, float | int | str]]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    axis.hist(rmse, bins=60, color="#D17A00", alpha=0.78)
    colors = {"best": "#2CA02C", "mid": "#FF8C00", "worst": "#D62728"}
    for item in selected:
        axis.axvline(
            float(item["selection_score"]), color=colors[str(item["selection"])],
            linestyle="--", linewidth=1.3,
            label=f"{item['selection']} row {item['dataset_row']}",
        )
    axis.set(title=f"WESAD RDDM per-record RMSE (n={len(rmse)})", xlabel="RMSE", ylabel="Records")
    axis.grid(True, alpha=0.18); axis.legend()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rddm_dir", type=Path, required=True)
    parser.add_argument("--rcfm_dir", type=Path, required=True)
    parser.add_argument("--old_rddm_dir", type=Path, required=True)
    parser.add_argument("--clinical_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    with np.load(args.rddm_dir / "raw_predictions.npz", allow_pickle=False) as artifact:
        target = np.asarray(artifact["targets"], dtype=np.float32)
        condition = np.asarray(artifact["conditions"], dtype=np.float32)
        subjects = np.asarray(artifact["subject_ids"]).astype(str)
        prediction = np.asarray(artifact["rddm_predictions"], dtype=np.float32)
    with np.load(args.rcfm_dir / "raw_predictions.npz", allow_pickle=False) as artifact:
        if not np.array_equal(target, artifact["targets"]):
            raise ValueError("RDDM and RCFM fixed-lag-v2 targets are not identical")
    with np.load(args.old_rddm_dir / "raw_predictions.npz", allow_pickle=False) as artifact:
        if not np.array_equal(target, artifact["targets"]):
            raise ValueError("v1 and v2 target rows are not identical")

    rmse = _per_record_rmse(target, prediction)
    pearson = _per_record_pearson(target, prediction)
    selected = _select_extremes(rmse)
    _write_selected(args.output_dir / "selected_records.csv", selected, rmse, pearson)

    subject_rows = []
    for subject in sorted(set(subjects)):
        indices = np.flatnonzero(subjects == subject)
        values = subject_waveform_metrics(target[indices], prediction[indices])
        subject_rows.append({"subject": subject, "windows": len(indices), **values})
    with (args.output_dir / "per_subject_waveform.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=subject_rows[0].keys())
        writer.writeheader(); writer.writerows(subject_rows)

    figures = {
        "rddm_best_mid_worst": _plot_cases(target, condition, prediction, selected, rmse, pearson),
        "rddm_record_rmse_distribution": _plot_distribution(rmse, selected),
    }
    for name, figure in figures.items():
        figure.savefig(args.output_dir / f"{name}.png", dpi=220)
        figure.savefig(args.output_dir / f"{name}.pdf")
        plt.close(figure)

    current = _metrics(args.rddm_dir / "waveform_summary.json", "rddm")
    old = _metrics(args.old_rddm_dir / "waveform_summary.json", "rddm")
    rcfm = _metrics(args.rcfm_dir / "waveform_summary.json", "rcfm")
    rcfm_ot = _metrics(args.rcfm_dir / "waveform_summary.json", "rcfm_ot")
    distribution = {
        "reference_mean": float(target.mean()),
        "generated_mean": float(prediction.mean()),
        "reference_sd": float(target.std()),
        "generated_sd": float(prediction.std()),
        "mean_bias": float((prediction - target).mean()),
    }
    summary = {
        "schema_version": 1,
        "selection": selected,
        "current_rddm_v2": current,
        "comparators": {
            "rcfm_v2": rcfm,
            "rcfm_ot_v2": rcfm_ot,
            "rddm_v1_same_seed_different_alignment_protocol": old,
        },
        "relative_change_percent": {
            "v2_rddm_minus_v1_rddm": {
                key: 100.0 * (float(current[key]) / float(old[key]) - 1.0)
                for key in ("rmse", "mae", "waveform_fd")
            },
            "v2_rddm_minus_v2_rcfm": {
                key: 100.0 * (float(current[key]) / float(rcfm[key]) - 1.0)
                for key in ("rmse", "mae", "waveform_fd")
            },
            "v2_rddm_minus_v2_rcfm_ot": {
                key: 100.0 * (float(current[key]) / float(rcfm_ot[key]) - 1.0)
                for key in ("rmse", "mae", "waveform_fd")
            },
        },
        "distribution": distribution,
        "raw_pairwise_clinical": _pairwise_clinical(args.clinical_dir / "per_window_ecg_parameters.csv"),
        "clinical_pairing_note": (
            "Raw pairwise rows are primary here; the clinical package's phase-comparison table "
            "uses the stricter real/raw/oracle common triplet."
        ),
        "claim_boundary": (
            "Single training seed, three held-out subjects, normalized amplitudes, and a v2 "
            "sensitivity alignment protocol; v1 comparison is descriptive and not exchangeable."
        ),
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
