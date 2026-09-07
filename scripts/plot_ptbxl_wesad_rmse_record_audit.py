"""Plot exact best, median-rank, and worst records by raw paired RMSE.

The audit deliberately reports dataset-local row numbers rather than subject or
source-record identifiers.  PTB-XL is ranked by its single CFM prediction.  WESAD
is ranked once by the mean RCFM/RCFM-OT RMSE so both models are inspected on the
same records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


SAMPLE_RATE = 128.0
SELECTIONS = ("best", "mid", "worst")
COLORS = {
    "CFM": "#3973AC",
    "RCFM": "#35975A",
    "RCFM-OT": "#9B59B6",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_waveforms(values: np.ndarray, expected_rows: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, None, :]
    if array.ndim != 3 or array.shape[1:] != (1, 512):
        raise ValueError(f"expected (rows,1,512), got {array.shape}")
    if expected_rows is not None and len(array) != expected_rows:
        raise ValueError("waveform row counts differ")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("waveform artifact contains NaN or Inf")
    return array


def _per_record_rmse(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction shapes differ")
    difference = prediction.astype(np.float64) - reference.astype(np.float64)
    return np.sqrt(np.mean(difference * difference, axis=(1, 2)))


def _per_record_pearson(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction shapes differ")
    first = reference[:, 0].astype(np.float64)
    second = prediction[:, 0].astype(np.float64)
    first -= first.mean(axis=1, keepdims=True)
    second -= second.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    result = np.full(len(first), np.nan, dtype=np.float64)
    usable = denominator > 0
    result[usable] = np.sum(first[usable] * second[usable], axis=1) / denominator[usable]
    return result


def _select_extremes(scores: np.ndarray) -> list[dict[str, float | int | str]]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("scores must be one-dimensional with at least three rows")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("selection scores contain NaN or Inf")
    order = np.argsort(values, kind="stable")
    ranks = (0, len(values) // 2, len(values) - 1)
    denominator = max(len(values) - 1, 1)
    return [
        {
            "selection": label,
            "dataset_row": int(order[rank]),
            "rank_zero_based": int(rank),
            "percentile": float(100.0 * rank / denominator),
            "selection_score": float(values[order[rank]]),
        }
        for label, rank in zip(SELECTIONS, ranks)
    ]


def _limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(array).reshape(-1) for array in arrays])
    low, high = float(values.min()), float(values.max())
    padding = max(0.06 * (high - low), 0.02)
    return low - padding, high + padding


def _decorate(axis: plt.Axes, time: np.ndarray) -> None:
    axis.set_xlim(time[0], time[-1])
    axis.grid(True, alpha=0.18, linewidth=0.45)
    axis.tick_params(labelsize=7)


def _plot_ptbxl(
    target: np.ndarray,
    condition: np.ndarray,
    prediction: np.ndarray,
    selections: list[dict[str, float | int | str]],
    rmse: np.ndarray,
    pearson: np.ndarray,
) -> plt.Figure:
    time = np.arange(512) / SAMPLE_RATE
    figure, axes = plt.subplots(3, 3, figsize=(15, 8.2), constrained_layout=True)
    for column, selected in enumerate(selections):
        row = int(selected["dataset_row"])
        limits = _limits([target[row], prediction[row]])
        axes[0, column].plot(time, condition[row, 0], color="#666666", linewidth=0.9)
        axes[0, column].set_ylabel("Lead III (condition)" if column == 0 else "")
        axes[0, column].set_title(
            f"{str(selected['selection']).upper()} | row {row} | "
            f"rank {int(selected['rank_zero_based']) + 1}/{len(target)}\n"
            f"RMSE={rmse[row]:.4f}, Pearson r={pearson[row]:.4f}",
            fontsize=10,
        )
        axes[1, column].plot(time, target[row, 0], color="#111111", linewidth=1.05, label="Real V5")
        axes[1, column].plot(time, prediction[row, 0], color=COLORS["CFM"], linewidth=0.9, label="CFM V5")
        axes[1, column].set_ylim(*limits)
        axes[1, column].set_ylabel("V5 overlay" if column == 0 else "")
        axes[2, column].axhline(0.0, color="#777777", linewidth=0.6)
        axes[2, column].plot(time, prediction[row, 0] - target[row, 0], color="#C44E52", linewidth=0.8)
        axes[2, column].set_ylabel("Generated - real" if column == 0 else "")
        axes[2, column].set_xlabel("Time (s)")
        for axis in axes[:, column]:
            _decorate(axis, time)
    axes[1, 0].legend(loc="upper right", fontsize=7, framealpha=0.8)
    figure.suptitle(
        "PTB-XL legacy CFM: Lead III → V5 | exact per-record RMSE ranking",
        fontsize=14,
    )
    return figure


def _plot_wesad(
    target: np.ndarray,
    condition: np.ndarray,
    predictions: dict[str, np.ndarray],
    selections: list[dict[str, float | int | str]],
    model_rmse: dict[str, np.ndarray],
    model_pearson: dict[str, np.ndarray],
) -> plt.Figure:
    time = np.arange(512) / SAMPLE_RATE
    figure, axes = plt.subplots(4, 3, figsize=(15, 10.2), constrained_layout=True)
    for column, selected in enumerate(selections):
        row = int(selected["dataset_row"])
        limits = _limits([target[row], *[values[row] for values in predictions.values()]])
        axes[0, column].plot(time, condition[row, 0], color="#666666", linewidth=0.9)
        axes[0, column].set_ylabel("BVP (condition)" if column == 0 else "")
        axes[0, column].set_title(
            f"{str(selected['selection']).upper()} | row {row} | "
            f"rank {int(selected['rank_zero_based']) + 1}/{len(target)}\n"
            f"mean-model RMSE={float(selected['selection_score']):.4f}",
            fontsize=10,
        )
        for axis_row, model in enumerate(("RCFM", "RCFM-OT"), start=1):
            axis = axes[axis_row, column]
            axis.plot(time, target[row, 0], color="#111111", linewidth=1.05, label="Real ECG")
            axis.plot(time, predictions[model][row, 0], color=COLORS[model], linewidth=0.9, label=model)
            axis.set_ylim(*limits)
            axis.set_ylabel(f"{model} overlay" if column == 0 else "")
            axis.text(
                0.01,
                0.94,
                f"RMSE={model_rmse[model][row]:.4f}, r={model_pearson[model][row]:.4f}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
            )
        axes[3, column].axhline(0.0, color="#777777", linewidth=0.6)
        for model in ("RCFM", "RCFM-OT"):
            axes[3, column].plot(
                time,
                predictions[model][row, 0] - target[row, 0],
                color=COLORS[model],
                linewidth=0.8,
                label=model,
            )
        axes[3, column].set_ylabel("Generated - real" if column == 0 else "")
        axes[3, column].set_xlabel("Time (s)")
        for axis in axes[:, column]:
            _decorate(axis, time)
    axes[1, 0].legend(loc="upper right", fontsize=7, framealpha=0.8)
    axes[2, 0].legend(loc="upper right", fontsize=7, framealpha=0.8)
    axes[3, 0].legend(loc="upper right", fontsize=7, framealpha=0.8)
    figure.suptitle(
        "WESAD fixed-lag-v2: shared records ranked by mean RCFM/RCFM-OT per-record RMSE",
        fontsize=14,
    )
    return figure


def _plot_distributions(
    ptb_rmse: np.ndarray,
    ptb_selections: list[dict[str, float | int | str]],
    wesad_rmse: dict[str, np.ndarray],
    wesad_score: np.ndarray,
    wesad_selections: list[dict[str, float | int | str]],
) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    axes[0].hist(ptb_rmse, bins=60, color=COLORS["CFM"], alpha=0.78)
    axes[0].set_title(f"PTB-XL CFM per-record RMSE (n={len(ptb_rmse)})")
    axes[1].hist(wesad_rmse["RCFM"], bins=60, histtype="step", linewidth=1.2, color=COLORS["RCFM"], label="RCFM")
    axes[1].hist(wesad_rmse["RCFM-OT"], bins=60, histtype="step", linewidth=1.2, color=COLORS["RCFM-OT"], label="RCFM-OT")
    axes[1].hist(wesad_score, bins=60, color="#888888", alpha=0.28, label="mean-model selection score")
    axes[1].set_title(f"WESAD per-record RMSE (n={len(wesad_score)})")
    line_colors = {"best": "#2CA02C", "mid": "#FF8C00", "worst": "#D62728"}
    for axis, selections in ((axes[0], ptb_selections), (axes[1], wesad_selections)):
        for selected in selections:
            axis.axvline(
                float(selected["selection_score"]),
                color=line_colors[str(selected["selection"])],
                linestyle="--",
                linewidth=1.3,
                label=f"{selected['selection']} row {selected['dataset_row']}",
            )
        axis.set_xlabel("RMSE (normalized units)")
        axis.set_ylabel("Records")
        axis.grid(True, alpha=0.18, linewidth=0.45)
        axis.legend(fontsize=8)
    figure.suptitle("RMSE distributions and exact selected ranks", fontsize=14)
    return figure


def _write_csv(
    path: Path,
    ptb_selections: list[dict[str, float | int | str]],
    ptb_rmse: np.ndarray,
    ptb_pearson: np.ndarray,
    wesad_selections: list[dict[str, float | int | str]],
    wesad_rmse: dict[str, np.ndarray],
    wesad_pearson: dict[str, np.ndarray],
) -> None:
    fields = (
        "dataset",
        "selection",
        "dataset_row",
        "rank_one_based",
        "percentile",
        "selection_score_definition",
        "selection_score",
        "model",
        "rmse",
        "pearson",
    )
    rows: list[dict[str, object]] = []
    for selected in ptb_selections:
        row = int(selected["dataset_row"])
        rows.append(
            {
                "dataset": "PTB-XL",
                "selection": selected["selection"],
                "dataset_row": row,
                "rank_one_based": int(selected["rank_zero_based"]) + 1,
                "percentile": selected["percentile"],
                "selection_score_definition": "CFM per-record raw paired RMSE",
                "selection_score": selected["selection_score"],
                "model": "CFM",
                "rmse": ptb_rmse[row],
                "pearson": ptb_pearson[row],
            }
        )
    for selected in wesad_selections:
        row = int(selected["dataset_row"])
        for model in ("RCFM", "RCFM-OT"):
            rows.append(
                {
                    "dataset": "WESAD",
                    "selection": selected["selection"],
                    "dataset_row": row,
                    "rank_one_based": int(selected["rank_zero_based"]) + 1,
                    "percentile": selected["percentile"],
                    "selection_score_definition": "mean RCFM/RCFM-OT per-record raw paired RMSE",
                    "selection_score": selected["selection_score"],
                    "model": model,
                    "rmse": wesad_rmse[model][row],
                    "pearson": wesad_pearson[model][row],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptbxl_dir", type=Path, required=True)
    parser.add_argument("--wesad_raw", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    ptb_reference_path = args.ptbxl_dir / "paired_reference.npz"
    ptb_prediction_path = args.ptbxl_dir / "predictions_epoch_999.npy"
    with np.load(ptb_reference_path, allow_pickle=False) as artifact:
        ptb_target = _as_waveforms(artifact["targets"])
        ptb_condition = _as_waveforms(artifact["conditions"], len(ptb_target))
    ptb_prediction = _as_waveforms(np.load(ptb_prediction_path, allow_pickle=False), len(ptb_target))

    with np.load(args.wesad_raw, allow_pickle=False) as artifact:
        wesad_target = _as_waveforms(artifact["targets"])
        wesad_condition = _as_waveforms(artifact["conditions"], len(wesad_target))
        wesad_predictions = {
            "RCFM": _as_waveforms(artifact["rcfm_predictions"], len(wesad_target)),
            "RCFM-OT": _as_waveforms(artifact["rcfm_ot_predictions"], len(wesad_target)),
        }

    ptb_rmse = _per_record_rmse(ptb_target, ptb_prediction)
    ptb_pearson = _per_record_pearson(ptb_target, ptb_prediction)
    ptb_selections = _select_extremes(ptb_rmse)
    wesad_rmse = {model: _per_record_rmse(wesad_target, prediction) for model, prediction in wesad_predictions.items()}
    wesad_pearson = {model: _per_record_pearson(wesad_target, prediction) for model, prediction in wesad_predictions.items()}
    wesad_score = np.mean(np.stack([wesad_rmse["RCFM"], wesad_rmse["RCFM-OT"]]), axis=0)
    wesad_selections = _select_extremes(wesad_score)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "rmse_distributions": _plot_distributions(ptb_rmse, ptb_selections, wesad_rmse, wesad_score, wesad_selections),
        "ptbxl_best_mid_worst": _plot_ptbxl(ptb_target, ptb_condition, ptb_prediction, ptb_selections, ptb_rmse, ptb_pearson),
        "wesad_best_mid_worst": _plot_wesad(wesad_target, wesad_condition, wesad_predictions, wesad_selections, wesad_rmse, wesad_pearson),
    }
    pdf_path = args.output_dir / "ptbxl_wesad_best_mid_worst_visual_audit.pdf"
    with PdfPages(pdf_path) as pdf:
        for name, figure in figures.items():
            figure.savefig(args.output_dir / f"{name}.png", dpi=220)
            pdf.savefig(figure, dpi=220)
            plt.close(figure)

    csv_path = args.output_dir / "selected_records_and_metrics.csv"
    _write_csv(csv_path, ptb_selections, ptb_rmse, ptb_pearson, wesad_selections, wesad_rmse, wesad_pearson)
    output_files = [pdf_path, csv_path, *[args.output_dir / f"{name}.png" for name in figures]]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "manual waveform inspection of exact best, median-rank, and worst records by RMSE",
        "privacy_boundary": "dataset-local rows only; subject and source-record identifiers omitted",
        "ptbxl": {
            "task": "legacy CFM Lead III to V5, official fold 10, seed-42 training, seed-31 inference",
            "selection": "stable ascending CFM raw paired per-record RMSE; ranks 0, floor(n/2), n-1",
            "rows": ptb_selections,
        },
        "wesad": {
            "task": "fixed-lag-v2 RCFM and RCFM-OT, shared seed-31 held-out predictions",
            "selection": "stable ascending mean of RCFM and RCFM-OT raw paired per-record RMSE; ranks 0, floor(n/2), n-1",
            "rows": wesad_selections,
        },
        "inputs": {
            str(ptb_reference_path): _sha256(ptb_reference_path),
            str(ptb_prediction_path): _sha256(ptb_prediction_path),
            str(args.wesad_raw): _sha256(args.wesad_raw),
        },
        "outputs": {path.name: _sha256(path) for path in output_files},
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "ptbxl": ptb_selections, "wesad": wesad_selections}, indent=2))


if __name__ == "__main__":
    main()
