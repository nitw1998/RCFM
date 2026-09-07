"""Create a multi-page visual audit of generated versus real ECG waveforms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_mimic_phase_corrected import _fixed_support_align
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic


MODELS = ("CFM", "RCFM-Pan", "RCFM-Pan+OT", "RDDM")
RAW_KEYS = {
    "CFM": "cfm_predictions",
    "RCFM-Pan": "rcfm_predictions",
    "RCFM-Pan+OT": "rcfm_ot_predictions",
    "RDDM": "rddm_predictions",
}
COLORS = {
    "CFM": "#3973AC",
    "RCFM-Pan": "#35975A",
    "RCFM-Pan+OT": "#9B59B6",
    "RDDM": "#D17A00",
}
QUANTILES = (("easier", 0.10), ("typical", 0.50), ("harder", 0.90))


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


def _load_mimic(flow_path: Path, rddm_path: Path):
    with np.load(flow_path, allow_pickle=False) as flow, np.load(
        rddm_path, allow_pickle=False
    ) as rddm:
        target = _as_waveforms(flow["targets"])
        condition = _as_waveforms(flow["conditions"], len(target))
        if not np.array_equal(target, rddm["targets"]) or not np.array_equal(
            condition, rddm["conditions"]
        ):
            raise ValueError("MIMIC flow and RDDM artifacts are not row-aligned")
        predictions = {
            model: _as_waveforms(
                flow[key] if model != "RDDM" else rddm[key], len(target)
            )
            for model, key in RAW_KEYS.items()
        }
    return target, condition, predictions


def _load_fourway(path: Path):
    with np.load(path, allow_pickle=False) as artifact:
        target = _as_waveforms(artifact["targets"])
        condition = _as_waveforms(artifact["conditions"], len(target))
        predictions = {
            model: _as_waveforms(artifact[key], len(target))
            for model, key in RAW_KEYS.items()
        }
    return target, condition, predictions


def _phase_arrays(target: np.ndarray, predictions: dict[str, np.ndarray], max_lag: int):
    center = target[:, :, max_lag:-max_lag]
    aligned: dict[str, np.ndarray] = {}
    shifts: dict[str, np.ndarray] = {}
    for model in MODELS:
        _, lag_values = _lag_diagnostic(target, predictions[model], max_lag, 128)
        shifts[model] = lag_values["best_lag_samples"].astype(np.int32)
        phase_target, _unshifted, phase_prediction = _fixed_support_align(
            target, predictions[model], shifts[model], max_lag
        )
        if not np.array_equal(phase_target, center):
            raise ValueError("phase targets differ across models")
        aligned[model] = np.asarray(phase_prediction, dtype=np.float32)
    return center, aligned, shifts


def _per_window_rmse(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    return np.sqrt(np.mean(difference * difference, axis=(1, 2)))


def _per_window_pearson(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    first = target[:, 0].astype(np.float64)
    second = prediction[:, 0].astype(np.float64)
    first -= first.mean(axis=1, keepdims=True)
    second -= second.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    values = np.full(len(first), np.nan, dtype=np.float64)
    usable = denominator > 0
    values[usable] = np.sum(first[usable] * second[usable], axis=1) / denominator[usable]
    return values


def _select_rows(target: np.ndarray, predictions: dict[str, np.ndarray]):
    consensus = np.mean(
        np.stack([_per_window_rmse(target, predictions[model]) for model in MODELS]),
        axis=0,
    )
    selected = []
    used: set[int] = set()
    for label, quantile in QUANTILES:
        value = float(np.quantile(consensus, quantile))
        order = np.argsort(np.abs(consensus - value), kind="stable")
        row = next(int(index) for index in order if int(index) not in used)
        used.add(row)
        selected.append((label, quantile, row, float(consensus[row])))
    return selected, consensus


def _limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([array.reshape(-1).astype(np.float64) for array in arrays])
    low, high = float(values.min()), float(values.max())
    padding = max((high - low) * 0.06, 0.02)
    return low - padding, high + padding


def _plot_page(
    pdf: PdfPages,
    dataset: str,
    selection: tuple[str, float, int, float],
    target: np.ndarray,
    condition: np.ndarray,
    predictions: dict[str, np.ndarray],
    phase_target: np.ndarray,
    phase_predictions: dict[str, np.ndarray],
    shifts: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    label, quantile, row, consensus_rmse = selection
    figure = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
    grid = figure.add_gridspec(5, 2, height_ratios=[0.75, 1, 1, 1, 1])
    condition_axis = figure.add_subplot(grid[0, :])
    raw_time = np.arange(512) / 128.0
    phase_time = np.arange(16, 496) / 128.0
    condition_axis.plot(raw_time, condition[row, 0], color="#666666", linewidth=0.9)
    condition_name = "PPG condition" if dataset != "mmECG" else "RCG condition"
    condition_axis.set_ylabel(condition_name, fontsize=8)
    condition_axis.set_xlim(raw_time[0], raw_time[-1])
    condition_axis.grid(True, alpha=0.18, linewidth=0.4)
    condition_axis.tick_params(labelsize=7)
    condition_axis.set_title(
        f"{dataset} — {label} case (raw consensus-RMSE quantile {quantile:.0%}, "
        f"dataset-local row {row}, consensus RMSE {consensus_rmse:.4f})",
        fontsize=11,
    )
    condition_axis.text(
        0.995, 0.92, "Panels below: black = real ECG; colored = generated ECG",
        transform=condition_axis.transAxes, ha="right", va="top", fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
    )

    raw_limits = _limits([target[row], *[predictions[model][row] for model in MODELS]])
    aligned_limits = _limits(
        [phase_target[row], *[phase_predictions[model][row] for model in MODELS]]
    )
    rows = []
    for model_index, model in enumerate(MODELS, start=1):
        raw_axis = figure.add_subplot(grid[model_index, 0])
        aligned_axis = figure.add_subplot(grid[model_index, 1])
        raw_generated = predictions[model][row, 0]
        aligned_generated = phase_predictions[model][row, 0]
        raw_reference = target[row, 0]
        aligned_reference = phase_target[row, 0]
        raw_rmse = float(_per_window_rmse(target[row : row + 1], predictions[model][row : row + 1])[0])
        raw_r = float(_per_window_pearson(target[row : row + 1], predictions[model][row : row + 1])[0])
        aligned_rmse = float(
            _per_window_rmse(
                phase_target[row : row + 1], phase_predictions[model][row : row + 1]
            )[0]
        )
        aligned_r = float(
            _per_window_pearson(
                phase_target[row : row + 1], phase_predictions[model][row : row + 1]
            )[0]
        )
        for axis, time_values, reference, generated, limits in (
            (raw_axis, raw_time, raw_reference, raw_generated, raw_limits),
            (aligned_axis, phase_time, aligned_reference, aligned_generated, aligned_limits),
        ):
            axis.plot(time_values, reference, color="#111111", linewidth=1.0, label="Real ECG")
            axis.plot(time_values, generated, color=COLORS[model], linewidth=0.9, alpha=0.9, label=model)
            axis.set_xlim(time_values[0], time_values[-1])
            axis.set_ylim(*limits)
            axis.grid(True, alpha=0.18, linewidth=0.4)
            axis.tick_params(labelsize=7)
        raw_axis.set_ylabel(model, fontsize=8, color=COLORS[model])
        raw_axis.text(
            0.995, 0.92, f"RMSE={raw_rmse:.3f}  r={raw_r:.3f}",
            transform=raw_axis.transAxes, ha="right", va="top", fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        )
        shift = int(shifts[model][row])
        aligned_axis.text(
            0.995, 0.92,
            f"RMSE={aligned_rmse:.3f}  r={aligned_r:.3f}  shift={shift:+d} ({shift / 128 * 1000:+.1f} ms)",
            transform=aligned_axis.transAxes, ha="right", va="top", fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        )
        if model_index == 1:
            raw_axis.set_title("Raw paired waveforms", fontsize=9)
            aligned_axis.set_title("Target-informed phase-corrected morphology (±16 samples)", fontsize=9)
        if model_index == len(MODELS):
            raw_axis.set_xlabel("Time (s)", fontsize=8)
            aligned_axis.set_xlabel("Original-window time (s); central 480 samples", fontsize=8)
        rows.append(
            {
                "dataset": dataset,
                "selection": label,
                "quantile": quantile,
                "dataset_row": row,
                "consensus_raw_rmse": consensus_rmse,
                "model": model,
                "raw_rmse": raw_rmse,
                "raw_pearson": raw_r,
                "oracle_shift_samples": shift,
                "oracle_aligned_rmse": aligned_rmse,
                "oracle_aligned_pearson": aligned_r,
            }
        )
    pdf.savefig(figure, dpi=200)
    plt.close(figure)
    return rows


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty directory: {output_dir}")
    if args.max_lag_samples != 16:
        raise ValueError("the frozen visual audit requires max_lag_samples=16")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "MIMIC-AFib flow": args.mimic_flow.resolve(),
        "MIMIC-AFib RDDM": args.mimic_rddm.resolve(),
        "WESAD": args.wesad.resolve(),
        "mmECG": args.mmecg.resolve(),
    }
    datasets = {
        "MIMIC-AFib": _load_mimic(sources["MIMIC-AFib flow"], sources["MIMIC-AFib RDDM"]),
        "WESAD": _load_fourway(sources["WESAD"]),
        "mmECG": _load_fourway(sources["mmECG"]),
    }
    pdf_path = output_dir / "ecg_generated_vs_real_visual_audit.pdf"
    selection_rows = []
    with PdfPages(pdf_path, metadata={"Title": "Generated versus real ECG visual audit"}) as pdf:
        for dataset, (target, condition, predictions) in datasets.items():
            phase_target, phase_predictions, shifts = _phase_arrays(
                target, predictions, args.max_lag_samples
            )
            selections, _scores = _select_rows(target, predictions)
            for selection in selections:
                selection_rows.extend(
                    _plot_page(
                        pdf, dataset, selection, target, condition, predictions,
                        phase_target, phase_predictions, shifts,
                    )
                )
    csv_path = output_dir / "selected_rows_and_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selection_rows[0].keys())
        writer.writeheader(); writer.writerows(selection_rows)
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "selection": "nearest rows to 10th, 50th, and 90th quantiles of mean raw per-window RMSE across four models",
        "datasets": list(datasets),
        "models": list(MODELS),
        "phase_correction": {
            "status": "target-informed visual morphology diagnostic only",
            "max_lag_samples": args.max_lag_samples,
            "sampling_rate_hz": 128,
            "fixed_support_samples": 480,
        },
        "privacy": "dataset-local row indices only; subject/source identifiers omitted",
        "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in sources.items()},
        "outputs": {
            pdf_path.name: _sha256(pdf_path),
            csv_path.name: _sha256(csv_path),
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }
    protocol_path = output_dir / "protocol.json"
    temporary = protocol_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, protocol_path)
    print(pdf_path)
    return pdf_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic_flow", type=Path, required=True)
    parser.add_argument("--mimic_rddm", type=Path, required=True)
    parser.add_argument("--wesad", type=Path, required=True)
    parser.add_argument("--mmecg", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
