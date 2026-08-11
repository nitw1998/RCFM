"""Summarize and plot the frozen raw PTB-XL four-model prediction artifact."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _sha256
from scripts.evaluate_ptbxl_fourway import MODEL_ORDER, TARGET_LEADS


LABELS = {"cfm": "CFM", "rcfm": "RCFM", "rcfm_ot": "RCFM-OT", "rddm": "RDDM-ECG"}
COLORS = {"cfm": "#2878b5", "rcfm": "#2f8f5b", "rcfm_ot": "#c43d4b", "rddm": "#d17a00"}


def _configure_ieee_style() -> None:
    """Use a Times-compatible serif face at IEEE double-column dimensions."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman", "Times"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "lines.linewidth": 0.8,
        }
    )


def _patient_means(values: np.ndarray, patient_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    patients, inverse = np.unique(patient_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64))
    counts = np.bincount(inverse)
    return patients, sums / counts


def _cluster_bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    patient_ids: np.ndarray,
    seed: int,
    replicates: int,
) -> dict[str, object]:
    if first.shape != second.shape or first.ndim != 1 or len(first) != len(patient_ids):
        raise ValueError("paired errors and patient IDs must align")
    patients, first_means = _patient_means(first, patient_ids)
    second_patients, second_means = _patient_means(second, patient_ids)
    if not np.array_equal(patients, second_patients) or replicates <= 0:
        raise ValueError("patient aggregation or bootstrap replicate count is invalid")
    difference = first_means - second_means
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(patients), size=(replicates, len(patients)))
    bootstrap = np.mean(difference[draws], axis=1)
    return {
        "difference_definition": "first_minus_second_after_mean_within_patient",
        "patients": int(len(patients)),
        "mean_difference": float(np.mean(difference)),
        "median_difference": float(np.median(difference)),
        "fraction_patients_favoring_second": float(np.mean(difference > 0)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "bootstrap_95_ci": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "inference_scope": "fixed_seed_fixed_checkpoint_test_patient_resampling_only",
    }


def _read_per_record(path: Path) -> dict[str, dict[str, np.ndarray]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {
        model: {
            metric: np.asarray([float(row[f"{model}_{metric}"]) for row in rows])
            for metric in ("rmse", "mae", "bias", "pearson_r")
        }
        for model in MODEL_ORDER
    }


def _plot_metrics(summary: dict[str, object], output: Path) -> list[Path]:
    models = list(MODEL_ORDER)
    _configure_ieee_style()
    figure, axes = plt.subplots(1, 4, figsize=(7.16, 2.15), constrained_layout=True)
    definitions = (
        ("rmse", "RMSE", None), ("mae", "MAE", None),
        ("waveform_fd_macro_lead", "Waveform FD\n(mean across leads)", "log"),
        ("per_record_pearson_median", "Median record Pearson", None),
    )
    for axis, (field, label, scale) in zip(axes, definitions):
        values = [summary["models"][model][field] for model in models]
        bars = axis.bar(np.arange(len(models)), values, color=[COLORS[model] for model in models], width=0.72)
        axis.set_xticks(np.arange(len(models)), [LABELS[model] for model in models], rotation=32, ha="right")
        axis.set_ylabel(label); axis.grid(axis="y", alpha=0.2)
        if scale:
            axis.set_yscale(scale)
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2.0, value),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=5.5,
            )
        if scale == "log":
            axis.set_ylim(min(values) * 0.75, max(values) * 1.45)
        else:
            axis.set_ylim(0.0, max(values) * 1.14)
    paths = []
    for suffix in ("png", "pdf"):
        path = output.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=600 if suffix == "png" else None)
        paths.append(path)
    plt.close(figure)
    return paths


def _plot_waveforms(
    targets: np.ndarray,
    predictions: dict[str, np.ndarray],
    per_record: dict[str, dict[str, np.ndarray]],
    record_ids: np.ndarray,
    output: Path,
) -> tuple[list[Path], int]:
    score = np.mean(np.stack([per_record[model]["rmse"] for model in MODEL_ORDER[:3]]), axis=0)
    order = np.argsort(score, kind="stable")
    row = int(order[len(order) // 2])
    lead_names = ("I", "aVF", "V1", "V3", "V5", "V6")
    lead_indices = [TARGET_LEADS.index(name) for name in lead_names]
    time = np.arange(targets.shape[-1]) / 128.0
    _configure_ieee_style()
    figure, axes = plt.subplots(
        len(lead_names), len(MODEL_ORDER), figsize=(7.16, 7.0),
        sharex=True, squeeze=False, constrained_layout=True,
    )
    for column, model in enumerate(MODEL_ORDER):
        for lead_row, (lead, lead_index) in enumerate(zip(lead_names, lead_indices)):
            axis = axes[lead_row, column]
            axis.plot(time, targets[row, lead_index], color="#111827", linewidth=0.9, label="Real")
            axis.plot(time, predictions[model][row, lead_index], color=COLORS[model], linewidth=0.8, alpha=0.9, label=LABELS[model])
            axis.grid(alpha=0.18)
            if column == 0:
                axis.set_ylabel(lead)
            if lead_row == 0:
                axis.set_title(f"{LABELS[model]} | RMSE={per_record[model]['rmse'][row]:.3f}")
            if lead_row == len(lead_names) - 1:
                axis.set_xlabel("Time (s)")
    axes[0, 0].legend(frameon=False, loc="upper right")
    figure.suptitle(
        f"PTB-XL fold 10: shared median-flow-error record {record_ids[row]} "
        "(raw synchronized pairs)",
        fontsize=8.5,
    )
    paths = []
    for suffix in ("png", "pdf"):
        path = output.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=600 if suffix == "png" else None)
        paths.append(path)
    plt.close(figure)
    return paths, row


def run(args: argparse.Namespace) -> Path:
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((input_dir / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "completed" or protocol["protocol"].get("phase_correction_applied") is not False:
        raise ValueError("analysis requires a completed unshifted PTB-XL artifact")
    summary = json.loads((input_dir / "waveform_summary.json").read_text(encoding="utf-8"))
    with np.load(input_dir / "paired_reference.npz", allow_pickle=False) as arrays:
        targets = arrays["targets"]
        record_ids = arrays["record_ids"]
        patient_ids = arrays["patient_ids"]
    predictions = {model: np.load(input_dir / f"{model}_predictions.npy", mmap_mode="r") for model in MODEL_ORDER}
    per_record = _read_per_record(input_dir / "per_record_metrics.csv")
    comparisons = {}
    pairs = tuple(itertools.combinations(MODEL_ORDER, 2))
    for pair_index, (first, second) in enumerate(pairs):
        comparisons[f"{first}_vs_{second}"] = {
            metric: _cluster_bootstrap_difference(
                per_record[first][metric], per_record[second][metric], patient_ids,
                args.bootstrap_seed + pair_index * 10 + metric_index, args.bootstrap_replicates,
            )
            for metric_index, metric in enumerate(("rmse", "mae"))
        }
    statistical_path = output_dir / "patient_cluster_comparisons.json"
    statistical_path.write_text(json.dumps({"schema_version": 1, "comparisons": comparisons}, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    outputs = [statistical_path, *_plot_metrics(summary, output_dir / "ptbxl_fourway_metrics")]
    waveform_paths, selected_row = _plot_waveforms(targets, predictions, per_record, record_ids, output_dir / "ptbxl_fourway_waveforms_median")
    outputs.extend(waveform_paths)
    analysis_protocol = {
        "schema_version": 1, "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv), "source_protocol_sha256": _sha256(input_dir / "protocol.json"),
        "source_waveform_summary_sha256": _sha256(input_dir / "waveform_summary.json"),
        "script_sha256": _sha256(Path(__file__)),
        "records": int(len(record_ids)), "patients": int(len(np.unique(patient_ids))),
        "selected_waveform_row": selected_row, "selected_record_id": str(record_ids[selected_row]),
        "primary_alignment": "raw synchronized same-record pairs; no phase correction",
        "bootstrap_scope": "patient-cluster resampling conditional on one fixed training seed",
        "figure_style": {
            "width_inches": 7.16,
            "font_family": "Liberation Serif (Times-compatible)",
            "pdf_fonttype": 42,
            "png_dpi": 600,
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (output_dir / "analysis_protocol.json").write_text(json.dumps(analysis_protocol, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_seed", type=int, default=2031)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    return parser


if __name__ == "__main__":
    output = run(build_argparser().parse_args())
    print(f"PTB-XL four-way analysis saved to {output}")
