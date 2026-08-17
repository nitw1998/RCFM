"""Evaluate raw CPSC2018 multi-lead ECG parameters and make IEEE-style figures."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import sys
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_fourway import MODEL_ORDER, TARGET_LEADS
from scripts.evaluate_cpsc_zscore_paired import _sha256
from src.rcfm.metrics.bland_altman import bland_altman, paired_correlation
from src.rcfm.metrics.clinical import delineate_ecg, measure_ecg_parameters


PARAMETERS = (
    "heart_rate_bpm", "rr_ms", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms",
    "p_amplitude", "r_amplitude", "qrs_peak_to_peak_amplitude", "t_amplitude", "st_deviation",
)
INTERVAL_PARAMETERS = ("rr_ms", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms")
AMPLITUDE_PARAMETERS = (
    "p_amplitude", "r_amplitude", "qrs_peak_to_peak_amplitude", "t_amplitude", "st_deviation"
)
PARAMETER_LABELS = {
    "heart_rate_bpm": "HR", "rr_ms": "RR", "pr_ms": "PR", "qrs_ms": "QRS",
    "qt_ms": "QT", "qtc_ms": "QTc", "p_amplitude": "P", "r_amplitude": "R",
    "qrs_peak_to_peak_amplitude": "QRS p-p", "t_amplitude": "T", "st_deviation": "ST",
}
MODEL_LABELS = {"cfm": "CFM", "rcfm": "RCFM", "rcfm_ot": "RCFM-OT", "rddm": "RDDM-ECG"}
MODEL_COLORS = {"cfm": "#2878b5", "rcfm": "#2f8f5b", "rcfm_ot": "#c43d4b", "rddm": "#d17a00"}
MODEL_LABELS["cfm_ot"] = "CFM+OT"
MODEL_COLORS["cfm_ot"] = "#76a5d5"
MODEL_LABELS["direct_cnn"] = "DirectCNN"
MODEL_COLORS["direct_cnn"] = "#6f7782"


def _configure_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman", "Times"],
        "font.size": 7.0, "axes.titlesize": 8.0, "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
    })


def _record_measurement(signal: np.ndarray, sampling_rate: float) -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Too few peaks detected.*")
        delineation = delineate_ecg(signal, sampling_rate=sampling_rate, method="dwt")
    if not delineation.success:
        return {"success": False, "failure_reason": delineation.failure_reason, "summary": {},
                "r_peak_count": 0, "n_rr": 0, "hrv_status": "not_measured_delineation_failure"}
    measurement = measure_ecg_parameters(
        signal, sampling_rate=sampling_rate, fiducials=delineation.fiducials,
        amplitude_unit="normalized", inverse_transformed=False, allow_normalized_amplitudes=True,
        qtc_formula="fridericia", st_offset_ms=60.0, continuous=False,
        minimum_hrv_duration_seconds=30.0, p_wave_applicable=True,
    )
    summary = {name: float(np.mean(values)) for name, values in measurement["parameters"].items() if values}
    if summary.get("rr_ms", 0) > 0:
        summary["heart_rate_bpm"] = 60000.0 / summary["rr_ms"]
    return {
        "success": True, "failure_reason": None, "summary": summary,
        "r_peak_count": int(len(delineation.fiducials.r_peaks)),
        "n_rr": int(measurement["hrv"]["n_rr"]),
        "hrv_status": measurement["hrv"]["status"],
        "p_wave_status": measurement["p_wave_status"],
        "amplitude_status": measurement["amplitude_status"],
    }


def _measure_task(task: tuple[np.ndarray, float]) -> dict[str, object]:
    return _record_measurement(*task)


def _measure_records(signals: np.ndarray, sampling_rate: float,
                     executor: ProcessPoolExecutor | None) -> list[dict[str, object]]:
    tasks = ((np.asarray(signal), sampling_rate) for signal in signals)
    if executor is None:
        return [_measure_task(task) for task in tasks]
    return list(executor.map(_measure_task, tasks, chunksize=16))


def _record_pairs(real: list[dict[str, object]], generated: list[dict[str, object]],
                  parameter: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [
        (index, real_item["summary"].get(parameter), generated_item["summary"].get(parameter))
        for index, (real_item, generated_item) in enumerate(zip(real, generated))
    ]
    rows = [(index, reference, prediction) for index, reference, prediction in rows
            if reference is not None and prediction is not None
            and np.isfinite(reference) and np.isfinite(prediction)]
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)
    return (np.asarray([row[0] for row in rows], dtype=np.int64),
            np.asarray([row[1] for row in rows], dtype=np.float64),
            np.asarray([row[2] for row in rows], dtype=np.float64))


def _agreement_row(model: str, lead: str, parameter: str,
                   real: np.ndarray, generated: np.ndarray) -> dict[str, object]:
    unit = "bpm" if parameter == "heart_rate_bpm" else ("ms" if parameter in INTERVAL_PARAMETERS else "normalized")
    base = {"model": model, "lead": lead, "parameter": parameter, "unit": unit,
            "n_records": int(len(real)), "difference_definition": "generated_minus_real"}
    if len(real) < 2:
        return {**base, "status": "insufficient_record_pairs"}
    error = generated - real
    correlation = paired_correlation(real, generated)
    agreement = bland_altman(real, generated)
    return {
        **base, "status": "ok", "real_mean": float(np.mean(real)),
        "generated_mean": float(np.mean(generated)), "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))), "pearson_r": correlation["r"],
        "pearson_status": correlation["status"], "ba_bias": agreement["bias"],
        "ba_difference_sd": agreement["difference_sd"], "ba_lower": agreement["lower_limit"],
        "ba_upper": agreement["upper_limit"],
        "ba_loa_width": agreement["upper_limit"] - agreement["lower_limit"],
    }


def _macro_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for model in MODEL_ORDER:
        for parameter in PARAMETERS:
            selected = [row for row in rows if row["model"] == model and row["parameter"] == parameter
                        and row["status"] == "ok"]
            base = {"model": model, "parameter": parameter,
                    "unit": (selected[0]["unit"] if selected else ""), "usable_leads": len(selected)}
            if not selected:
                output.append({**base, "status": "insufficient_leads"})
                continue
            finite_r = [row["pearson_r"] for row in selected if row["pearson_r"] is not None and np.isfinite(row["pearson_r"])]
            fisher = float(np.tanh(np.mean(np.arctanh(np.clip(finite_r, -0.999999, 0.999999))))) if finite_r else None
            output.append({
                **base, "status": "ok", "n_records_min": min(row["n_records"] for row in selected),
                "n_records_max": max(row["n_records"] for row in selected),
                "mae": float(np.mean([row["mae"] for row in selected])),
                "rmse": float(np.mean([row["rmse"] for row in selected])),
                "pearson_r": fisher, "pearson_aggregation": "unweighted_Fisher_z_mean_across_leads",
                "ba_bias": float(np.mean([row["ba_bias"] for row in selected])),
                "ba_loa_width": float(np.mean([row["ba_loa_width"] for row in selected])),
                "lead_aggregation": "unweighted_macro_mean",
            })
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _save_figure(figure: plt.Figure, stem: Path) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=600 if suffix == "png" else None)
        paths.append(path)
    plt.close(figure)
    return paths


def _annotated_heatmap(axis: plt.Axes, values: np.ndarray, parameters: tuple[str, ...],
                       title: str, fmt: str, cmap: str, vmin=None, vmax=None) -> None:
    image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(parameters)), [PARAMETER_LABELS[item] for item in parameters])
    axis.set_yticks(np.arange(len(MODEL_ORDER)), [MODEL_LABELS[item] for item in MODEL_ORDER])
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                axis.text(column, row, format(values[row, column], fmt), ha="center", va="center", fontsize=5.5)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.03)


def _matrix(macro: Mapping[tuple[str, str], Mapping[str, object]], parameters: tuple[str, ...], field: str) -> np.ndarray:
    return np.asarray([[macro.get((model, parameter), {}).get(field, np.nan) for parameter in parameters]
                       for model in MODEL_ORDER], dtype=np.float64)


def _plot_summary(waveform: Mapping[str, object], macro_rows: list[dict[str, object]], output: Path) -> list[Path]:
    _configure_ieee_style()
    macro = {(row["model"], row["parameter"]): row for row in macro_rows if row["status"] == "ok"}
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), constrained_layout=True)
    correlations = [waveform["models"][model]["per_record_pearson_median"] for model in MODEL_ORDER]
    loa = [waveform["models"][model]["pointwise_bland_altman_descriptive_only"]["upper_limit"]
           - waveform["models"][model]["pointwise_bland_altman_descriptive_only"]["lower_limit"] for model in MODEL_ORDER]
    for axis, values, title, ylim in (
        (axes[0], correlations, "Waveform correlation\nmedian record Pearson", (0, 1)),
        (axes[1], loa, "Waveform Bland-Altman\n95% LoA width (normalized)", (0, max(loa) * 1.15)),
    ):
        bars = axis.bar(np.arange(len(MODEL_ORDER)), values,
                        color=[MODEL_COLORS[item] for item in MODEL_ORDER], width=0.72)
        axis.set_xticks(np.arange(len(MODEL_ORDER)), [MODEL_LABELS[item] for item in MODEL_ORDER],
                        rotation=30, ha="right")
        axis.set_ylim(*ylim); axis.set_title(title); axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 2),
                          textcoords="offset points", ha="center", fontsize=5.5)
    _annotated_heatmap(axes[2], _matrix(macro, INTERVAL_PARAMETERS, "mae"), INTERVAL_PARAMETERS,
                       "Clinical interval MAE (ms)\n11-lead macro mean", ".1f", "YlOrRd")
    return _save_figure(figure, output)


def _plot_clinical(macro_rows: list[dict[str, object]], output: Path) -> list[Path]:
    _configure_ieee_style()
    macro = {(row["model"], row["parameter"]): row for row in macro_rows if row["status"] == "ok"}
    figure, axes = plt.subplots(2, 3, figsize=(7.16, 4.5), constrained_layout=True)
    panels = (
        (INTERVAL_PARAMETERS, "mae", "Interval MAE (ms)", ".1f", "YlOrRd", None, None),
        (INTERVAL_PARAMETERS, "pearson_r", "Interval Pearson r", ".2f", "RdBu_r", -1, 1),
        (INTERVAL_PARAMETERS, "ba_loa_width", "Interval BA LoA width (ms)", ".1f", "YlOrRd", None, None),
        (AMPLITUDE_PARAMETERS, "mae", "Morphology MAE\nnormalized units", ".3f", "YlOrRd", None, None),
        (AMPLITUDE_PARAMETERS, "pearson_r", "Morphology Pearson r\nnormalized units", ".2f", "RdBu_r", -1, 1),
        (AMPLITUDE_PARAMETERS, "ba_loa_width", "Morphology BA LoA width\nnormalized units", ".3f", "YlOrRd", None, None),
    )
    for axis, (parameters, field, title, fmt, cmap, vmin, vmax) in zip(axes.flat, panels):
        _annotated_heatmap(axis, _matrix(macro, parameters, field), parameters, title, fmt, cmap, vmin, vmax)
    figure.suptitle("CPSC2018 record-level ECG-parameter agreement (raw synchronized pairs)", fontsize=8.5)
    return _save_figure(figure, output)


def _plot_bland_altman(pairs: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
                       model: str, output: Path) -> list[Path]:
    _configure_ieee_style()
    figure, axes = plt.subplots(3, 4, figsize=(7.16, 6.0), constrained_layout=True)
    for axis, parameter in zip(axes.flat, PARAMETERS):
        real, generated = pairs.get((model, parameter), (np.empty(0), np.empty(0)))
        if len(real) >= 2:
            means, differences = (real + generated) / 2, generated - real
            agreement = bland_altman(real, generated)
            axis.scatter(means, differences, s=4, alpha=0.16, color=MODEL_COLORS[model], rasterized=True)
            axis.axhline(agreement["bias"], color="#111827", linewidth=0.9, label="Bias")
            axis.axhline(agreement["lower_limit"], color="#6b7280", linewidth=0.75, linestyle="--", label="95% LoA")
            axis.axhline(agreement["upper_limit"], color="#6b7280", linewidth=0.75, linestyle="--")
        unit = "bpm" if parameter == "heart_rate_bpm" else ("ms" if parameter in INTERVAL_PARAMETERS else "normalized")
        axis.set_title(f"{PARAMETER_LABELS[parameter]} (n={len(real)})")
        axis.set_xlabel(f"Pair mean ({unit})"); axis.set_ylabel(f"Generated - real ({unit})"); axis.grid(alpha=0.16)
    for axis in axes.flat[len(PARAMETERS) :]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, loc="best")
    figure.suptitle(f"{MODEL_LABELS[model]} Lead I ECG-parameter Bland-Altman (record level)", fontsize=8.5)
    return _save_figure(figure, output)


def _count_summary(values: list[int]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.int64)
    return {"n": int(len(array)), "min": int(np.min(array)) if len(array) else None,
            "median": float(np.median(array)) if len(array) else None,
            "max": int(np.max(array)) if len(array) else None,
            "fraction_at_least_2_rr": float(np.mean(array >= 2)) if len(array) else None,
            "fraction_at_least_5_rr": float(np.mean(array >= 5)) if len(array) else None}


def run(args: argparse.Namespace) -> Path:
    global MODEL_ORDER
    if args.models:
        requested = tuple(item.strip() for item in args.models.split(",") if item.strip())
        if not requested or any(item not in MODEL_LABELS for item in requested):
            raise ValueError("unsupported CPSC2018 clinical model selection")
        MODEL_ORDER = requested
    input_dir, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_protocol = json.loads((input_dir / "protocol.json").read_text(encoding="utf-8"))
    phase_applied = source_protocol["protocol"].get(
        "phase_correction_applied",
        source_protocol["protocol"].get("phase_correction_applied_to_primary"),
    )
    if phase_applied is None and source_protocol["protocol"].get("raw_full_window_primary") is True:
        phase_applied = False
    if source_protocol.get("status") not in {"completed", "smoke_completed"} or phase_applied is not False:
        raise ValueError("CPSC2018 clinical analysis requires completed raw, unshifted predictions")
    pair_path = input_dir / "paired_reference.npz"
    if not pair_path.exists():
        pair_path = input_dir / "paired_predictions.npz"
    with np.load(pair_path, allow_pickle=False) as artifact:
        targets, record_ids = np.asarray(artifact["targets"], dtype=np.float32), np.asarray(artifact["record_ids"])
        embedded_predictions = {
            model: np.asarray(artifact[f"{model}_predictions"], dtype=np.float32)
            for model in MODEL_ORDER
            if f"{model}_predictions" in artifact.files
        }
    count = len(targets) if args.max_records is None else min(args.max_records, len(targets))
    targets, record_ids = targets[:count], record_ids[:count]
    predictions = {
        model: embedded_predictions[model][:count]
        if model in embedded_predictions
        else np.asarray(np.load(input_dir / f"{model}_predictions.npy", mmap_mode="r")[:count])
        for model in MODEL_ORDER
    }
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    rows, detail_rows, delineation = [], [], {}
    lead_i_pairs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    real_hrv_audit = {}
    try:
        for lead_index, lead in enumerate(TARGET_LEADS):
            print(f"clinical delineation: real lead {lead}", flush=True)
            real = _measure_records(targets[:, lead_index], args.sampling_rate, executor)
            failures = Counter(item["failure_reason"] for item in real if not item["success"])
            delineation[lead] = {"real_success": int(sum(item["success"] for item in real)),
                                 "real_failure_reasons": dict(failures)}
            real_hrv_audit[lead] = {
                "r_peak_count": _count_summary([item["r_peak_count"] for item in real if item["success"]]),
                "nn_interval_count": _count_summary([item["n_rr"] for item in real if item["success"]]),
                "hrv_status_counts": dict(Counter(item["hrv_status"] for item in real)),
            }
            for model in MODEL_ORDER:
                print(f"clinical delineation: {model} lead {lead}", flush=True)
                generated = _measure_records(predictions[model][:, lead_index], args.sampling_rate, executor)
                failures = Counter(item["failure_reason"] for item in generated if not item["success"])
                delineation[lead][f"{model}_success"] = int(sum(item["success"] for item in generated))
                delineation[lead][f"{model}_failure_reasons"] = dict(failures)
                for parameter in PARAMETERS:
                    indices, real_values, generated_values = _record_pairs(real, generated, parameter)
                    rows.append(_agreement_row(model, lead, parameter, real_values, generated_values))
                    if lead == "I":
                        lead_i_pairs[(model, parameter)] = (real_values, generated_values)
                for index, (real_item, generated_item) in enumerate(zip(real, generated)):
                    row = {"record_id": str(record_ids[index]), "lead": lead, "model": model,
                           "real_success": real_item["success"], "generated_success": generated_item["success"],
                           "real_failure_reason": real_item["failure_reason"],
                           "generated_failure_reason": generated_item["failure_reason"],
                           "real_r_peak_count": real_item["r_peak_count"], "generated_r_peak_count": generated_item["r_peak_count"],
                           "real_n_rr": real_item["n_rr"], "generated_n_rr": generated_item["n_rr"]}
                    for parameter in PARAMETERS:
                        row[f"real_{parameter}"] = real_item["summary"].get(parameter)
                        row[f"generated_{parameter}"] = generated_item["summary"].get(parameter)
                    detail_rows.append(row)
    finally:
        if executor is not None:
            executor.shutdown()
    macro = _macro_rows(rows)
    per_lead_path = output / "per_lead_record_agreement.csv"; _write_csv(per_lead_path, rows)
    macro_path = output / "macro_lead_agreement.csv"; _write_csv(macro_path, macro)
    detail_path = output / "per_record_parameters.csv"; _write_csv(detail_path, detail_rows)
    waveform = json.loads((input_dir / "waveform_summary.json").read_text(encoding="utf-8"))
    summary_path = output / "cpsc2018_clinical_summary.json"
    summary = {
        "schema_version": 1,
        "protocol": {"records": count, "analysis_unit": "record", "leads": list(TARGET_LEADS),
                     "sampling_rate_hz": args.sampling_rate, "window_seconds": 4,
                     "alignment": "raw synchronized no phase correction",
                     "delineation": "independent NeuroKit2 DWT per record and lead",
                     "lead_aggregation": "unweighted macro mean of independent lead agreements",
                     "qtc_formula": "fridericia", "st_offset_ms": 60.0,
                     "amplitude_unit": "normalized_record_minmax_neg1_1_not_mV",
                     "hrv": "blocked_non_continuous_four_second_records"},
        "hrv_eligibility_audit": {
            "decision": "unavailable_for_claims", "minimum_protocol_duration_seconds": 30.0,
            "record_duration_seconds": 4.0, "records_are_continuous": False,
            "cross_record_concatenation": "prohibited", "real_signal_distributions": real_hrv_audit,
        },
        "delineation": delineation, "per_lead_agreement": rows,
        "macro_lead_agreement": macro, "waveform_agreement": waveform["models"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    outputs = [per_lead_path, macro_path, detail_path, summary_path]
    outputs.extend(_plot_summary(waveform, macro, output / "cpsc2018_clinical_main"))
    outputs.extend(_plot_clinical(macro, output / "cpsc2018_ecg_parameter_agreement"))
    for model in MODEL_ORDER:
        outputs.extend(_plot_bland_altman(lead_i_pairs, model, output / f"{model}_lead_i_ecg_parameter_bland_altman"))
    protocol = {
        "schema_version": 1, "status": "completed" if count == 686 else "smoke_completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv),
        "script_sha256": _sha256(Path(__file__)), "source_protocol_sha256": _sha256(input_dir / "protocol.json"),
        "records": count, "workers": args.workers,
        "figure_style": {"width_inches": 7.16, "font_family": "Liberation Serif (Times-compatible)",
                         "pdf_fonttype": 42, "png_dpi": 600},
        "claim_boundary": "Validation-only record-level descriptive analysis; normalized amplitudes are not physical; HRV is blocked.",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "neurokit2": getattr(nk, "__version__", "unknown")},
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sampling_rate", type=float, default=128.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--models", help="comma-separated model keys; defaults to the historical four-model set")
    return parser


if __name__ == "__main__":
    result = run(build_argparser().parse_args())
    print(f"CPSC2018 clinical agreement saved to {result}")
