"""Evaluate raw PTB-XL multi-lead ECG parameters and make IEEE-style figures."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import platform
import shlex
import sys
import warnings
from collections import defaultdict
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
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _sha256
from scripts.evaluate_ptbxl_fourway import MODEL_ORDER, TARGET_INDICES, TARGET_LEADS
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
MODEL_LABELS = {
    "cfm": "CFM", "cfm_ot": "CFM+OT", "rcfm": "RCFM", "rcfm_ot": "RCFM-OT",
    "pan": "RCFM-Pan", "pan_ot": "RCFM-Pan-OT", "diag": "RCFM-DiagMask",
    "diag_ot": "RCFM-DiagMask-OT", "rddm": "RDDM-ECG",
    "semantic": "RCFM-SemanticMask",
    "ecgmamba_diag": "ECGMamba-Diag (neg. ctrl.)",
    "ecgmamba_semantic": "ECGMamba-Sem",
    "diag_l003": r"DiagMask $\lambda=0.03$",
    "diag_l010": r"DiagMask $\lambda=0.1$",
    "diag_l030": r"DiagMask $\lambda=0.3$",
}
MODEL_COLORS = {
    "cfm": "#2878b5", "cfm_ot": "#76a5d5", "rcfm": "#2f8f5b", "rcfm_ot": "#c43d4b",
    "pan": "#2f8f5b", "pan_ot": "#73b58c", "diag": "#b44c97", "diag_ot": "#7b3f98",
    "rddm": "#d17a00",
    "semantic": "#008b8b",
    "ecgmamba_diag": "#8c564b",
    "ecgmamba_semantic": "#009e73",
    "diag_l003": "#d98abf", "diag_l010": "#b44c97", "diag_l030": "#6f2d78",
}
MODEL_LABELS["direct_cnn"] = "Direct CNN"
MODEL_COLORS["direct_cnn"] = "#6b4c9a"


def _valid_source_protocol(source_protocol: Mapping[str, object], max_records: int | None) -> bool:
    status = source_protocol.get("status")
    status_allowed = status == "completed" or (
        status == "smoke_completed" and max_records is not None
    )
    protocol = source_protocol.get("protocol")
    return bool(
        status_allowed
        and isinstance(protocol, Mapping)
        and protocol.get("phase_correction_applied") is False
    )


def _configure_ieee_style() -> None:
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
        }
    )


def _inverse_oracle_minmax(
    normalized: np.ndarray, minima: np.ndarray, ranges: np.ndarray
) -> np.ndarray:
    values = np.asarray(normalized, dtype=np.float32)
    offsets = np.asarray(minima, dtype=np.float32)
    scales = np.asarray(ranges, dtype=np.float32)
    if values.ndim != 3 or offsets.shape != values.shape[:2] or scales.shape != offsets.shape:
        raise ValueError("normalized waveforms and per-record/per-lead coefficients must align")
    if np.any(scales <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("inverse min-max inputs must be finite with positive ranges")
    return ((values + 1.0) * scales[..., None] / 2.0 + offsets[..., None]).astype(np.float32)


def _physical_record_measurement(
    signal: np.ndarray, sampling_rate: float, p_wave_applicable: bool = True
) -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Too few peaks detected.*")
        delineation = delineate_ecg(signal, sampling_rate=sampling_rate, method="dwt")
    if not delineation.success:
        return {"success": False, "failure_reason": delineation.failure_reason, "summary": {}}
    measurement = measure_ecg_parameters(
        signal,
        sampling_rate=sampling_rate,
        fiducials=delineation.fiducials,
        amplitude_unit="mV",
        inverse_transformed=True,
        qtc_formula="fridericia",
        st_offset_ms=60.0,
        continuous=False,
        p_wave_applicable=p_wave_applicable,
    )
    summary = {
        name: float(np.mean(values))
        for name, values in measurement["parameters"].items()
        if values
    }
    if summary.get("rr_ms", 0) > 0:
        summary["heart_rate_bpm"] = 60000.0 / summary["rr_ms"]
    return {
        "success": True,
        "failure_reason": None,
        "summary": summary,
        "p_wave_status": measurement["p_wave_status"],
        "amplitude_status": measurement["amplitude_status"],
        "hrv_status": measurement["hrv"]["status"],
    }


def _measure_task(task: tuple[np.ndarray, float, bool]) -> dict[str, object]:
    return _physical_record_measurement(*task)


def _measure_records(
    signals: np.ndarray,
    sampling_rate: float,
    p_wave_applicable: np.ndarray,
    executor: ProcessPoolExecutor | None,
) -> list[dict[str, object]]:
    if len(signals) != len(p_wave_applicable):
        raise ValueError("P-wave applicability must align with signals")
    tasks = (
        (np.asarray(signal), sampling_rate, bool(applicable))
        for signal, applicable in zip(signals, p_wave_applicable)
    )
    if executor is None:
        return [_measure_task(task) for task in tasks]
    return list(executor.map(_measure_task, tasks, chunksize=16))


def _patient_parameter_pairs(
    real: list[dict[str, object]],
    generated: list[dict[str, object]],
    patient_ids: np.ndarray,
    parameter: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for patient, real_record, generated_record in zip(patient_ids, real, generated):
        pair = (real_record["summary"].get(parameter), generated_record["summary"].get(parameter))
        if any(value is None or not np.isfinite(value) for value in pair):
            continue
        grouped[str(patient)][0].append(float(pair[0]))
        grouped[str(patient)][1].append(float(pair[1]))
    patients = np.asarray(sorted(grouped))
    return (
        patients,
        np.asarray([np.mean(grouped[patient][0]) for patient in patients]),
        np.asarray([np.mean(grouped[patient][1]) for patient in patients]),
    )


def _agreement_row(
    model: str,
    lead: str,
    parameter: str,
    real: np.ndarray,
    generated: np.ndarray,
) -> dict[str, object]:
    unit = "bpm" if parameter == "heart_rate_bpm" else ("ms" if parameter in INTERVAL_PARAMETERS else "mV")
    base = {"model": model, "lead": lead, "parameter": parameter, "unit": unit, "n_patients": int(len(real))}
    if len(real) < 2:
        return {**base, "status": "insufficient_patient_pairs"}
    error = generated - real
    correlation = paired_correlation(real, generated)
    agreement = bland_altman(real, generated)
    return {
        **base,
        "status": "ok",
        "reference_mean": float(np.mean(real)),
        "generated_mean": float(np.mean(generated)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "pearson_r": correlation["r"],
        "ba_bias": agreement["bias"],
        "ba_difference_sd": agreement["difference_sd"],
        "ba_lower": agreement["lower_limit"],
        "ba_upper": agreement["upper_limit"],
        "ba_loa_width": agreement["upper_limit"] - agreement["lower_limit"],
        "inference_scope": "patient_level_descriptive_single_training_seed",
    }


def _macro_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for model in MODEL_ORDER:
        for parameter in PARAMETERS:
            matched = [row for row in rows if row["model"] == model and row["parameter"] == parameter and row["status"] == "ok"]
            base = {"model": model, "parameter": parameter, "lead_aggregation": "unweighted_mean_across_independent_lead_agreements", "usable_leads": len(matched)}
            if not matched:
                output.append({**base, "status": "insufficient_leads"})
                continue
            fields = ("n_patients", "mae", "rmse", "ba_bias", "ba_difference_sd", "ba_lower", "ba_upper", "ba_loa_width")
            correlations = np.asarray([row["pearson_r"] for row in matched if row["pearson_r"] is not None], dtype=np.float64)
            fisher_correlation = (
                float(np.tanh(np.mean(np.arctanh(np.clip(correlations, -0.999999, 0.999999)))))
                if len(correlations)
                else None
            )
            output.append(
                {
                    **base,
                    "status": "ok",
                    "unit": matched[0]["unit"],
                    "pearson_r": fisher_correlation,
                    "pearson_aggregation": "unweighted_Fisher_z_mean_across_leads",
                    **{
                        field: float(np.mean([row[field] for row in matched if row[field] is not None]))
                        for field in fields
                    },
                }
            )
    return output


def _matrix(macro: Mapping[tuple[str, str], Mapping[str, object]], parameters: tuple[str, ...], field: str) -> np.ndarray:
    return np.asarray([[macro[(model, parameter)][field] for parameter in parameters] for model in MODEL_ORDER], dtype=np.float64)


def _load_p_wave_applicability(
    metadata_csv: Path, record_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    metadata = pd.read_csv(metadata_csv, usecols=["ecg_id", "scp_codes"])
    if metadata.ecg_id.duplicated().any():
        raise ValueError("PTB-XL metadata ECG IDs must be unique")
    code_map = {
        int(row.ecg_id): set(ast.literal_eval(str(row.scp_codes)))
        for row in metadata.itertuples(index=False)
    }
    missing = [str(record_id) for record_id in record_ids if int(record_id) not in code_map]
    if missing:
        raise ValueError(f"PTB-XL metadata is missing evaluated ECG IDs: {missing[:5]}")
    excluded_codes = {"AFIB", "AFLT"}
    applicable = np.asarray(
        [not bool(code_map[int(record_id)] & excluded_codes) for record_id in record_ids],
        dtype=bool,
    )
    return applicable, {
        "policy": "PR and P amplitude excluded for official AFIB or AFLT statements",
        "excluded_codes": sorted(excluded_codes),
        "applicable_records": int(np.sum(applicable)),
        "not_applicable_records": int(np.sum(~applicable)),
    }


def _annotated_heatmap(axis: plt.Axes, values: np.ndarray, parameters: tuple[str, ...], title: str, fmt: str, cmap: str = "viridis", vmin=None, vmax=None) -> None:
    image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(parameters)), [PARAMETER_LABELS[item] for item in parameters])
    axis.set_yticks(np.arange(len(MODEL_ORDER)), [MODEL_LABELS[item] for item in MODEL_ORDER])
    axis.set_title(title)
    midpoint = (np.nanmin(values) + np.nanmax(values)) / 2.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(column, row, format(value, fmt), ha="center", va="center", fontsize=5.6, color="white" if value > midpoint else "black")
    return image


def _save_figure(figure: plt.Figure, stem: Path) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=600 if suffix == "png" else None)
        paths.append(path)
    plt.close(figure)
    return paths


def _plot_main(summary: Mapping[str, object], macro_rows: list[dict[str, object]], output: Path) -> list[Path]:
    _configure_ieee_style()
    macro = {(row["model"], row["parameter"]): row for row in macro_rows if row["status"] == "ok"}
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), constrained_layout=True)
    correlation = [summary["models"][model]["per_record_pearson_median"] for model in MODEL_ORDER]
    loa = []
    for model in MODEL_ORDER:
        ba = summary["models"][model]["pointwise_bland_altman_descriptive_only"]
        loa.append(ba["upper_limit"] - ba["lower_limit"])
    for axis, values, title, limit in (
        (axes[0], correlation, "Waveform correlation\nmedian record Pearson", (0, 1.0)),
        (axes[1], loa, "Waveform Bland-Altman\n95% LoA width", (0, max(loa) * 1.15)),
    ):
        bars = axis.bar(np.arange(len(MODEL_ORDER)), values, color=[MODEL_COLORS[m] for m in MODEL_ORDER], width=0.72)
        axis.set_xticks(np.arange(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=30, ha="right")
        axis.set_ylim(*limit); axis.set_title(title); axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=5.5)
    values = _matrix(macro, INTERVAL_PARAMETERS, "mae")
    _annotated_heatmap(axes[2], values, INTERVAL_PARAMETERS, "Clinical interval MAE (ms)\n11-lead macro mean", ".1f", cmap="YlOrRd")
    return _save_figure(figure, output)


def _plot_clinical(macro_rows: list[dict[str, object]], output: Path) -> list[Path]:
    _configure_ieee_style()
    macro = {(row["model"], row["parameter"]): row for row in macro_rows if row["status"] == "ok"}
    figure, axes = plt.subplots(2, 3, figsize=(7.16, 4.5), constrained_layout=True)
    panels = (
        (INTERVAL_PARAMETERS, "mae", "Interval MAE (ms)", ".1f", "YlOrRd", None, None),
        (INTERVAL_PARAMETERS, "pearson_r", "Interval Pearson r", ".2f", "RdBu_r", -1, 1),
        (INTERVAL_PARAMETERS, "ba_loa_width", "Interval BA LoA\nwidth (ms)", ".1f", "YlOrRd", None, None),
        (AMPLITUDE_PARAMETERS, "mae", "Amplitude MAE (mV)\nOracle inverse", ".3f", "YlOrRd", None, None),
        (AMPLITUDE_PARAMETERS, "pearson_r", "Amplitude Pearson r\nOracle inverse", ".2f", "RdBu_r", -1, 1),
        (AMPLITUDE_PARAMETERS, "ba_loa_width", "Amplitude BA LoA\nwidth (mV)\nOracle inverse", ".3f", "YlOrRd", None, None),
    )
    for axis, (parameters, field, title, fmt, cmap, vmin, vmax) in zip(axes.flat, panels):
        _annotated_heatmap(axis, _matrix(macro, parameters, field), parameters, title, fmt, cmap, vmin, vmax)
    figure.suptitle("PTB-XL patient-level ECG-parameter agreement (raw synchronized pairs)", fontsize=8.5)
    return _save_figure(figure, output)


def _plot_bland_altman(
    pairs: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    model: str,
    output: Path,
    lead: str = "I",
    parameters: tuple[str, ...] = PARAMETERS,
) -> list[Path]:
    _configure_ieee_style()
    figure, axes = plt.subplots(3, 4, figsize=(7.16, 6.0), constrained_layout=True)
    color = MODEL_COLORS[model]
    for axis, parameter in zip(axes.flat, parameters):
        real, generated = pairs[(model, parameter)]
        means = (real + generated) / 2.0
        differences = generated - real
        agreement = bland_altman(real, generated)
        if len(means) > 900:
            indices = np.linspace(0, len(means) - 1, 900, dtype=np.int64)
            plot_means, plot_differences = means[indices], differences[indices]
        else:
            plot_means, plot_differences = means, differences
        axis.scatter(plot_means, plot_differences, s=4, alpha=0.16, color=color, rasterized=True)
        axis.axhline(agreement["bias"], color="#111827", linewidth=0.9, label="Bias")
        axis.axhline(agreement["lower_limit"], color="#6b7280", linewidth=0.75, linestyle="--", label="95% LoA")
        axis.axhline(agreement["upper_limit"], color="#6b7280", linewidth=0.75, linestyle="--")
        unit = "bpm" if parameter == "heart_rate_bpm" else ("ms" if parameter in INTERVAL_PARAMETERS else "mV")
        axis.set_title(f"{PARAMETER_LABELS[parameter]} (n={len(real)})")
        axis.set_xlabel(f"Pair mean ({unit})")
        axis.set_ylabel(f"Generated - real ({unit})")
        axis.grid(alpha=0.16)
    for axis in axes.flat[len(parameters) :]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, loc="best")
    figure.suptitle(
        f"{MODEL_LABELS[model]} Lead {lead} ECG-parameter Bland-Altman (patient level)",
        fontsize=8.5,
    )
    return _save_figure(figure, output)


def run(args: argparse.Namespace) -> Path:
    global MODEL_ORDER
    MODEL_ORDER = tuple(args.models)
    unsupported = [model for model in MODEL_ORDER if model not in MODEL_LABELS]
    if not MODEL_ORDER or unsupported or len(set(MODEL_ORDER)) != len(MODEL_ORDER):
        raise ValueError(f"invalid PTB-XL clinical model list: {unsupported}")
    input_dir, data_dir, output_dir = args.input_dir.resolve(), args.data_dir.resolve(), args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_protocol = json.loads((input_dir / "protocol.json").read_text(encoding="utf-8"))
    if not _valid_source_protocol(source_protocol, args.max_records):
        raise ValueError("PTB-XL clinical analysis requires completed raw predictions")
    source_comparisons = source_protocol.get("analysis_comparisons") or []
    analysis_comparisons = [
        pair for pair in source_comparisons
        if len(pair) == 2 and pair[0] in MODEL_ORDER and pair[1] in MODEL_ORDER
    ]
    with np.load(input_dir / "paired_reference.npz", allow_pickle=False) as artifact:
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        patient_ids = np.asarray(artifact["patient_ids"])
        record_ids = np.asarray(artifact["record_ids"])
    count = len(targets) if args.max_records is None else min(args.max_records, len(targets))
    targets, patient_ids, record_ids = targets[:count], patient_ids[:count], record_ids[:count]
    p_wave_applicable, p_wave_policy = _load_p_wave_applicability(
        args.metadata_csv.resolve(), record_ids
    )
    minima = np.load(data_dir / "record_minima_test.npy", allow_pickle=False)[:count, list(TARGET_INDICES)]
    ranges = np.load(data_dir / "record_ranges_test.npy", allow_pickle=False)[:count, list(TARGET_INDICES)]
    physical_targets = _inverse_oracle_minmax(targets, minima, ranges)
    source = np.load(data_dir / "X_test_resampled.npy", mmap_mode="r")[:count, :512, :]
    expected = np.transpose(source[:, :, list(TARGET_INDICES)], (0, 2, 1))
    if not np.allclose(physical_targets, expected, atol=2e-6, rtol=2e-6):
        raise ValueError("target inverse transform does not reproduce stored PTB-XL mV waveforms")
    physical_predictions = {
        model: _inverse_oracle_minmax(
            np.asarray(np.load(input_dir / f"{model}_predictions.npy", mmap_mode="r")[:count]), minima, ranges
        )
        for model in MODEL_ORDER
    }
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    rows: list[dict[str, object]] = []
    per_record_rows: list[dict[str, object]] = []
    delineation = {}
    if args.representative_lead not in TARGET_LEADS:
        raise ValueError(f"representative lead must be one of {TARGET_LEADS}")
    representative_ba_pairs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    try:
        for lead_index, lead in enumerate(TARGET_LEADS):
            print(f"clinical delineation: real lead {lead}", flush=True)
            real = _measure_records(
                physical_targets[:, lead_index], args.sampling_rate, p_wave_applicable, executor
            )
            delineation.setdefault(lead, {})["real_success"] = int(sum(item["success"] for item in real))
            for model in MODEL_ORDER:
                print(f"clinical delineation: {model} lead {lead}", flush=True)
                generated = _measure_records(
                    physical_predictions[model][:, lead_index],
                    args.sampling_rate,
                    p_wave_applicable,
                    executor,
                )
                delineation[lead][f"{model}_success"] = int(sum(item["success"] for item in generated))
                for parameter in PARAMETERS:
                    patients, real_values, generated_values = _patient_parameter_pairs(real, generated, patient_ids, parameter)
                    rows.append(_agreement_row(model, lead, parameter, real_values, generated_values))
                    if lead == args.representative_lead:
                        representative_ba_pairs[(model, parameter)] = (real_values, generated_values)
                for index, (real_item, generated_item) in enumerate(zip(real, generated)):
                    row = {
                        "record_id": str(record_ids[index]), "patient_id": str(patient_ids[index]),
                        "lead": lead, "model": model, "real_success": real_item["success"],
                        "generated_success": generated_item["success"],
                        "p_wave_applicable": bool(p_wave_applicable[index]),
                    }
                    for parameter in PARAMETERS:
                        row[f"real_{parameter}"] = real_item["summary"].get(parameter)
                        row[f"generated_{parameter}"] = generated_item["summary"].get(parameter)
                    per_record_rows.append(row)
    finally:
        if executor is not None:
            executor.shutdown()
    macro = _macro_rows(rows)
    row_path = output_dir / "per_lead_patient_agreement.csv"
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    macro_path = output_dir / "macro_lead_agreement.csv"
    with macro_path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted(set().union(*(row.keys() for row in macro)))
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(macro)
    record_path = output_dir / "per_record_parameters.csv"
    with record_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(per_record_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(per_record_rows)
    waveform_summary = json.loads((input_dir / "waveform_summary.json").read_text(encoding="utf-8"))
    summary_path = output_dir / "ptbxl_clinical_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": {
                    "records": count, "patients": int(len(np.unique(patient_ids))),
                    "leads": list(TARGET_LEADS), "sampling_rate_hz": args.sampling_rate,
                    "window_seconds": 4, "alignment": "raw synchronized no phase correction",
                    "delineation": "independent NeuroKit2 DWT per record and lead",
                    "patient_aggregation": "mean paired records within patient before agreement",
                    "lead_aggregation": "unweighted macro mean of independent lead agreements",
                    "qtc_formula": "fridericia", "st_offset_ms": 60.0,
                    "amplitude_unit": "mV", "generated_inverse": "oracle ground-truth target min/range",
                    "hrv": "blocked_four_second_records", "p_wave": p_wave_policy,
                    "representative_bland_altman_lead": args.representative_lead,
                },
                "delineation": delineation,
                "per_lead_agreement": rows,
                "macro_lead_agreement": macro,
                "waveform_agreement": waveform_summary["models"],
            },
            indent=2, sort_keys=True, allow_nan=False,
        ),
        encoding="utf-8",
    )
    outputs = [row_path, macro_path, record_path, summary_path]
    outputs.extend(_plot_main(waveform_summary, macro, output_dir / "ptbxl_clinical_main"))
    outputs.extend(_plot_clinical(macro, output_dir / "ptbxl_ecg_parameter_agreement"))
    for model in MODEL_ORDER:
        outputs.extend(
            _plot_bland_altman(
                representative_ba_pairs,
                model,
                output_dir / f"{model}_lead_{args.representative_lead.lower()}_ecg_parameter_bland_altman",
                lead=args.representative_lead,
            )
        )
    status = "completed" if count == 2203 else "smoke_completed"
    protocol = {
        "schema_version": 1, "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv), "script_sha256": _sha256(Path(__file__)),
        "source_protocol_sha256": _sha256(input_dir / "protocol.json"),
        "source_waveform_summary_sha256": _sha256(input_dir / "waveform_summary.json"),
        "records": count, "patients": int(len(np.unique(patient_ids))), "workers": args.workers,
        "representative_bland_altman_lead": args.representative_lead,
        "models": list(MODEL_ORDER),
        "analysis_comparisons": analysis_comparisons or None,
        "metadata_csv": {"path": str(args.metadata_csv.resolve()), "sha256": _sha256(args.metadata_csv)},
        "p_wave_policy": p_wave_policy,
        "figure_style": {"width_inches": 7.16, "font_family": "Liberation Serif (Times-compatible)", "pdf_fonttype": 42, "png_dpi": 600},
        "claim_boundary": "Amplitude results use oracle target scalers; HRV blocked; automatic DWT and one training seed support descriptive analysis only.",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "neurokit2": getattr(nk, "__version__", "unknown")},
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--sampling_rate", type=float, default=128.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--representative_lead", choices=TARGET_LEADS, default="V3")
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_ORDER),
        help="Prediction filename prefixes (default: cfm rcfm rcfm_ot rddm).",
    )
    return parser


if __name__ == "__main__":
    output = run(build_argparser().parse_args())
    print(f"PTB-XL clinical agreement saved to {output}")
