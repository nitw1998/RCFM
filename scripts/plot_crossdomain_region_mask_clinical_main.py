"""Render CPSC2018 and MIMIC-AFib summaries in the PTB-XL figure style.

The source runs compare CFM, Pan-region RCFM, Pan-region RCFM with minibatch
OT, and RDDM. They are not silently relabelled as downstream DiagMask runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ("cfm", "rcfm", "rcfm_ot", "rddm")
MODEL_LABELS = {"cfm": "CFM", "rcfm": "RCFM-Pan", "rcfm_ot": "RCFM-Pan+OT", "rddm": "RDDM"}
MODEL_COLORS = {"cfm": "#2878b5", "rcfm": "#2f8f5b", "rcfm_ot": "#c43d4b", "rddm": "#d17a00"}
INTERVAL_PARAMETERS = ("rr_ms", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms")
PARAMETER_LABELS = {"rr_ms": "RR", "pr_ms": "PR", "qrs_ms": "QRS", "qt_ms": "QT", "qtc_ms": "QTc"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _require_finite(values: np.ndarray, name: str) -> np.ndarray:
    allowed = {(len(MODEL_ORDER),), (len(MODEL_ORDER), len(INTERVAL_PARAMETERS))}
    if values.shape not in allowed:
        raise ValueError(f"Unexpected {name} shape: {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite value in {name}")
    return values


def extract_cpsc(summary: Mapping[str, object]) -> dict[str, np.ndarray]:
    protocol = summary["protocol"]
    if protocol.get("alignment") != "raw synchronized no phase correction":
        raise ValueError("CPSC input is not the frozen raw-synchronized protocol")
    if int(protocol.get("records", -1)) != 686:
        raise ValueError("CPSC input does not contain the expected 686 records")
    waveform = summary["waveform_agreement"]
    correlations, loa_widths = [], []
    for model in MODEL_ORDER:
        row = waveform[model]
        correlations.append(row["per_record_pearson_median"])
        ba = row["pointwise_bland_altman_descriptive_only"]
        loa_widths.append(ba["upper_limit"] - ba["lower_limit"])
    indexed = {(row["model"], row["parameter"]): row for row in summary["macro_lead_agreement"] if row.get("status") == "ok"}
    interval_mae = np.asarray(
        [[indexed[(model, parameter)]["mae"] for parameter in INTERVAL_PARAMETERS] for model in MODEL_ORDER],
        dtype=np.float64,
    )
    return {
        "correlation": _require_finite(np.asarray(correlations), "CPSC correlations"),
        "loa_width": _require_finite(np.asarray(loa_widths), "CPSC LoA widths"),
        "interval_mae": _require_finite(interval_mae, "CPSC interval MAE"),
    }


def extract_mimic(summary: Mapping[str, object]) -> dict[str, np.ndarray]:
    protocol = summary["protocol"]
    if protocol.get("phase_correction") != "target-informed per-window oracle Pearson maximization":
        raise ValueError("MIMIC input is not the frozen oracle-aligned protocol")
    if int(protocol.get("records", -1)) != 1800 or int(protocol.get("support_samples", -1)) != 480:
        raise ValueError("MIMIC input does not contain the expected 1,800 windows on 480-sample support")
    waveform = summary["waveform_agreement"]
    correlations, loa_widths = [], []
    for model in MODEL_ORDER:
        row = waveform[model]["oracle_aligned"]
        correlations.append(row["per_record_pearson_median"])
        ba = row["pointwise_bland_altman"]
        loa_widths.append(ba["upper_limit"] - ba["lower_limit"])
    indexed = {
        (row["model"], row["parameter"]): row for row in summary["parameter_agreement"]
        if row.get("status") == "ok" and row.get("phase_mode") == "oracle_aligned"
    }
    interval_mae = np.asarray(
        [[indexed[(model, parameter)]["mae"] for parameter in INTERVAL_PARAMETERS] for model in MODEL_ORDER],
        dtype=np.float64,
    )
    return {
        "correlation": _require_finite(np.asarray(correlations), "MIMIC correlations"),
        "loa_width": _require_finite(np.asarray(loa_widths), "MIMIC LoA widths"),
        "interval_mae": _require_finite(interval_mae, "MIMIC interval MAE"),
    }


def _configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman", "Times"],
        "font.size": 7.0, "axes.titlesize": 8.0, "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
    })


def plot_main(values: Mapping[str, np.ndarray], output_stem: Path, titles: Sequence[str]) -> list[Path]:
    _configure_style()
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), constrained_layout=True)
    for axis, metric, title, limit in (
        (axes[0], values["correlation"], titles[0], (0.0, 1.0)),
        (axes[1], values["loa_width"], titles[1], (0.0, float(np.max(values["loa_width"])) * 1.15)),
    ):
        bars = axis.bar(np.arange(len(MODEL_ORDER)), metric, color=[MODEL_COLORS[m] for m in MODEL_ORDER], width=0.72)
        axis.set_xticks(np.arange(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=30, ha="right")
        axis.set_ylim(*limit)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, metric):
            axis.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 2),
                          textcoords="offset points", ha="center", fontsize=5.5)

    matrix = values["interval_mae"]
    axes[2].imshow(matrix, aspect="auto", cmap="YlOrRd")
    axes[2].set_xticks(np.arange(len(INTERVAL_PARAMETERS)), [PARAMETER_LABELS[p] for p in INTERVAL_PARAMETERS])
    axes[2].set_yticks(np.arange(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER])
    axes[2].set_title(titles[2])
    midpoint = (float(np.min(matrix)) + float(np.max(matrix))) / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axes[2].text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=5.6,
                         color="white" if value > midpoint else "black")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        path = output_stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=600 if suffix == "png" else None)
        outputs.append(path)
    plt.close(figure)
    return outputs


def _serializable(values: Mapping[str, np.ndarray]) -> dict[str, list[object]]:
    return {key: value.tolist() for key, value in values.items()}


def run(args: argparse.Namespace) -> Path:
    cpsc_path, mimic_path = args.cpsc_summary.resolve(), args.mimic_summary.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cpsc_values, mimic_values = extract_cpsc(_load_json(cpsc_path)), extract_mimic(_load_json(mimic_path))
    outputs = {
        "cpsc2018": plot_main(cpsc_values, output_dir / "cpsc2018_region_mask_clinical_main", (
            "Waveform correlation\nmedian record Pearson", "Waveform Bland-Altman\n95% LoA width (normalized)",
            "Clinical interval MAE (ms)\n11-lead macro mean")),
        "mimic_afib": plot_main(mimic_values, output_dir / "mimic_afib_region_mask_clinical_main", (
            "Waveform correlation\nmedian 4-s-window Pearson", "Waveform Bland-Altman\n95% LoA width (aligned)",
            "Clinical interval MAE (ms)\noracle-aligned windows")),
    }
    published = []
    if args.publish_dir is not None:
        publish_dir = args.publish_dir.resolve()
        publish_dir.mkdir(parents=True, exist_ok=True)
        for paths in outputs.values():
            for source in paths:
                destination = publish_dir / source.name
                shutil.copy2(source, destination)
                published.append(str(destination))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "reference_figure": str(args.reference_figure.resolve()),
        "model_order": list(MODEL_ORDER), "model_labels": MODEL_LABELS,
        "inputs": {"cpsc2018": {"path": str(cpsc_path), "sha256": _sha256(cpsc_path)},
                   "mimic_afib": {"path": str(mimic_path), "sha256": _sha256(mimic_path)}},
        "values": {"cpsc2018": _serializable(cpsc_values), "mimic_afib": _serializable(mimic_values)},
        "outputs": [str(path) for paths in outputs.values() for path in paths], "published_outputs": published,
        "claim_boundaries": {
            "model_semantics": "RCFM and RCFM-OT in the source summaries are Pan-region models, not DiagMask models.",
            "cpsc2018": "Raw synchronized 686-record descriptive analysis; normalized amplitudes; 11-lead macro clinical MAE.",
            "mimic_afib": "Target-informed oracle-aligned descriptive analysis of 1,800 four-second windows on 480-sample support.",
            "mimic_af_status": "The legacy QC test sidecar has zero AF-positive test windows; this is not the lead-matched AF transfer/faithfulness cohort.",
        },
    }
    report_path = output_dir / "protocol.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpsc_summary", type=Path, default=Path("runs/evaluation/cpsc2018_fourway_clinical_raw_v1/cpsc2018_clinical_summary.json"))
    parser.add_argument("--mimic_summary", type=Path, default=Path("runs/clinical/mimic_afib_fourway_phase_clinical_maxlag16_v3/ecg_phase_clinical_summary.json"))
    parser.add_argument("--reference_figure", type=Path, default=Path("paper/data/figures/ptbxl_region_mask_clinical_main.pdf"))
    parser.add_argument("--output_dir", type=Path, default=Path("runs/figures/crossdomain_region_mask_clinical_main_v1"))
    parser.add_argument("--publish_dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
