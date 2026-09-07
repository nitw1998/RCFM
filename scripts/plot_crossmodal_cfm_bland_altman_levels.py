"""Plot window- and subject-level Bland--Altman views for CFM ECG parameters.

The window view is deliberately descriptive because windows from the same person
are correlated.  The subject view first averages paired measurements within each
person and then treats the person as the plotting unit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.metrics.bland_altman import bland_altman


PARAMETERS = (
    "heart_rate_bpm",
    "rr_ms",
    "pr_ms",
    "qrs_ms",
    "qt_ms",
    "qtc_ms",
    "p_amplitude",
    "r_amplitude",
    "qrs_peak_to_peak_amplitude",
    "t_amplitude",
    "st_deviation",
)
DISPLAY_NAMES = {
    "heart_rate_bpm": "Heart rate",
    "rr_ms": "RR interval",
    "pr_ms": "PR interval",
    "qrs_ms": "QRS duration",
    "qt_ms": "QT interval",
    "qtc_ms": "QTc interval",
    "p_amplitude": "P amplitude",
    "r_amplitude": "R amplitude",
    "qrs_peak_to_peak_amplitude": "QRS peak-to-peak",
    "t_amplitude": "T amplitude",
    "st_deviation": "ST deviation",
}
UNITS = {
    "heart_rate_bpm": "bpm",
    "rr_ms": "ms",
    "pr_ms": "ms",
    "qrs_ms": "ms",
    "qt_ms": "ms",
    "qtc_ms": "ms",
}
PHASE_STYLE = {
    "unshifted": ("Unshifted", "#4C78A8"),
    "oracle_aligned": ("Oracle-aligned", "#D1495B"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: str | None) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        result = float(value)
    except ValueError:
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _normalized_row(
    *, window_id: str, subject_id: str, phase: str, source: dict[str, str],
    generated_prefix: str,
) -> dict[str, object]:
    if not window_id or not subject_id:
        raise ValueError("window and subject identifiers must be nonempty")
    return {
        "window_id": window_id,
        "subject_id": subject_id,
        "phase": phase,
        "reference": {p: _number(source.get(f"reference_{p}")) for p in PARAMETERS},
        "generated": {p: _number(source.get(f"{generated_prefix}{p}")) for p in PARAMETERS},
    }


def load_mimic(directory: Path) -> list[dict[str, object]]:
    rows = _read_csv(directory / "per_window_ecg_parameters.csv")
    output: list[dict[str, object]] = []
    for row in rows:
        for phase in PHASE_STYLE:
            output.append(_normalized_row(
                window_id=row["window_index"], subject_id=row["subject_id"],
                phase=phase, source=row, generated_prefix=f"{phase}_",
            ))
    return output


def load_wesad(directory: Path) -> list[dict[str, object]]:
    output = []
    for row in _read_csv(directory / "per_window_ecg_parameters.csv"):
        phase = row["phase_mode"]
        if phase not in PHASE_STYLE:
            raise ValueError(f"unsupported WESAD phase mode: {phase}")
        output.append(_normalized_row(
            window_id=row["window_index"], subject_id=row["subject_id"],
            phase=phase, source=row, generated_prefix="generated_",
        ))
    return output


def load_mmecg(directory: Path, reference_directory: Path) -> list[dict[str, object]]:
    rows = _read_csv(directory / "per_window_parameters.csv")
    with np.load(reference_directory / "paired_reference.npz", allow_pickle=False) as archive:
        record_ids = np.asarray(archive["record_ids"]).astype(str)
        subject_ids = np.asarray(archive["subject_ids"]).astype(str)
    if len(rows) != len(record_ids) or len(rows) != len(subject_ids):
        raise ValueError("mmECG clinical rows and prediction identities have different lengths")
    output = []
    seen: set[int] = set()
    for row in rows:
        index = int(row["window_index"])
        if index < 0 or index >= len(subject_ids) or index in seen:
            raise ValueError(f"invalid or duplicate mmECG window index: {index}")
        seen.add(index)
        if row["record_id"] != record_ids[index]:
            raise ValueError(f"mmECG record identity mismatch at window {index}")
        output.append(_normalized_row(
            window_id=str(index), subject_id=subject_ids[index], phase="unshifted",
            source=row, generated_prefix="generated_",
        ))
    if seen != set(range(len(rows))):
        raise ValueError("mmECG window indices are incomplete")
    return output


def finite_parameter_pairs(
    rows: list[dict[str, object]], parameter: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subjects: list[str] = []
    reference: list[float] = []
    generated: list[float] = []
    for row in rows:
        real = float(row["reference"][parameter])
        fake = float(row["generated"][parameter])
        if np.isfinite(real) and np.isfinite(fake):
            subjects.append(str(row["subject_id"]))
            reference.append(real)
            generated.append(fake)
    return (
        np.asarray(subjects, dtype=str),
        np.asarray(reference, dtype=np.float64),
        np.asarray(generated, dtype=np.float64),
    )


def aggregate_subject_pairs(
    subjects: np.ndarray, reference: np.ndarray, generated: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if subjects.shape != reference.shape or reference.shape != generated.shape:
        raise ValueError("subject IDs and paired values must have identical shapes")
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for subject, real, fake in zip(subjects, reference, generated):
        if np.isfinite(real) and np.isfinite(fake):
            grouped[str(subject)].append((float(real), float(fake)))
    ids = np.asarray(sorted(grouped), dtype=str)
    real_means = np.asarray([np.mean([x[0] for x in grouped[s]]) for s in ids], dtype=np.float64)
    fake_means = np.asarray([np.mean([x[1] for x in grouped[s]]) for s in ids], dtype=np.float64)
    counts = np.asarray([len(grouped[s]) for s in ids], dtype=np.int64)
    return ids, real_means, fake_means, counts


def _agreement_row(
    dataset: str, level: str, phase: str, parameter: str,
    reference: np.ndarray, generated: np.ndarray, subjects: np.ndarray,
) -> dict[str, object]:
    base = {
        "dataset": dataset,
        "level": level,
        "phase_mode": phase,
        "parameter": parameter,
        "unit": UNITS.get(parameter, "normalized"),
        "n": int(reference.size),
        "n_subjects": int(np.unique(subjects).size),
        "difference_definition": "generated_minus_reference",
        "inference_status": (
            "descriptive_correlated_repeated_windows" if level == "window"
            else "descriptive_subject_level_within_subject_means"
        ),
    }
    if reference.size < 2:
        return {**base, "status": "insufficient_pairs"}
    agreement = bland_altman(reference, generated)
    return {
        **base,
        "status": "ok",
        "reference_mean": float(np.mean(reference)),
        "generated_mean": float(np.mean(generated)),
        "bias": agreement["bias"],
        "difference_sd": agreement["difference_sd"],
        "lower_limit": agreement["lower_limit"],
        "upper_limit": agreement["upper_limit"],
        "loa_width": agreement["upper_limit"] - agreement["lower_limit"],
        "mae": float(np.mean(np.abs(generated - reference))),
        "rmse": float(np.sqrt(np.mean(np.square(generated - reference)))),
    }


def compute_views(rows: list[dict[str, object]], dataset: str) -> tuple[dict, list[dict[str, object]]]:
    phases = [phase for phase in PHASE_STYLE if any(row["phase"] == phase for row in rows)]
    views: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {
        "window": {}, "subject": {},
    }
    summaries: list[dict[str, object]] = []
    for phase in phases:
        phase_rows = [row for row in rows if row["phase"] == phase]
        views["window"][phase] = {}
        views["subject"][phase] = {}
        for parameter in PARAMETERS:
            subjects, real, fake = finite_parameter_pairs(phase_rows, parameter)
            views["window"][phase][parameter] = (subjects, real, fake)
            summaries.append(_agreement_row(dataset, "window", phase, parameter, real, fake, subjects))
            subject_ids, subject_real, subject_fake, _ = aggregate_subject_pairs(subjects, real, fake)
            views["subject"][phase][parameter] = (subject_ids, subject_real, subject_fake)
            summaries.append(_agreement_row(
                dataset, "subject", phase, parameter,
                subject_real, subject_fake, subject_ids,
            ))
    return views, summaries


def _plot_dataset(
    dataset: str, views: dict, level: str, output: Path, model_label: str = "CFM"
) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "Times"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(3, 4, figsize=(7.16, 6.45), constrained_layout=True)
    phases = list(views[level])
    for axis, parameter in zip(axes.flat, PARAMETERS):
        title_counts = []
        for phase in phases:
            subjects, real, fake = views[level][phase][parameter]
            phase_label, color = PHASE_STYLE[phase]
            if real.size >= 2:
                means = (real + fake) / 2.0
                differences = fake - real
                agreement = bland_altman(real, fake)
                axis.scatter(
                    means, differences,
                    s=5 if level == "window" else 12,
                    alpha=0.11 if level == "window" else 0.68,
                    color=color, edgecolors="none", rasterized=True,
                )
                axis.axhline(agreement["bias"], color=color, linewidth=0.9)
                axis.axhline(agreement["lower_limit"], color=color, linestyle="--", linewidth=0.7)
                axis.axhline(agreement["upper_limit"], color=color, linestyle="--", linewidth=0.7)
            short_phase = "Raw" if phase == "unshifted" else "Aligned"
            if level == "window":
                title_counts.append(f"{short_phase}: n={real.size}, S={np.unique(subjects).size}")
            else:
                title_counts.append(f"{short_phase}: S={real.size}")
        axis.set_title(f"{DISPLAY_NAMES[parameter]}\n" + "\n".join(title_counts), fontsize=6.7)
        unit = UNITS.get(parameter, "normalized")
        axis.set_xlabel(f"Pair mean ({unit})")
        axis.set_ylabel(f"Generated - reference ({unit})")
        axis.grid(alpha=0.15, linewidth=0.4)
    axes.flat[-1].set_visible(False)
    legend = [
        Line2D([0], [0], marker="o", linestyle="-", markersize=4, linewidth=0.9,
               color=PHASE_STYLE[phase][1], label=PHASE_STYLE[phase][0])
        for phase in phases
    ]
    legend.extend([
        Line2D([0], [0], color="black", linewidth=0.9, label="Bias"),
        Line2D([0], [0], color="black", linewidth=0.7, linestyle="--", label="95% limits"),
    ])
    axes.flat[-1].legend(handles=legend, loc="center", frameon=False, fontsize=7.2)
    axes.flat[-1].set_visible(True)
    if level == "window":
        subtitle = "Window-level descriptive Bland–Altman (correlated repeated windows; all finite pairs)"
    else:
        subtitle = "Subject-level Bland–Altman (within-subject paired means)"
    figure.suptitle(f"{dataset} {model_label} ECG parameters — {subtitle}", fontsize=9)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    figure.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    inputs = {
        "MIMIC-AFib": args.mimic_dir.resolve(),
        "WESAD": args.wesad_dir.resolve(),
        "mmECG": args.mmecg_dir.resolve(),
    }
    for directory in inputs.values():
        protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        if protocol.get("status") != "completed":
            raise ValueError(f"input clinical artifact is not completed: {directory}")

    datasets = {
        "MIMIC-AFib": load_mimic(inputs["MIMIC-AFib"]),
        "WESAD": load_wesad(inputs["WESAD"]),
        "mmECG": load_mmecg(inputs["mmECG"], args.mmecg_reference_dir.resolve()),
    }
    all_summaries: list[dict[str, object]] = []
    filename_prefix = {"MIMIC-AFib": "mimic_afib", "WESAD": "wesad", "mmECG": "mmecg"}
    generated_files: list[Path] = []
    dataset_counts: dict[str, dict[str, object]] = {}
    for dataset, rows in datasets.items():
        identities = {(str(row["phase"]), str(row["window_id"])) for row in rows}
        if len(identities) != len(rows):
            raise ValueError(f"duplicate phase/window identities in {dataset}")
        views, summaries = compute_views(rows, dataset)
        all_summaries.extend(summaries)
        dataset_counts[dataset] = {
            "phase_rows": {phase: sum(row["phase"] == phase for row in rows) for phase in views["window"]},
            "subjects": len({str(row["subject_id"]) for row in rows}),
        }
        for level in ("window", "subject"):
            stem = output / f"{filename_prefix[dataset]}_{level}_level_bland_altman"
            _plot_dataset(dataset, views, level, stem)
            generated_files.extend([stem.with_suffix(".pdf"), stem.with_suffix(".png")])

    summary_csv = output / "bland_altman_level_summary.csv"
    _write_csv(summary_csv, all_summaries)
    generated_files.append(summary_csv)
    summary_json = output / "summary.json"
    summary_json.write_text(json.dumps({
        "status": "completed",
        "difference_definition": "generated_minus_reference",
        "window_level": "All finite paired windows; descriptive only because repeated windows within subjects are correlated.",
        "subject_level": "Reference and generated values are averaged within subject over phase-specific finite pairs before Bland-Altman analysis.",
        "dataset_counts": dataset_counts,
        "rows": all_summaries,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generated_files.append(summary_json)

    source_files = {
        "mimic_clinical_csv": inputs["MIMIC-AFib"] / "per_window_ecg_parameters.csv",
        "mimic_protocol": inputs["MIMIC-AFib"] / "protocol.json",
        "wesad_clinical_csv": inputs["WESAD"] / "per_window_ecg_parameters.csv",
        "wesad_protocol": inputs["WESAD"] / "protocol.json",
        "mmecg_clinical_csv": inputs["mmECG"] / "per_window_parameters.csv",
        "mmecg_protocol": inputs["mmECG"] / "protocol.json",
        "mmecg_identities": args.mmecg_reference_dir.resolve() / "paired_reference.npz",
    }
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "execution": {"python": platform.python_version(), "numpy": np.__version__, "script_sha256": _sha256(Path(__file__))},
        "method": {
            "window_level": "all phase-specific finite pairs; no downsampling; descriptive correlated repeated measurements",
            "subject_level": "arithmetic means of reference and generated values within subject over the same phase-specific finite pairs",
            "limits": "bias +/- 1.96 sample SD of generated-minus-reference differences",
            "pairing": "pairwise complete per parameter and phase; no cross-phase complete-case restriction",
        },
        "claim_boundary": "Window-level limits are descriptive and do not account for within-subject correlation. Subject-level plots have only 34 MIMIC-AFib, 15 WESAD, and 11 mmECG subject plotting units, under non-grouped random-window validation. Oracle alignment uses the paired target and is not deployable.",
        "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in source_files.items()},
        "outputs": {path.name: _sha256(path) for path in generated_files},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic_dir", type=Path, required=True)
    parser.add_argument("--wesad_dir", type=Path, required=True)
    parser.add_argument("--mmecg_dir", type=Path, required=True)
    parser.add_argument("--mmecg_reference_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
