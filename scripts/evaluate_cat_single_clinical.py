"""ECG-parameter and Bland-Altman analysis for single-output CAT endpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

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

from scripts.evaluate_cpsc_zscore_paired import _sha256
from scripts.evaluate_mimic_phase_clinical import (
    DISPLAY_NAMES, INTERVAL_PARAMETERS, PARAMETERS, PHASE_MODES,
    _agreement_record, _parameter_triplets, _waveform_summary,
)
from scripts.evaluate_mmecg_phase_clinical import (
    _configure_ieee_fonts, _measure_records, _subject_agreement,
)
from src.rcfm.metrics.bland_altman import bland_altman


EXPECTED = {"WESAD": 4213, "mmECG": 2877}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _plot(output: Path, dataset: str, model_prefix: str, model_label: str,
          pairs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]) -> list[Path]:
    colors = {"unshifted": "#6f7782", "oracle_aligned": "#c43d4b"}
    figure, axes = plt.subplots(3, 4, figsize=(7.16, 5.2), constrained_layout=True)
    for axis, parameter in zip(axes.flat, PARAMETERS):
        for phase in PHASE_MODES:
            reference, generated = pairs[(phase, parameter)]
            if len(reference) < 2:
                continue
            agreement = bland_altman(reference, generated)
            axis.scatter((reference + generated) / 2, generated - reference, s=2,
                         alpha=0.14, color=colors[phase], rasterized=True)
            axis.axhline(agreement["bias"], color=colors[phase], linewidth=0.8,
                         label="Raw" if phase == "unshifted" else "Oracle")
            axis.axhline(agreement["lower_limit"], color=colors[phase], linewidth=0.45, linestyle="--")
            axis.axhline(agreement["upper_limit"], color=colors[phase], linewidth=0.45, linestyle="--")
        unit = INTERVAL_PARAMETERS.get(parameter, "normalized")
        axis.set_title(DISPLAY_NAMES[parameter]); axis.set_xlabel(f"Pair mean ({unit})")
        axis.set_ylabel(f"Generated - real ({unit})"); axis.grid(alpha=0.15, linewidth=0.4)
    for axis in axes.flat[len(PARAMETERS):]: axis.set_visible(False)
    axes.flat[0].legend(frameon=False)
    figure.suptitle(f"{model_label} {dataset} ECG-parameter agreement", fontsize=8)
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"{model_prefix}_{dataset.lower()}_parameter_bland_altman.{suffix}"
        figure.savefig(path, dpi=600 if suffix == "png" else None); paths.append(path)
    plt.close(figure)
    return paths


def run(args: argparse.Namespace) -> Path:
    dataset = args.dataset
    expected = EXPECTED[dataset]
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True); _configure_ieee_fonts()
    with np.load(args.phase_predictions.resolve(), allow_pickle=False) as artifact:
        prefix = args.model_prefix
        required = {"targets", "subject_ids", f"{prefix}_unshifted_predictions", f"{prefix}_oracle_aligned_predictions"}
        if missing := sorted(required - set(artifact.files)):
            raise ValueError("CAT clinical artifact is missing: " + ", ".join(missing))
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        subjects = np.asarray(artifact["subject_ids"]).astype(str)
        predictions = {phase: np.asarray(artifact[f"{prefix}_{phase}_predictions"], dtype=np.float32)
                       for phase in PHASE_MODES}
    shape = (expected, 1, 480)
    if targets.shape != shape or subjects.shape != (expected,) or any(value.shape != shape for value in predictions.values()):
        raise ValueError("CAT clinical arrays violate the frozen shape contract")
    if not np.all(np.isfinite(targets)) or any(not np.all(np.isfinite(value)) for value in predictions.values()):
        raise FloatingPointError("CAT clinical arrays contain NaN or Inf")
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        real = _measure_records(targets, args.sampling_rate, executor)
        generated = {phase: _measure_records(values, args.sampling_rate, executor)
                     for phase, values in predictions.items()}
    finally:
        if executor is not None: executor.shutdown()
    window_rows, subject_rows, subject_detail, record_rows = [], [], [], []
    waveform, delineation, plot_pairs = {}, {}, {}
    for phase in PHASE_MODES:
        measured = generated[phase]
        wave = _waveform_summary(targets, predictions[phase]); pearson = wave.pop("pearson_per_record")
        waveform[phase] = wave
        delineation[phase] = {"real_success": int(sum(x["success"] for x in real)),
                              "generated_success": int(sum(x["success"] for x in measured)),
                              "paired_success": int(sum(a["success"] and b["success"] for a, b in zip(real, measured))),
                              "total": expected}
        for index, (a, b) in enumerate(zip(real, measured)):
            row = {"window": index, "subject_id": subjects[index], "phase_mode": phase,
                   "waveform_pearson_r": pearson[index], "real_delineation_success": a["success"],
                   "generated_delineation_success": b["success"], "real_failure_reason": a["failure_reason"],
                   "generated_failure_reason": b["failure_reason"], "real_hrv_status": a["hrv_status"],
                   "generated_hrv_status": b["hrv_status"]}
            for parameter in PARAMETERS:
                row[f"real_{parameter}"] = a["summary"].get(parameter)
                row[f"generated_{parameter}"] = b["summary"].get(parameter)
            record_rows.append(row)
    for parameter in PARAMETERS:
        paired, reference, before, after = _parameter_triplets(real, generated["unshifted"], generated["oracle_aligned"], parameter)
        for phase, values in (("unshifted", before), ("oracle_aligned", after)):
            window_rows.append(_agreement_record(prefix, phase, parameter, reference, values,
                inference_status="descriptive_only_windows_clustered_within_three_subjects"))
            summary, detail = _subject_agreement(subjects, paired, reference, values, prefix, phase, parameter)
            subject_rows.append(summary); subject_detail.extend(detail); plot_pairs[(phase, parameter)] = (reference, values)
    model_label = args.model_label or ("CAT-PPG (reproduced)" if dataset == "WESAD" else "CAT-RCG (adapted)")
    plot_paths = _plot(output, dataset, prefix, model_label, plot_pairs)
    paths = {"window": output / "window_parameter_agreement.csv",
             "subject": output / "subject_parameter_agreement.csv",
             "detail": output / "per_subject_parameter_means.csv",
             "records": output / "per_window_ecg_parameters.csv"}
    _write_csv(paths["window"], window_rows); _write_csv(paths["subject"], subject_rows)
    _write_csv(paths["detail"], subject_detail); _write_csv(paths["records"], record_rows)
    summary_path = output / "clinical_summary.json"
    summary_path.write_text(json.dumps({"schema_version": 1, "dataset": dataset,
        "model": model_label,
        "protocol": {"windows": expected, "subjects": sorted(set(subjects)), "support_samples": 480,
                     "raw_results_primary": True, "phase_correction": "target-informed oracle +/-16 samples",
                     "amplitude_unit": "normalized", "physical_amplitude_claim_allowed": False,
                     "hrv_status": "blocked_non_continuous_short_windows",
                     "subject_inference": "descriptive_only_extremely_underpowered_n3"},
        "waveform_agreement": waveform, "delineation": delineation,
        "window_parameter_agreement": window_rows, "subject_parameter_agreement": subject_rows},
        indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    outputs = [*paths.values(), summary_path, *plot_paths]
    protocol = {"schema_version": 1, "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": shlex.join(sys.argv), "input": {"path": str(args.phase_predictions.resolve()), "sha256": _sha256(args.phase_predictions)},
                "execution": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                              "neurokit2": getattr(nk, "__version__", "unknown"), "workers": args.workers},
                "outputs": {path.name: _sha256(path) for path in outputs}}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--phase_predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sampling_rate", type=float, default=128.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model_prefix", default="cat")
    parser.add_argument("--model_label")
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
