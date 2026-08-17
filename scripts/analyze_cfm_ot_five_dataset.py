"""Aggregate CFM/CFM+OT waveform, clinical, and paired-inference evidence."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.assemble_cfm_ot_pair import _paired_inference
from scripts.evaluate_cpsc_zscore_paired import _json, _sha256
from scripts.evaluate_ptbxl_clinical_agreement import PARAMETERS
from src.rcfm.metrics.statistics import holm_adjust


DATASETS = ("mimic_afib", "ptbxl", "cpsc2018", "wesad", "mmecg")
MODELS = ("cfm", "cfm_ot")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _waveform_rows(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows, inference = [], {}
    raw_p = {}
    for dataset in DATASETS:
        directory = root / dataset
        summary = json.loads((directory / "waveform_summary.json").read_text(encoding="utf-8"))
        for model in MODELS:
            values = summary["models"][model]
            rows.append({
                "dataset": dataset, "model": model, "phase_mode": "raw_full_window",
                "rmse": values["rmse"], "mae": values["mae"],
                "waveform_fd": values.get("waveform_fd_macro_lead", values.get("waveform_fd")),
                "pearson_median": values.get("per_record_pearson_median", values.get("per_record_pearson", {}).get("median")),
            })
            phase = summary.get("phase_sensitivity")
            if phase:
                aligned = phase[model]["oracle_aligned_fixed_support"]
                rows.append({
                    "dataset": dataset, "model": model, "phase_mode": "oracle_aligned_fixed_support",
                    "rmse": aligned["rmse"], "mae": aligned["mae"], "waveform_fd": aligned["waveform_fd"],
                    "pearson_median": aligned["per_record_pearson"]["median"],
                })
        paired = json.loads((directory / "paired_significance.json").read_text(encoding="utf-8"))
        inference[dataset] = paired
        if dataset in {"ptbxl", "cpsc2018"}:
            for metric, result in paired["comparisons"].items():
                if result.get("raw_p_value") is not None:
                    raw_p[f"{dataset}/{metric}"] = float(result["raw_p_value"])
    adjusted = holm_adjust(raw_p)
    for name, value in adjusted.items():
        dataset, metric = name.split("/")
        inference[dataset]["comparisons"][metric]["holm_adjusted_p_value_6_tests"] = value
    return rows, inference


def _clinical_error_map(path: Path) -> dict[str, dict[tuple[str, ...], dict[str, float]]]:
    output: dict[str, dict[tuple[str, ...], dict[str, float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            if model not in MODELS:
                continue
            key_fields = [row["record_id"]]
            if "patient_id" in row:
                key_fields.append(row["patient_id"])
            key_fields.append(row["lead"])
            key = tuple(key_fields)
            values = {}
            for parameter in PARAMETERS:
                try:
                    real = float(row[f"real_{parameter}"])
                    generated = float(row[f"generated_{parameter}"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(real) and np.isfinite(generated):
                    values[parameter] = abs(generated - real)
            output.setdefault(model, {})[key] = values
    return output


def _clinical_significance(path: Path, dataset: str, seed: int, replicates: int) -> dict[str, object]:
    errors = _clinical_error_map(path)
    results = {}
    for parameter_index, parameter in enumerate(PARAMETERS):
        common = sorted(set(errors["cfm"]) & set(errors["cfm_ot"]))
        common = [key for key in common if parameter in errors["cfm"][key] and parameter in errors["cfm_ot"][key]]
        cfm = np.asarray([errors["cfm"][key][parameter] for key in common], dtype=np.float64)
        cfm_ot = np.asarray([errors["cfm_ot"][key][parameter] for key in common], dtype=np.float64)
        identities = np.asarray([key[1] if dataset == "ptbxl" else key[0] for key in common])
        result = _paired_inference(cfm_ot, cfm, identities, seed + parameter_index, replicates, "confirmatory_eligible")
        result.update({"parameter": parameter, "complete_case_record_lead_cells": len(common),
                       "estimand": "absolute_clinical_error_cfm_ot_minus_cfm",
                       "identity_caveat": "patient_level" if dataset == "ptbxl" else "record_level_patient_ids_unavailable"})
        results[parameter] = result
    return {"dataset": dataset, "comparisons": results}


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _clinical_summary_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    sources = (
        ("ptbxl", "raw", "patient_macro_11_leads", args.ptbxl_clinical / "macro_lead_agreement.csv"),
        ("cpsc2018", "raw", "record_macro_11_leads", args.cpsc_clinical / "macro_lead_agreement.csv"),
        ("mimic_afib", None, "record", args.mimic_clinical / "ecg_parameter_agreement_summary.csv"),
        ("wesad", None, "equal_subject_n3", args.wesad_clinical / "subject_parameter_agreement.csv"),
        ("mmecg", None, "equal_subject_n3", args.mmecg_clinical / "subject_parameter_agreement.csv"),
    )
    output = []
    for dataset, phase, aggregation, path in sources:
        for row in _read_rows(path):
            if row.get("model") not in MODELS or row.get("status") != "ok":
                continue
            output.append({"dataset": dataset, "model": row["model"],
                           "phase_mode": phase or row.get("phase_mode"), "aggregation": aggregation,
                           "parameter": row["parameter"], "unit": row.get("unit"),
                           "n": row.get("n_patients", row.get("n_records", row.get("n"))),
                           "mae": row.get("mae"), "rmse": row.get("rmse"), "pearson_r": row.get("pearson_r"),
                           "ba_bias": row.get("ba_bias", row.get("bland_altman_bias")),
                           "ba_lower": row.get("ba_lower", row.get("bland_altman_lower_limit")),
                           "ba_upper": row.get("ba_upper", row.get("bland_altman_upper_limit"))})
    return output


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    waveform_rows, waveform_inference = _waveform_rows(args.evaluation_root.resolve())
    waveform_path = output / "waveform_results.csv"; _write_csv(waveform_path, waveform_rows)
    waveform_sig_path = output / "waveform_paired_significance.json"; _json(waveform_sig_path, waveform_inference)
    clinical_rows = _clinical_summary_rows(args)
    clinical_path = output / "clinical_parameter_results.csv"; _write_csv(clinical_path, clinical_rows)
    clinical_inference = {
        "ptbxl": _clinical_significance(args.ptbxl_clinical / "per_record_parameters.csv", "ptbxl", 5051, args.bootstrap_replicates),
        "cpsc2018": _clinical_significance(args.cpsc_clinical / "per_record_parameters.csv", "cpsc2018", 5151, args.bootstrap_replicates),
        "mimic_afib": {"status": "blocked_identity_unavailable"},
        "wesad": {"status": "descriptive_only_extremely_underpowered_n3"},
        "mmecg": {"status": "descriptive_only_extremely_underpowered_n3"},
    }
    raw_p = {f"{dataset}/{parameter}": result["raw_p_value"]
             for dataset in ("ptbxl", "cpsc2018")
             for parameter, result in clinical_inference[dataset]["comparisons"].items()}
    for name, value in holm_adjust(raw_p).items():
        dataset, parameter = name.split("/")
        clinical_inference[dataset]["comparisons"][parameter]["holm_adjusted_p_value_22_tests"] = value
    clinical_sig_path = output / "clinical_paired_significance.json"; _json(clinical_sig_path, clinical_inference)
    hrv_path = output / "wesad_hrv_results.csv"
    _write_csv(hrv_path, [row for row in _read_rows(args.wesad_clinical / "hrv_agreement.csv") if row["model"] in MODELS])
    sources = [args.evaluation_root / dataset / "protocol.json" for dataset in DATASETS]
    sources += [args.ptbxl_clinical / "protocol.json", args.cpsc_clinical / "protocol.json",
                args.mimic_clinical / "protocol.json", args.wesad_clinical / "protocol.json",
                args.mmecg_clinical / "protocol.json"]
    outputs = (waveform_path, waveform_sig_path, clinical_path, clinical_sig_path, hrv_path)
    protocol = {"schema_version": 1, "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": shlex.join(sys.argv), "bootstrap_replicates": args.bootstrap_replicates,
                "multiplicity": {"waveform": "Holm across PTB-XL/CPSC x RMSE/MAE/Pearson (6 tests)",
                                 "clinical": "Holm across PTB-XL/CPSC x 11 parameters (22 tests)"},
                "blocked_inference": {"mimic_afib": "identity unavailable", "wesad": "n=3 subjects",
                                      "mmecg": "n=3 subjects", "waveform_fd": "full-distribution metric"},
                "sources": {str(path): _sha256(path) for path in sources},
                "outputs": {path.name: _sha256(path) for path in outputs},
                "software": {"python": platform.python_version(), "numpy": np.__version__}}
    _json(output / "protocol.json", protocol)
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation_root", type=Path, required=True)
    parser.add_argument("--ptbxl_clinical", type=Path, required=True)
    parser.add_argument("--cpsc_clinical", type=Path, required=True)
    parser.add_argument("--mimic_clinical", type=Path, required=True)
    parser.add_argument("--wesad_clinical", type=Path, required=True)
    parser.add_argument("--mmecg_clinical", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    return parser


if __name__ == "__main__":
    result = run(build_argparser().parse_args())
    print(f"Five-dataset CFM+OT analysis saved to {result}")
