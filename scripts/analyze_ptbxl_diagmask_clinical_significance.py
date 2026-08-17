"""Patient-paired clinical-error inference for the PTB-XL DiagMask ablation."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.assemble_ptbxl_diagmask_sixway import PRIMARY_COMPARISONS, _paired_test
from scripts.evaluate_cpsc_zscore_paired import _json, _sha256
from scripts.evaluate_ptbxl_clinical_agreement import PARAMETERS, TARGET_LEADS
from src.rcfm.metrics.statistics import holm_adjust


def _resolve_comparisons(protocol: dict[str, object]) -> tuple[tuple[str, str], ...]:
    configured = protocol.get("analysis_comparisons")
    if configured is None:
        return PRIMARY_COMPARISONS
    comparisons = tuple(tuple(item) for item in configured)
    if not comparisons or any(len(pair) != 2 or pair[0] == pair[1] for pair in comparisons):
        raise ValueError("clinical protocol has invalid analysis comparisons")
    return comparisons


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def _load_errors(path: Path) -> dict[str, dict[tuple[str, str, str], dict[str, float]]]:
    output: dict[str, dict[tuple[str, str, str], dict[str, float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            key = (row["record_id"], row["patient_id"], row["lead"])
            if key in output.setdefault(model, {}):
                raise ValueError(f"duplicate clinical row for {model} {key}")
            values = {}
            for parameter in PARAMETERS:
                real = _float(row.get(f"real_{parameter}"))
                generated = _float(row.get(f"generated_{parameter}"))
                if real is not None and generated is not None:
                    values[parameter] = abs(generated - real)
            output[model][key] = values
    return output


def _paired_complete_case(
    errors: dict[str, dict[tuple[str, str, str], dict[str, float]]],
    comparison: str,
    reference: str,
    parameter: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    keys = sorted(set(errors[comparison]) & set(errors[reference]))
    comparison_values, reference_values, patient_ids = [], [], []
    for key in keys:
        comparison_value = errors[comparison][key].get(parameter)
        reference_value = errors[reference][key].get(parameter)
        if comparison_value is None or reference_value is None:
            continue
        comparison_values.append(comparison_value)
        reference_values.append(reference_value)
        patient_ids.append(key[1])
    return (
        np.asarray(comparison_values, dtype=np.float64),
        np.asarray(reference_values, dtype=np.float64),
        np.asarray(patient_ids),
        len(comparison_values),
    )


def run(args: argparse.Namespace) -> Path:
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((input_dir / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "completed" or protocol.get("representative_bland_altman_lead") != "V3":
        raise ValueError("clinical significance requires a completed V3 artifact")
    primary_comparisons = _resolve_comparisons(protocol)
    errors = _load_errors(input_dir / "per_record_parameters.csv")
    required_models = {model for pair in primary_comparisons for model in pair}
    if not required_models.issubset(errors):
        raise ValueError("clinical artifact is missing a pre-registered model")
    comparisons = {}
    raw_p = {}
    flat_rows = []
    for pair_index, (comparison, reference) in enumerate(primary_comparisons):
        pair_name = f"{comparison}_vs_{reference}"
        comparisons[pair_name] = {}
        for parameter_index, parameter in enumerate(PARAMETERS):
            comparison_values, reference_values, patient_ids, cells = _paired_complete_case(
                errors, comparison, reference, parameter
            )
            result = _paired_test(
                comparison_values,
                reference_values,
                patient_ids,
                args.bootstrap_seed + pair_index * 100 + parameter_index,
                args.bootstrap_replicates,
            )
            result.update(
                {
                    "parameter": parameter,
                    "complete_case_record_lead_cells": cells,
                    "lead_scope": list(TARGET_LEADS),
                    "estimand": "absolute_parameter_error_macro_over_complete_case_record_lead_cells_within_patient",
                    "favorable_direction": "negative_comparison_minus_reference",
                }
            )
            comparisons[pair_name][parameter] = result
            raw_p[f"{pair_name}/{parameter}"] = float(result["raw_p_value"])
    adjusted = holm_adjust(raw_p)
    test_count = len(raw_p)
    for name, value in adjusted.items():
        pair_name, parameter = name.split("/")
        result = comparisons[pair_name][parameter]
        result[f"holm_adjusted_p_value_{test_count}_tests"] = value
        flat_rows.append(
            {
                "comparison": pair_name,
                "parameter": parameter,
                "patients": result["patients"],
                "complete_case_record_lead_cells": result["complete_case_record_lead_cells"],
                "mean_absolute_error_difference": result["mean_difference"],
                "paired_difference_std": result["paired_difference_std"],
                "ci_lower": result["bootstrap_95_ci"][0],
                "ci_upper": result["bootstrap_95_ci"][1],
                "raw_p_value": result["raw_p_value"],
                "holm_adjusted_p_value": value,
                "rank_biserial": result["effect_size"]["value"],
            }
        )
    result_path = output_dir / "clinical_patient_paired_significance.json"
    _json(
        result_path,
        {
            "schema_version": 1,
            "comparisons": comparisons,
            "multiplicity": f"Holm adjustment jointly across {len(primary_comparisons)} pre-registered comparisons x 11 clinical parameters",
            "missingness": "pairwise complete same-record same-lead cells; no imputation",
            "claim_boundary": "One training seed and automatic DWT: p-values describe fold-10 patient sampling uncertainty only.",
        },
    )
    csv_path = output_dir / "clinical_patient_paired_significance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(flat_rows, key=lambda row: (row["comparison"], row["parameter"])))
    output_protocol = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "source_clinical_protocol_sha256": _sha256(input_dir / "protocol.json"),
        "source_parameters_sha256": _sha256(input_dir / "per_record_parameters.csv"),
        "bootstrap_replicates": args.bootstrap_replicates,
        "outputs": {result_path.name: _sha256(result_path), csv_path.name: _sha256(csv_path)},
    }
    _json(output_dir / "protocol.json", output_protocol)
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_seed", type=int, default=3031)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    return parser


if __name__ == "__main__":
    output = run(build_argparser().parse_args())
    print(f"PTB-XL clinical significance saved to {output}")
