"""Replot one PTB-XL representative lead from cached patient-safe clinical rows."""

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
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _sha256
from scripts.evaluate_ptbxl_clinical_agreement import (
    MODEL_LABELS,
    PARAMETERS,
    _agreement_row,
    _plot_bland_altman,
)
from scripts.evaluate_ptbxl_fourway import TARGET_LEADS


def _cached_parameters(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        parameter
        for parameter in PARAMETERS
        if f"real_{parameter}" in frame.columns and f"generated_{parameter}" in frame.columns
    )


def _patient_pairs(
    frame: pd.DataFrame, model: str, lead: str, parameter: str
) -> tuple[np.ndarray, np.ndarray]:
    subset = frame.loc[
        (frame["model"] == model) & (frame["lead"] == lead),
        ["patient_id", f"real_{parameter}", f"generated_{parameter}"],
    ].dropna()
    grouped = subset.groupby("patient_id", sort=True)[
        [f"real_{parameter}", f"generated_{parameter}"]
    ].mean()
    return (
        grouped[f"real_{parameter}"].to_numpy(dtype=np.float64),
        grouped[f"generated_{parameter}"].to_numpy(dtype=np.float64),
    )


def run(args: argparse.Namespace) -> Path:
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.lead not in TARGET_LEADS:
        raise ValueError(f"representative lead must be one of {TARGET_LEADS}")

    source_protocol_path = source_dir / "protocol.json"
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    if source_protocol.get("status") != "completed":
        raise ValueError("representative-lead plot requires a completed clinical source")
    source_csv = source_dir / "per_record_parameters.csv"
    frame = pd.read_csv(source_csv)
    parameters = _cached_parameters(frame)
    if not parameters:
        raise ValueError("source clinical rows contain no cached ECG parameters")

    rows = []
    outputs = []
    missing = [parameter for parameter in PARAMETERS if parameter not in parameters]
    for model in args.models:
        if model not in MODEL_LABELS:
            raise ValueError(f"unsupported model: {model}")
        pairs = {}
        for parameter in parameters:
            real, generated = _patient_pairs(frame, model, args.lead, parameter)
            pairs[(model, parameter)] = (real, generated)
            rows.append(_agreement_row(model, args.lead, parameter, real, generated))
        outputs.extend(
            _plot_bland_altman(
                pairs,
                model,
                output_dir / f"{model}_lead_{args.lead.lower()}_ecg_parameter_bland_altman",
                lead=args.lead,
                parameters=parameters,
            )
        )

    table_path = output_dir / f"lead_{args.lead.lower()}_patient_agreement.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(table_path)

    protocol = {
        "schema_version": 1,
        "status": "completed_cached_replot",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "representative_lead": args.lead,
        "models": list(args.models),
        "parameters": list(parameters),
        "missing_cached_parameters": missing,
        "aggregation": "mean paired records within patient before agreement",
        "difference_definition": "generated_minus_reference",
        "source_protocol_sha256": _sha256(source_protocol_path),
        "source_per_record_parameters_sha256": _sha256(source_csv),
        "claim_boundary": (
            "Cached V3 patient-level descriptive replot. The source predates missing parameters "
            "listed in missing_cached_parameters; no values were imputed. Amplitudes use oracle target scalers."
        ),
        "figure_style": {
            "font_family": "Liberation Serif (Times-compatible)",
            "pdf_fonttype": 42,
            "png_dpi": 600,
            "width_inches": 7.16,
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--lead", default="V3")
    parser.add_argument("--models", nargs="+", default=["cfm", "rcfm", "rcfm_ot", "rddm"])
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
