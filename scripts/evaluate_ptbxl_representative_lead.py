"""Evaluate all ECG parameters for one PTB-XL representative target lead."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import neurokit2 as nk
import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _sha256
from scripts.evaluate_ptbxl_clinical_agreement import (
    MODEL_LABELS,
    PARAMETERS,
    _agreement_row,
    _inverse_oracle_minmax,
    _load_p_wave_applicability,
    _measure_records,
    _patient_parameter_pairs,
    _plot_bland_altman,
    _valid_source_protocol,
)
from scripts.evaluate_ptbxl_fourway import TARGET_INDICES, TARGET_LEADS


def run(args: argparse.Namespace) -> Path:
    input_dir = args.input_dir.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.lead not in TARGET_LEADS:
        raise ValueError(f"representative lead must be one of {TARGET_LEADS}")
    unsupported = [model for model in args.models if model not in MODEL_LABELS]
    if unsupported:
        raise ValueError(f"unsupported models: {unsupported}")

    source_protocol_path = input_dir / ("summary.json" if args.legacy_single_target else "protocol.json")
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    if args.legacy_single_target:
        if not (
            source_protocol.get("protocol") == "legacy_ptbxl_single_lead_cfm_v1"
            and source_protocol.get("split_source") == "official_fold10"
            and source_protocol.get("target_lead") == args.lead
            and str(args.epoch) in source_protocol.get("epochs", {})
        ):
            raise ValueError("legacy representative-lead analysis requires completed fold-10 predictions")
    elif not _valid_source_protocol(source_protocol, None):
        raise ValueError("representative-lead analysis requires completed raw predictions")
    with np.load(input_dir / "paired_reference.npz", allow_pickle=False) as artifact:
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        patient_ids = np.asarray(
            artifact["patient_ids"]
            if "patient_ids" in artifact.files
            else artifact["subject_ids"]
        )
        record_ids = (
            np.asarray(artifact["record_ids"])
            if "record_ids" in artifact.files
            else np.asarray(np.load(data_dir / "record_ids_test.npy", allow_pickle=False))
        )

    lead_position = 0 if args.legacy_single_target else TARGET_LEADS.index(args.lead)
    source_lead_index = TARGET_INDICES[lead_position]
    if args.legacy_single_target:
        source_lead_index = TARGET_INDICES[TARGET_LEADS.index(args.lead)]
    minima = np.load(data_dir / "record_minima_test.npy", allow_pickle=False)[:, source_lead_index]
    ranges = np.load(data_dir / "record_ranges_test.npy", allow_pickle=False)[:, source_lead_index]
    physical_target = _inverse_oracle_minmax(
        targets[:, lead_position : lead_position + 1], minima[:, None], ranges[:, None]
    )[:, 0]
    source = np.load(data_dir / "X_test_resampled.npy", mmap_mode="r")[: len(targets), :512, source_lead_index]
    if not np.allclose(physical_target, source, atol=2e-6, rtol=2e-6):
        raise ValueError(f"{args.lead} inverse transform does not reproduce stored PTB-XL mV waveforms")
    physical_predictions = {}
    for model in args.models:
        prediction_name = (
            f"predictions_epoch_{args.epoch}.npy"
            if args.legacy_single_target
            else f"{model}_predictions.npy"
        )
        prediction = np.asarray(
            np.load(input_dir / prediction_name, mmap_mode="r")[:, lead_position : lead_position + 1]
        )
        physical_predictions[model] = _inverse_oracle_minmax(
            prediction, minima[:, None], ranges[:, None]
        )[:, 0]

    p_wave_applicable, p_wave_policy = _load_p_wave_applicability(
        args.metadata_csv.resolve(), record_ids
    )
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        print(f"clinical delineation: real lead {args.lead}", flush=True)
        real = _measure_records(physical_target, args.sampling_rate, p_wave_applicable, executor)
        rows = []
        pairs = {}
        delineation = {"real_success": int(sum(item["success"] for item in real))}
        outputs = []
        for model in args.models:
            print(f"clinical delineation: {model} lead {args.lead}", flush=True)
            generated = _measure_records(
                physical_predictions[model], args.sampling_rate, p_wave_applicable, executor
            )
            delineation[f"{model}_success"] = int(sum(item["success"] for item in generated))
            for parameter in PARAMETERS:
                _, real_values, generated_values = _patient_parameter_pairs(
                    real, generated, patient_ids, parameter
                )
                pairs[(model, parameter)] = (real_values, generated_values)
                rows.append(
                    _agreement_row(model, args.lead, parameter, real_values, generated_values)
                )
            outputs.extend(
                _plot_bland_altman(
                    pairs,
                    model,
                    output_dir / f"{model}_lead_{args.lead.lower()}_ecg_parameter_bland_altman",
                    lead=args.lead,
                    parameters=PARAMETERS,
                )
            )
    finally:
        if executor is not None:
            executor.shutdown()

    table_path = output_dir / f"lead_{args.lead.lower()}_patient_agreement.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(table_path)

    protocol = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "representative_lead": args.lead,
        "records": int(len(targets)),
        "patients": int(len(np.unique(patient_ids))),
        "models": list(args.models),
        "parameters": list(PARAMETERS),
        "sampling_rate_hz": args.sampling_rate,
        "alignment": "raw synchronized no phase correction",
        "delineation": "independent NeuroKit2 DWT per record for representative lead",
        "delineation_success": delineation,
        "aggregation": "mean paired records within patient before agreement",
        "difference_definition": "generated_minus_reference",
        "amplitude_unit": "mV",
        "generated_inverse": "oracle ground-truth target min/range",
        "p_wave_policy": p_wave_policy,
        "hrv": "blocked_four_second_records",
        "source_protocol_sha256": _sha256(source_protocol_path),
        "source_predictions": {
            model: _sha256(
                input_dir
                / (
                    f"predictions_epoch_{args.epoch}.npy"
                    if args.legacy_single_target
                    else f"{model}_predictions.npy"
                )
            )
            for model in args.models
        },
        "claim_boundary": (
            f"{args.lead} patient-level descriptive analysis; amplitudes use oracle target scalers; "
            "HRV blocked; automatic DWT and one training seed do not support population inference."
        ),
        "figure_style": {
            "font_family": "Liberation Serif (Times-compatible)",
            "pdf_fonttype": 42,
            "png_dpi": 600,
            "width_inches": 7.16,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "neurokit2": getattr(nk, "__version__", "unknown"),
        },
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--lead", default="V3")
    parser.add_argument("--sampling_rate", type=float, default=128.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--models", nargs="+", default=["cfm", "rcfm", "rcfm_ot", "rddm"])
    parser.add_argument("--legacy_single_target", action="store_true")
    parser.add_argument("--epoch", type=int, default=999)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
