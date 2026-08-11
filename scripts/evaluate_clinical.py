"""Evaluate paired real/generated ECG clinical parameters by subject."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path

import neurokit2 as nk
import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.metrics.bland_altman import bland_altman, paired_correlation
from src.rcfm.metrics.clinical import delineate_ecg, measure_ecg_parameters


PARAMETERS = (
    "rr_ms",
    "pr_ms",
    "qrs_ms",
    "qt_ms",
    "qtc_ms",
    "p_amplitude",
    "r_amplitude",
    "t_amplitude",
    "st_deviation",
    "sdnn_ms",
    "rmssd_ms",
)


def _records(path: Path) -> list[np.ndarray]:
    values = np.load(path, allow_pickle=True)
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0]
    return [np.asarray(record, dtype=np.float64).reshape(-1) for record in values]


def _record_summary(measurement: dict[str, object]) -> dict[str, float]:
    summary: dict[str, float] = {}
    parameters = measurement["parameters"]
    for name, values in parameters.items():
        if values:
            summary[name] = float(np.mean(values))
    hrv = measurement["hrv"]
    if hrv["status"] == "ok":
        summary["sdnn_ms"] = float(hrv["sdnn_ms"])
        summary["rmssd_ms"] = float(hrv["rmssd_ms"])
    return summary


def _subject_summaries(records: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        for parameter, value in record["summary"].items():
            grouped[str(record["subject_id"])][parameter].append(float(value))
    return {
        subject: {parameter: float(np.mean(values)) for parameter, values in parameters.items()}
        for subject, parameters in grouped.items()
    }


def _paired_records(
    measured_by_index: dict[str, dict[int, dict[str, object]]],
) -> tuple[list[int], dict[str, list[dict[str, object]]]]:
    paired_indices = sorted(
        set(measured_by_index["real"]) & set(measured_by_index["generated"])
    )
    return paired_indices, {
        kind: [measured_by_index[kind][index] for index in paired_indices]
        for kind in ("real", "generated")
    }


def _agreement(
    real_subjects: dict[str, dict[str, float]],
    generated_subjects: dict[str, dict[str, float]],
    analysis_unit: str = "subject",
) -> dict[str, object]:
    common_subjects = sorted(set(real_subjects) & set(generated_subjects))
    output: dict[str, object] = {}
    for parameter in PARAMETERS:
        pairs = [
            (real_subjects[subject].get(parameter), generated_subjects[subject].get(parameter))
            for subject in common_subjects
        ]
        pairs = [(real, generated) for real, generated in pairs if real is not None and generated is not None]
        if len(pairs) < 2:
            output[parameter] = {
                "status": f"insufficient_{analysis_unit}_pairs",
                "n": len(pairs),
            }
            continue
        real_values, generated_values = zip(*pairs)
        output[parameter] = {
            "status": "ok",
            "correlation": paired_correlation(real_values, generated_values),
            "bland_altman": bland_altman(real_values, generated_values),
        }
    return output


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    real = _records(args.real)
    generated = _records(args.generated)
    subject_ids = np.load(args.subject_ids, allow_pickle=False).reshape(-1)
    if not (len(real) == len(generated) == len(subject_ids)):
        raise ValueError("real, generated, and subject_ids must have the same record count")
    if any(len(a) != len(b) for a, b in zip(real, generated)):
        raise ValueError("each real/generated record pair must have the same length")
    applicability_path = getattr(args, "p_wave_applicability", None)
    if applicability_path is None:
        p_wave_applicability = np.ones(len(real), dtype=bool)
    else:
        p_wave_applicability = np.asarray(
            np.load(applicability_path, allow_pickle=False), dtype=bool
        ).reshape(-1)
        if len(p_wave_applicability) != len(real):
            raise ValueError("p-wave applicability must align with ECG records")

    measured_by_index: dict[str, dict[int, dict[str, object]]] = {
        "real": {},
        "generated": {},
    }
    failures: dict[str, Counter[str]] = {"real": Counter(), "generated": Counter()}
    statuses: dict[str, dict[str, Counter[str]]] = {
        "real": {
            "p_wave": Counter(),
            "amplitude": Counter(),
            "hrv": Counter(),
        },
        "generated": {
            "p_wave": Counter(),
            "amplitude": Counter(),
            "hrv": Counter(),
        },
    }
    for kind, signals in (("real", real), ("generated", generated)):
        for record_index, (subject_id, signal) in enumerate(zip(subject_ids, signals)):
            delineation = delineate_ecg(
                signal,
                sampling_rate=args.sampling_rate,
                method=args.delineation_method,
                clean_method=args.clean_method,
            )
            if not delineation.success:
                failures[kind][str(delineation.failure_reason)] += 1
                continue
            measurement = measure_ecg_parameters(
                signal,
                sampling_rate=args.sampling_rate,
                fiducials=delineation.fiducials,
                amplitude_unit=args.unit,
                inverse_transformed=args.inverse_transformed,
                qtc_formula=args.qtc_formula,
                st_offset_ms=args.st_offset_ms,
                continuous=args.continuous,
                minimum_hrv_duration_seconds=args.minimum_hrv_seconds,
                p_wave_applicable=bool(p_wave_applicability[record_index]),
                allow_normalized_amplitudes=getattr(
                    args, "allow_normalized_amplitudes", False
                ),
            )
            statuses[kind]["p_wave"][str(measurement["p_wave_status"])] += 1
            statuses[kind]["amplitude"][str(measurement["amplitude_status"])] += 1
            statuses[kind]["hrv"][str(measurement["hrv"]["status"])] += 1
            measured_by_index[kind][record_index] = {
                "subject_id": str(subject_id),
                "summary": _record_summary(measurement),
                "p_wave_status": measurement["p_wave_status"],
                "amplitude_status": measurement["amplitude_status"],
                "hrv_status": measurement["hrv"]["status"],
            }

    independent_records = {
        kind: list(records.values()) for kind, records in measured_by_index.items()
    }
    paired_indices, paired_records = _paired_records(measured_by_index)
    subject_summaries = {
        kind: _subject_summaries(records) for kind, records in paired_records.items()
    }
    independent_subject_summaries = {
        kind: _subject_summaries(records) for kind, records in independent_records.items()
    }
    total = len(real)
    delineation_summary = {
        kind: {
            "total_records": total,
            "successful_records": len(measured_by_index[kind]),
            "failed_records": total - len(measured_by_index[kind]),
            "success_rate": len(measured_by_index[kind]) / total if total else 0.0,
            "failure_reasons": dict(failures[kind]),
        }
        for kind in ("real", "generated")
    }
    analysis_unit = getattr(args, "analysis_unit", "subject")
    if analysis_unit not in {"record", "subject"}:
        raise ValueError("analysis_unit must be record or subject")
    return {
        "schema_version": 3,
        "metadata": {
            "sampling_rate_hz": args.sampling_rate,
            "lead": args.lead,
            "analysis_unit": analysis_unit,
            "normalization_id": getattr(args, "normalization_id", None),
            "unit": args.unit,
            "inverse_transformed": args.inverse_transformed,
            "continuous_records": args.continuous,
            "minimum_hrv_seconds": args.minimum_hrv_seconds,
            "qtc_formula": args.qtc_formula,
            "st_offset_ms": args.st_offset_ms,
            "st_reference": "pre-QRS isoelectric median",
            "r_amplitude_definition": "signed R-peak value minus pre-QRS isoelectric median",
            "amplitude_claim_policy": (
                "exploratory normalized morphology only; not physical ECG amplitude"
                if getattr(args, "allow_normalized_amplitudes", False)
                else "physical amplitude blocked unless inverse-transformed units are verified"
            ),
            "delineation_algorithm": f"neurokit2.ecg_delineate:{args.delineation_method}",
            "clean_method": args.clean_method,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "neurokit2": getattr(nk, "__version__", "unknown"),
            },
            "command": shlex.join(sys.argv),
        },
        "delineation": delineation_summary,
        "measurement_status_counts": {
            kind: {
                status_type: dict(counter)
                for status_type, counter in kind_statuses.items()
            }
            for kind, kind_statuses in statuses.items()
        },
        "usable_units": {
            "real": len(subject_summaries["real"]),
            "generated": len(subject_summaries["generated"]),
        },
        "pairing": {
            "policy": "same_record_requires_independent_real_and_generated_delineation_success",
            "paired_records": len(paired_indices),
            "paired_units": len(set(str(subject_ids[index]) for index in paired_indices)),
            "analysis_unit": analysis_unit,
        },
        "independent_usable_units": {
            kind: len(summaries) for kind, summaries in independent_subject_summaries.items()
        },
        "unit_summaries": subject_summaries,
        "agreement": _agreement(
            subject_summaries["real"],
            subject_summaries["generated"],
            analysis_unit=analysis_unit,
        ),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--subject_ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling_rate", type=float, required=True)
    parser.add_argument("--lead", required=True)
    parser.add_argument("--analysis_unit", choices=["record", "subject"], default="subject")
    parser.add_argument("--normalization_id", default=None)
    parser.add_argument("--p_wave_applicability", type=Path, default=None)
    parser.add_argument("--unit", choices=["normalized", "mV", "uV"], default="normalized")
    parser.add_argument("--inverse_transformed", action="store_true")
    parser.add_argument(
        "--allow_normalized_amplitudes",
        action="store_true",
        help=(
            "Compute explicitly exploratory normalized-domain P/R/T/ST morphology values. "
            "These values are not physical amplitudes and cannot support mV claims."
        ),
    )
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--minimum_hrv_seconds", type=float, default=30.0)
    parser.add_argument(
        "--qtc_formula",
        choices=["bazett", "fridericia", "framingham", "hodges"],
        default="fridericia",
    )
    parser.add_argument("--st_offset_ms", type=float, default=60.0)
    parser.add_argument("--delineation_method", choices=["dwt", "cwt", "peak"], default="dwt")
    parser.add_argument("--clean_method", default="neurokit")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"clinical evaluation saved: records={result['delineation']['real']['total_records']} "
        f"real_success={result['delineation']['real']['success_rate']:.3f} "
        f"generated_success={result['delineation']['generated']['success_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
