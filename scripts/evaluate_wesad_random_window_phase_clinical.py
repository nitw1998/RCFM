#!/usr/bin/env python3
"""ECG parameters and Bland--Altman analysis for phase-corrected WESAD models."""

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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_wesad_record_minmax_clinical_mpl")

import neurokit2 as nk
import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _sha256
from scripts.evaluate_mimic_phase_clinical import (
    PARAMETERS,
    PHASE_MODES,
    _agreement_record,
    _measure_records,
    _parameter_triplets,
    _plot_bland_altman,
    _waveform_summary,
)


EXPECTED_WINDOWS = 4342
EXPECTED_SHAPE = (EXPECTED_WINDOWS, 1, 480)
EXPECTED_VERSION = (
    "wesad-all-windows-random80-20-subject-overlap-linear-resample-source-record-minmax-v2"
)
EXPECTED_NORMALIZATION = "source_record_minmax_neg1_1_v1"
CLINICAL_SPECS = {
    "wesad_record_minmax": {
        "dataset": "WESAD", "windows": 4342, "sampling_rate": 128.0,
        "dataset_variants": {"record_minmax", "wesad_record_minmax"},
        "dataset_version": EXPECTED_VERSION, "normalization_id": EXPECTED_NORMALIZATION,
        "labels_required": True, "amplitude_unit": "source-record-normalized units",
        "inference": "descriptive single-training-seed n15",
    },
    "mmecg": {
        "dataset": "mmECG", "windows": 2494, "sampling_rate": 200.0,
        "dataset_variants": {"mmecg"},
        "dataset_version": "mmecg-all-windows-random80-20-subject-record-overlap-window-minmax-v1",
        "normalization_id": "window_minmax_neg1_1_v1",
        "labels_required": False, "amplitude_unit": "window-normalized units",
        "inference": "descriptive single-training-seed n11 with overlapping windows",
    },
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def subject_agreement(
    subjects: np.ndarray,
    paired_rows: np.ndarray,
    reference: np.ndarray,
    generated: np.ndarray,
    phase_mode: str,
    parameter: str,
    model: str = "cfm",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Average paired windows within subject before agreement calculations."""

    detail: list[dict[str, object]] = []
    for subject in sorted(set(subjects)):
        take = np.asarray(
            [index for index, row in enumerate(paired_rows) if subjects[row] == subject],
            dtype=np.int64,
        )
        if not len(take):
            continue
        real_mean = float(np.mean(reference[take]))
        generated_mean = float(np.mean(generated[take]))
        detail.append({
            "model": model, "phase_mode": phase_mode, "parameter": parameter,
            "subject_id": subject, "reference_mean": real_mean,
            "generated_mean": generated_mean, "difference": generated_mean - real_mean,
            "paired_windows": int(len(take)),
        })
    real = np.asarray([row["reference_mean"] for row in detail], dtype=np.float64)
    fake = np.asarray([row["generated_mean"] for row in detail], dtype=np.float64)
    result = _agreement_record(
        model, phase_mode, parameter, real, fake,
        inference_status=f"descriptive_subject_aggregated_n{len(detail)}",
    )
    result.update({
        "n_subjects": len(detail),
        "n_windows_contributing": int(sum(row["paired_windows"] for row in detail)),
    })
    return result, detail


def run(args: argparse.Namespace) -> Path:
    spec = CLINICAL_SPECS[args.dataset]
    phase_dir = args.phase_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    phase_protocol_path = phase_dir / "protocol.json"
    phase_protocol = json.loads(phase_protocol_path.read_text(encoding="utf-8"))
    contract = phase_protocol.get("protocol", {})
    if (
        phase_protocol.get("status") != "completed"
        or contract.get("dataset") != spec["dataset"]
        or contract.get("dataset_variant") not in spec["dataset_variants"]
        or contract.get("model", args.model) != args.model
        or contract.get("dataset_version") != spec["dataset_version"]
        or contract.get("normalization_id") != spec["normalization_id"]
        or float(contract.get("sampling_rate_hz", -1)) != spec["sampling_rate"]
        or contract.get("phase_correction_target_informed") is not True
        or int(contract.get("fixed_support_samples", -1)) != 480
    ):
        raise ValueError("phase artifact violates the random-window clinical contract")

    phase_path = phase_dir / "phase_predictions_maxlag16.npz"
    with np.load(phase_path, allow_pickle=False) as artifact:
        required = {
            "targets", "unshifted_predictions", "oracle_aligned_predictions",
            "subject_ids", "record_ids", "oracle_shift_samples",
        }
        if spec["labels_required"]:
            required.add("labels")
        if missing := sorted(required - set(artifact.files)):
            raise ValueError("phase artifact is missing: " + ", ".join(missing))
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        predictions = {
            "unshifted": np.asarray(artifact["unshifted_predictions"], dtype=np.float32),
            "oracle_aligned": np.asarray(artifact["oracle_aligned_predictions"], dtype=np.float32),
        }
        subjects = np.asarray(artifact["subject_ids"]).astype(str)
        record_ids = np.asarray(artifact["record_ids"]).astype(str)
        labels = (
            np.asarray(artifact["labels"], dtype=np.int16)
            if "labels" in artifact.files else None
        )
        shifts = np.asarray(artifact["oracle_shift_samples"], dtype=np.int32)
    if (
        targets.shape != (spec["windows"], 1, 480)
        or any(values.shape != (spec["windows"], 1, 480) for values in predictions.values())
        or any(values.shape != (spec["windows"],) for values in (subjects, record_ids, shifts))
        or (labels is not None and labels.shape != (spec["windows"],))
    ):
        raise ValueError("clinical arrays violate the dataset phase contract")
    if not np.all(np.isfinite(targets)) or any(
        not np.all(np.isfinite(values)) for values in predictions.values()
    ):
        raise FloatingPointError("clinical waveforms contain NaN or Inf")

    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        real = _measure_records(targets, spec["sampling_rate"], executor)
        generated = {
            phase: _measure_records(values, spec["sampling_rate"], executor)
            for phase, values in predictions.items()
        }
    finally:
        if executor is not None:
            executor.shutdown()

    window_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    subject_detail: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    waveform: dict[str, object] = {}
    delineation: dict[str, object] = {}
    plot_pairs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for phase in PHASE_MODES:
        summary = _waveform_summary(targets, predictions[phase])
        pearson = summary.pop("pearson_per_record")
        waveform[phase] = summary
        measured = generated[phase]
        delineation[phase] = {
            "reference_success": int(sum(item["success"] for item in real)),
            "generated_success": int(sum(item["success"] for item in measured)),
            "paired_success": int(sum(a["success"] and b["success"] for a, b in zip(real, measured))),
            "total": spec["windows"],
        }
        for index, (reference_item, generated_item) in enumerate(zip(real, measured)):
            row = {
                "window_index": index, "record_id": record_ids[index],
                "subject_id": subjects[index],
                "phase_mode": phase, "oracle_shift_samples": int(shifts[index]),
                "waveform_pearson_r": pearson[index],
                "reference_success": reference_item["success"],
                "generated_success": generated_item["success"],
            }
            if labels is not None:
                row["label"] = int(labels[index])
            for parameter in PARAMETERS:
                row[f"reference_{parameter}"] = reference_item["summary"].get(parameter)
                row[f"generated_{parameter}"] = generated_item["summary"].get(parameter)
            parameter_rows.append(row)

    for parameter in PARAMETERS:
        paired, reference, before, after = _parameter_triplets(
            real, generated["unshifted"], generated["oracle_aligned"], parameter
        )
        for phase, values in (("unshifted", before), ("oracle_aligned", after)):
            window_rows.append(_agreement_record(
                args.model, phase, parameter, reference, values,
                inference_status="descriptive_windows_clustered_within_subjects",
            ))
            result, detail = subject_agreement(
                subjects, paired, reference, values, phase, parameter, args.model
            )
            subject_rows.append(result)
            subject_detail.extend(detail)
            plot_pairs[(phase, parameter)] = (reference, values)

    plot_paths = _plot_bland_altman(output, f"{args.model}_record_minmax", plot_pairs)
    paths = {
        "window": output / "window_parameter_agreement.csv",
        "subject": output / "subject_parameter_agreement.csv",
        "subject_detail": output / "per_subject_parameter_means.csv",
        "parameters": output / "per_window_ecg_parameters.csv",
    }
    for key, rows in (
        ("window", window_rows), ("subject", subject_rows),
        ("subject_detail", subject_detail), ("parameters", parameter_rows),
    ):
        _write_csv(paths[key], rows)

    summary_path = output / "clinical_summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": 1, "dataset": spec["dataset"], "model": args.model,
        "protocol": {
            "windows": spec["windows"], "subjects": sorted(set(subjects)),
            "support_samples": 480, "sampling_rate_hz": spec["sampling_rate"],
            "normalization_id": spec["normalization_id"],
            "phase_correction": "target-informed per-window Pearson-maximizing integer shift +/-16 samples",
            "primary_phase_mode_requested": "oracle_aligned",
            "amplitude_unit": spec["amplitude_unit"],
            "physical_amplitude_claim_allowed": False,
            "hrv": "blocked because randomly selected four-second windows are not a continuous sequence",
            "subject_inference": spec["inference"],
        },
        "waveform_agreement": waveform, "delineation": delineation,
        "window_parameter_agreement": window_rows,
        "subject_parameter_agreement": subject_rows,
    }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    outputs = [*paths.values(), summary_path, *plot_paths]
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps({
        "schema_version": 1, "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "input": {"phase_protocol_sha256": _sha256(phase_protocol_path),
                  "phase_predictions_sha256": _sha256(phase_path)},
        "execution": {"python": platform.python_version(), "numpy": np.__version__,
                      "scipy": scipy.__version__, "neurokit2": getattr(nk, "__version__", "unknown"),
                      "workers": args.workers, "script_sha256": _sha256(Path(__file__))},
        "outputs": {path.name: _sha256(path) for path in outputs},
        "claim_boundary": (
            "Oracle alignment uses the paired ECG target and is not deployable; amplitudes are "
            "record-normalized, not mV; one training seed and a non-grouped random-window split."
        ),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(CLINICAL_SPECS), default="wesad_record_minmax")
    parser.add_argument("--model", choices=("cfm", "rcfm", "rcfm_ot", "rddm"), default="cfm")
    parser.add_argument("--workers", type=int, default=16)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
