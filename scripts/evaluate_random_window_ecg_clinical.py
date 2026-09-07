"""Compute ECG-parameter agreement and Bland--Altman results for random windows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import sys
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_random_window_clinical_mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _sha256
from scripts.evaluate_ptbxl_clinical_agreement import _load_p_wave_applicability
from scripts.evaluate_ptbxl_fourway import TARGET_INDICES, TARGET_LEADS
from src.rcfm.metrics.bland_altman import bland_altman, paired_correlation
from src.rcfm.metrics.clinical import ECGFiducials, delineate_ecg, measure_ecg_parameters


PARAMETERS = (
    "heart_rate_bpm", "rr_ms", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms",
    "p_amplitude", "r_amplitude", "qrs_peak_to_peak_amplitude", "t_amplitude", "st_deviation",
)
INTERVAL_PARAMETERS = {"rr_ms", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms"}
AMPLITUDE_PARAMETERS = {
    "p_amplitude", "r_amplitude", "qrs_peak_to_peak_amplitude", "t_amplitude", "st_deviation"
}
LABELS = {
    "heart_rate_bpm": "HR", "rr_ms": "RR", "pr_ms": "PR", "qrs_ms": "QRS",
    "qt_ms": "QT", "qtc_ms": "QTc", "p_amplitude": "P", "r_amplitude": "R",
    "qrs_peak_to_peak_amplitude": "QRS p-p", "t_amplitude": "T", "st_deviation": "ST",
}
DATASET_SPECS = {
    "ptbxl": {
        "name": "PTBXL", "leads": TARGET_LEADS, "physical": True,
        "delineation_minimum_seconds": 4.0,
    },
    "cpsc2018": {
        "name": "CPSC2018", "leads": TARGET_LEADS, "physical": False,
        "delineation_minimum_seconds": 4.0,
    },
    "mmecg": {
        "name": "mmECG", "leads": ("single_channel_ECG",), "physical": False,
        "delineation_minimum_seconds": 8.0,
    },
    "wesad": {
        "name": "WESAD", "leads": ("chest_ECG",), "physical": False,
        "delineation_minimum_seconds": 4.0,
    },
}


def inverse_shared_record_minmax(
    normalized: np.ndarray, minima: np.ndarray, ranges: np.ndarray
) -> np.ndarray:
    values = np.asarray(normalized, dtype=np.float32)
    offsets = np.asarray(minima, dtype=np.float32).reshape(-1)
    scales = np.asarray(ranges, dtype=np.float32).reshape(-1)
    if values.ndim != 3 or len(values) != len(offsets) or len(scales) != len(offsets):
        raise ValueError("shared record scalers must align with waveform windows")
    if np.any(scales <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("inverse inputs must be finite and ranges positive")
    return ((values + 1.0) * scales[:, None, None] / 2.0 + offsets[:, None, None]).astype(np.float32)


def _measure(task: tuple[np.ndarray, float, bool, bool, float]) -> dict[str, object]:
    signal, sampling_rate, physical, p_wave_applicable, minimum_seconds = task
    minimum_samples = int(np.ceil(minimum_seconds * sampling_rate))
    padding = max(0, minimum_samples - len(signal))
    left_padding = padding // 2
    padded = (
        np.pad(signal, (left_padding, padding - left_padding), mode="reflect")
        if padding else signal
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Too few peaks detected.*")
        result = delineate_ecg(padded, sampling_rate=sampling_rate, method="dwt")
    if not result.success:
        return {"success": False, "failure_reason": result.failure_reason, "summary": {}}
    if padding:
        centered_r = result.fiducials.r_peaks - left_padding
        keep = (centered_r >= 0) & (centered_r < len(signal))
        if int(np.sum(keep)) < 2:
            return {
                "success": False,
                "failure_reason": "fewer_than_two_center_r_peaks",
                "summary": {},
            }
        shifted = {}
        for name in result.fiducials.__dataclass_fields__:
            values = (
                np.asarray(getattr(result.fiducials, name), dtype=np.int64)[keep]
                - left_padding
            )
            values[(values < 0) | (values >= len(signal))] = -1
            shifted[name] = values
        fiducials = ECGFiducials(**shifted)
    else:
        fiducials = result.fiducials
    measurement = measure_ecg_parameters(
        signal, sampling_rate=sampling_rate, fiducials=fiducials,
        amplitude_unit="mV" if physical else "normalized",
        inverse_transformed=physical, allow_normalized_amplitudes=not physical,
        qtc_formula="fridericia", st_offset_ms=60.0, continuous=False,
        p_wave_applicable=p_wave_applicable,
    )
    summary = {
        name: float(np.mean(values))
        for name, values in measurement["parameters"].items() if values
    }
    if summary.get("rr_ms", 0) > 0:
        summary["heart_rate_bpm"] = 60000.0 / summary["rr_ms"]
    return {
        "success": True, "failure_reason": None, "summary": summary,
        "r_peaks": int(len(fiducials.r_peaks)),
        "amplitude_status": measurement["amplitude_status"],
        "hrv_status": measurement["hrv"]["status"],
    }


def _measure_many(
    signals: np.ndarray, sampling_rate: float, physical: bool,
    p_wave_applicable: np.ndarray, minimum_seconds: float,
    executor: ProcessPoolExecutor | None,
) -> list[dict[str, object]]:
    tasks = (
        (np.asarray(signal), sampling_rate, physical, bool(applicable), minimum_seconds)
        for signal, applicable in zip(signals, p_wave_applicable)
    )
    if executor is None:
        return [_measure(task) for task in tasks]
    return list(executor.map(_measure, tasks, chunksize=16))


def grouped_parameter_pairs(
    reference: list[dict[str, object]], generated: list[dict[str, object]],
    group_ids: np.ndarray, parameter: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for group, real_item, generated_item in zip(group_ids, reference, generated):
        real = real_item["summary"].get(parameter)
        fake = generated_item["summary"].get(parameter)
        if real is None or fake is None or not np.isfinite(real) or not np.isfinite(fake):
            continue
        grouped[str(group)][0].append(float(real))
        grouped[str(group)][1].append(float(fake))
    ids = np.asarray(sorted(grouped))
    return (
        ids,
        np.asarray([np.mean(grouped[item][0]) for item in ids], dtype=np.float64),
        np.asarray([np.mean(grouped[item][1]) for item in ids], dtype=np.float64),
    )


def _unit(parameter: str, physical: bool) -> str:
    if parameter == "heart_rate_bpm":
        return "bpm"
    if parameter in INTERVAL_PARAMETERS:
        return "ms"
    return "mV" if physical else "normalized"


def _agreement_row(
    dataset: str, lead: str, parameter: str, groups: np.ndarray,
    reference: np.ndarray, generated: np.ndarray, physical: bool, model: str = "cfm",
) -> dict[str, object]:
    base = {
        "dataset": dataset, "model": model, "lead": lead, "parameter": parameter,
        "unit": _unit(parameter, physical), "n_groups": int(len(groups)),
        "group_level": (
            "patient" if dataset == "ptbxl" else "subject" if dataset == "wesad" else "source_record"
        ),
        "difference_definition": "generated_minus_reference",
    }
    if len(reference) < 2:
        return {**base, "status": "insufficient_group_pairs"}
    error = generated - reference
    correlation = paired_correlation(reference, generated)
    agreement = bland_altman(reference, generated)
    return {
        **base, "status": "ok", "reference_mean": float(np.mean(reference)),
        "generated_mean": float(np.mean(generated)), "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))), "pearson_r": correlation["r"],
        "pearson_status": correlation["status"], "ba_bias": agreement["bias"],
        "ba_difference_sd": agreement["difference_sd"], "ba_lower": agreement["lower_limit"],
        "ba_upper": agreement["upper_limit"],
        "ba_loa_width": agreement["upper_limit"] - agreement["lower_limit"],
    }


def _macro_rows(rows: list[dict[str, object]], model: str = "cfm") -> list[dict[str, object]]:
    output = []
    for parameter in PARAMETERS:
        selected = [row for row in rows if row["parameter"] == parameter and row["status"] == "ok"]
        if not selected:
            output.append({"model": model, "parameter": parameter, "status": "insufficient_leads"})
            continue
        correlations = [row["pearson_r"] for row in selected if row["pearson_r"] is not None]
        fisher = (
            float(np.tanh(np.mean(np.arctanh(np.clip(correlations, -0.999999, 0.999999)))))
            if correlations else None
        )
        output.append({
            "model": model, "parameter": parameter, "status": "ok", "unit": selected[0]["unit"],
            "usable_leads": len(selected), "n_groups_min": min(row["n_groups"] for row in selected),
            "n_groups_max": max(row["n_groups"] for row in selected),
            "mae": float(np.mean([row["mae"] for row in selected])),
            "rmse": float(np.mean([row["rmse"] for row in selected])),
            "pearson_r": fisher, "pearson_aggregation": "unweighted_Fisher_z_mean_across_leads",
            "ba_bias": float(np.mean([row["ba_bias"] for row in selected])),
            "ba_loa_width": float(np.mean([row["ba_loa_width"] for row in selected])),
            "lead_aggregation": "unweighted_macro_mean_of_lead_level_agreements",
        })
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_ba(
    pairs: dict[str, tuple[np.ndarray, np.ndarray]], lead: str, physical: bool, output: Path,
    model: str = "cfm",
) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times"],
                         "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(3, 4, figsize=(7.16, 6.0), constrained_layout=True)
    for axis, parameter in zip(axes.flat, PARAMETERS):
        real, fake = pairs.get(parameter, (np.empty(0), np.empty(0)))
        if len(real) >= 2:
            means, differences = (real + fake) / 2.0, fake - real
            if len(means) > 900:
                indices = np.linspace(0, len(means) - 1, 900, dtype=np.int64)
                means, differences = means[indices], differences[indices]
            agreement = bland_altman(real, fake)
            axis.scatter(means, differences, s=4, alpha=0.18, color="#2878b5", rasterized=True)
            axis.axhline(agreement["bias"], color="black", linewidth=0.9)
            axis.axhline(agreement["lower_limit"], color="gray", linestyle="--", linewidth=0.75)
            axis.axhline(agreement["upper_limit"], color="gray", linestyle="--", linewidth=0.75)
        axis.set_title(f"{LABELS[parameter]} (n={len(real)})")
        axis.set_xlabel(f"Pair mean ({_unit(parameter, physical)})")
        axis.set_ylabel(f"Generated - real ({_unit(parameter, physical)})")
        axis.grid(alpha=0.16)
    axes.flat[-1].set_visible(False)
    figure.suptitle(f"{model.upper().replace('_', '-')} Lead {lead} ECG-parameter Bland-Altman", fontsize=8.5)
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=600)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    input_dir, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((input_dir / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "completed" or protocol["protocol"].get("phase_correction_applied") is not False:
        raise ValueError("clinical analysis requires completed raw synchronized predictions")
    dataset_spec = DATASET_SPECS[args.dataset]
    expected_dataset = dataset_spec["name"]
    leads = tuple(dataset_spec["leads"])
    if args.representative_lead not in leads:
        raise ValueError(
            f"representative lead/channel {args.representative_lead!r} is not available for {expected_dataset}"
        )
    if protocol["protocol"].get("dataset") != expected_dataset:
        raise ValueError("prediction artifact dataset does not match --dataset")
    with np.load(input_dir / "paired_reference.npz", allow_pickle=False) as artifact:
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        record_ids = np.asarray(artifact["record_ids"]).astype(str)
        patient_ids = np.asarray(artifact["patient_ids"]).astype(str) if args.dataset == "ptbxl" else None
        subject_ids = np.asarray(artifact["subject_ids"]).astype(str) if args.dataset == "wesad" else None
    predictions = np.asarray(
        np.load(input_dir / f"{args.model}_predictions.npy", mmap_mode="r"), dtype=np.float32
    )
    if predictions.shape != targets.shape or targets.shape[1:] != (len(leads), 512):
        raise ValueError("prediction/reference shapes must match the dataset channel contract")
    dataset_dir = args.data_root.resolve() / expected_dataset
    physical = bool(dataset_spec["physical"])
    delineation_minimum_seconds = float(dataset_spec["delineation_minimum_seconds"])
    if physical:
        minima = np.load(dataset_dir / "record_joint_minima_val.npy", allow_pickle=False)
        ranges = np.load(dataset_dir / "record_joint_ranges_val.npy", allow_pickle=False)
        targets_eval = inverse_shared_record_minmax(targets, minima, ranges)
        predictions_eval = inverse_shared_record_minmax(predictions, minima, ranges)
        raw = np.load(dataset_dir / "X_val_resampled.npy", mmap_mode="r")[:, :512, :]
        expected = np.transpose(raw[:, :, list(TARGET_INDICES)], (0, 2, 1))
        # The loader normalizes and stores float32 values before this audit; the
        # full 8,735-window inverse has a worst observed round-trip error below
        # 3e-6 mV, so retain a small explicit float32 tolerance.
        if not np.allclose(targets_eval, expected, atol=4e-6, rtol=2e-6):
            raise ValueError("PTB-XL inverse transform does not reproduce stored mV targets")
        if args.metadata_csv is None:
            raise ValueError("PTB-XL requires --metadata_csv for AFIB/AFLT P-wave policy")
        p_wave, p_wave_policy = _load_p_wave_applicability(args.metadata_csv.resolve(), record_ids)
        group_ids = patient_ids
        group_level = "patient"
    else:
        targets_eval, predictions_eval = targets, predictions
        p_wave = np.ones(len(targets), dtype=bool)
        p_wave_policy = {
            "policy": f"all {expected_dataset} windows treated as P-wave applicable when delineated"
        }
        if args.dataset == "wesad":
            group_ids = subject_ids
            group_level = "subject"
        else:
            group_ids = record_ids
            group_level = "source_record"
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    delineation: dict[str, object] = {}
    representative_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    try:
        for lead_index, lead in enumerate(leads):
            print(f"clinical delineation: reference {lead}", flush=True)
            real = _measure_many(
                targets_eval[:, lead_index], args.sampling_rate, physical, p_wave,
                delineation_minimum_seconds, executor,
            )
            print(f"clinical delineation: {args.model} {lead}", flush=True)
            fake = _measure_many(
                predictions_eval[:, lead_index], args.sampling_rate, physical, p_wave,
                delineation_minimum_seconds, executor,
            )
            delineation[lead] = {
                "reference_success": int(sum(item["success"] for item in real)),
                "generated_success": int(sum(item["success"] for item in fake)),
                "reference_failures": dict(Counter(item["failure_reason"] for item in real if not item["success"])),
                "generated_failures": dict(Counter(item["failure_reason"] for item in fake if not item["success"])),
            }
            for parameter in PARAMETERS:
                groups, real_values, fake_values = grouped_parameter_pairs(real, fake, group_ids, parameter)
                rows.append(_agreement_row(
                    args.dataset, lead, parameter, groups, real_values, fake_values,
                    physical, args.model,
                ))
                if lead == args.representative_lead:
                    representative_pairs[parameter] = (real_values, fake_values)
            for index, (real_item, fake_item) in enumerate(zip(real, fake)):
                detail = {
                    "window_index": index, "record_id": record_ids[index], "group_id": str(group_ids[index]),
                    "group_level": group_level, "lead": lead,
                    "reference_success": real_item["success"], "generated_success": fake_item["success"],
                }
                for parameter in PARAMETERS:
                    detail[f"reference_{parameter}"] = real_item["summary"].get(parameter)
                    detail[f"generated_{parameter}"] = fake_item["summary"].get(parameter)
                details.append(detail)
    finally:
        if executor is not None:
            executor.shutdown()
    macro = _macro_rows(rows, args.model)
    per_lead_path = output / "per_lead_group_agreement.csv"
    macro_path = output / "macro_lead_agreement.csv"
    details_path = output / "per_window_parameters.csv"
    _write_csv(per_lead_path, rows)
    _write_csv(macro_path, macro)
    _write_csv(details_path, details)
    figure_stem = output / f"{args.model}_lead_{args.representative_lead.lower()}_ecg_parameter_bland_altman"
    _plot_ba(representative_pairs, args.representative_lead, physical, figure_stem, args.model)
    summary_path = output / "clinical_summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": 1, "dataset": expected_dataset,
        "model": args.model.upper().replace("_", "-"),
        "protocol": {
            "windows": len(targets), "groups": int(len(np.unique(group_ids))), "group_level": group_level,
            "leads": list(leads), "sampling_rate_hz": args.sampling_rate, "window_seconds": 4,
            "alignment": "raw synchronized no phase correction", "delineation": "independent NeuroKit2 DWT",
            "delineation_minimum_seconds_reflect_padding": delineation_minimum_seconds,
            "qtc_formula": "fridericia", "st_offset_ms": 60.0,
            "amplitude_unit": "mV_oracle_inverse" if physical else "normalized_exploratory_not_physical",
            "hrv": "blocked_noncontinuous_four_second_windows", "p_wave_policy": p_wave_policy,
            "representative_bland_altman_lead": args.representative_lead,
        },
        "delineation": delineation, "per_lead_agreement": rows, "macro_lead_agreement": macro,
    }, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    output_files = [per_lead_path, macro_path, details_path, summary_path,
                    figure_stem.with_suffix(".pdf"), figure_stem.with_suffix(".png")]
    result_protocol = {
        "schema_version": 1, "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv), "dataset": expected_dataset, "model": args.model,
        "windows": len(targets), "groups": int(len(np.unique(group_ids))), "group_level": group_level,
        "workers": args.workers, "source_protocol_sha256": _sha256(input_dir / "protocol.json"),
        "source_prediction_sha256": _sha256(input_dir / f"{args.model}_predictions.npy"),
        "claim_boundary": (
            "Single-seed descriptive validation analysis; non-grouped train/validation split; "
            + ("PTB-XL amplitudes use oracle target-informed mV scalers."
               if physical else f"{expected_dataset} amplitudes are normalized exploratory values, not physical units.")
        ),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "neurokit2": getattr(nk, "__version__", "unknown")},
        "outputs": {path.name: _sha256(path) for path in output_files},
    }
    (output / "protocol.json").write_text(
        json.dumps(result_protocol, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
    parser.add_argument("--model", choices=("cfm", "rcfm", "rcfm_ot"), default="cfm")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--metadata_csv", type=Path)
    parser.add_argument("--sampling_rate", type=float, default=128.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--representative_lead", default="V3")
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
