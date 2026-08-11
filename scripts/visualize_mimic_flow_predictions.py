"""Generate and inspect matched MIMIC-AFib CFM/RCFM/RCFM-OT waveforms."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import (
    _array_sha256,
    _checkpoint_contract,
    _generate,
    _json,
    _pearson_rows,
    _sha256,
    _waveform_metrics,
)
from train_rcfm import build_datasets


MODEL_ORDER = ("cfm", "rcfm", "rcfm_ot")
DISPLAY_NAMES = {"cfm": "CFM", "rcfm": "RCFM", "rcfm_ot": "RCFM-OT"}
COLORS = {"cfm": "#2878b5", "rcfm": "#2f8f5b", "rcfm_ot": "#c43d4b"}
MATCHED_FIELDS = (
    "task",
    "datasets",
    "dataset_version",
    "split_hash",
    "normalization_id",
    "condition_unit",
    "target_unit",
    "alignment_id",
    "condition_lead",
    "target_lead",
    "condition_lead_index",
    "target_lead_index",
    "window_size",
    "attention_heads",
    "flow_matcher",
    "sigma",
    "seed",
)
MATCHED_OUTPUT_FIELDS = ("channels", "length", "sampling_rate_hz", "target_lead")


def _validate_contracts(contracts: Mapping[str, Mapping[str, object]]) -> None:
    if set(contracts) != set(MODEL_ORDER):
        raise ValueError("comparison requires cfm, rcfm, and rcfm_ot checkpoints")
    expected_kinds = {
        "cfm": "canonical_multistep_cfm",
        "rcfm": "canonical_multistep_rcfm",
        "rcfm_ot": "canonical_multistep_rcfm",
    }
    for name, kind in expected_kinds.items():
        if contracts[name]["kind"] != kind:
            raise ValueError(f"{name} checkpoint kind must be {kind}")
        if int(contracts[name]["epoch"]) != 500:
            raise ValueError(f"{name} checkpoint must be the frozen epoch-500 endpoint")
    reference = contracts["cfm"]
    for name in ("rcfm", "rcfm_ot"):
        mismatched = [
            field
            for field in MATCHED_FIELDS
            if reference["config"].get(field) != contracts[name]["config"].get(field)
        ]
        if mismatched:
            raise ValueError(f"{name} checkpoint disagrees on: " + ", ".join(mismatched))
        if reference["normalization"] != contracts[name]["normalization"]:
            raise ValueError(f"{name} checkpoint normalization metadata differs")
        output_mismatches = [
            field
            for field in MATCHED_OUTPUT_FIELDS
            if reference["output_spec"].get(field) != contracts[name]["output_spec"].get(field)
        ]
        if output_mismatches:
            raise ValueError(
                f"{name} checkpoint output specification differs on: "
                + ", ".join(output_mismatches)
            )
    config = reference["config"]
    output_spec = reference["output_spec"]
    if config.get("task") != "ppg2ecg" or config.get("datasets") != ["MIMIC-AFib"]:
        raise ValueError("this entry requires the MIMIC-AFib PPG-to-ECG task")
    if config.get("normalization_id") != "rddm_window_minmax_neg1_1_v1":
        raise ValueError("this entry requires RDDM-compatible window min-max normalization")
    if int(output_spec.get("channels", 0)) != 1 or int(output_spec.get("length", 0)) != 512:
        raise ValueError("this entry requires one 512-sample ECG output channel")
    for name, contract in contracts.items():
        candidate = contract["output_spec"]
        if candidate.get("target_leads", [candidate["target_lead"]]) != [
            candidate["target_lead"]
        ]:
            raise ValueError(f"{name} checkpoint has inconsistent target_leads metadata")
        if candidate.get("target_lead_indices") is not None:
            raise ValueError(f"{name} PPG checkpoint must not declare an ECG lead index")
    cfm = contracts["cfm"]["config"]
    rcfm = contracts["rcfm"]["config"]
    rcfm_ot = contracts["rcfm_ot"]["config"]
    if float(cfm.get("region_weight", -1)) != 0 or bool(cfm.get("use_minibatch_ot")):
        raise ValueError("CFM must have zero region weight and OT disabled")
    if float(rcfm.get("region_weight", 0)) <= 0 or bool(rcfm.get("use_minibatch_ot")):
        raise ValueError("RCFM must have positive region weight and OT disabled")
    if float(rcfm_ot.get("region_weight", 0)) != float(rcfm["region_weight"]):
        raise ValueError("RCFM and RCFM-OT region weights must match")
    if not bool(rcfm_ot.get("use_minibatch_ot")):
        raise ValueError("RCFM-OT must enable minibatch OT")
    if rcfm_ot.get("ot_method") != "exact" or rcfm_ot.get("ot_sampling_strategy") != "assignment":
        raise ValueError("RCFM-OT must use exact assignment coupling")


def _validation_noise(
    record_count: int,
    channels: int,
    length: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Reproduce validate_epoch's independently seeded noise for each batch."""

    if min(record_count, channels, length, batch_size) <= 0:
        raise ValueError("noise dimensions and batch size must be positive")
    batches = []
    for batch_index, start in enumerate(range(0, record_count, batch_size)):
        count = min(batch_size, record_count - start)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + batch_index)
        batches.append(
            torch.randn(
                (count, channels, length),
                generator=generator,
                dtype=torch.float32,
            ).numpy()
        )
    return np.concatenate(batches)


def _select_examples(per_model_rmse: Mapping[str, np.ndarray]) -> dict[str, int]:
    if set(per_model_rmse) != set(MODEL_ORDER):
        raise ValueError("example selection requires all three model RMSE arrays")
    arrays = [np.asarray(per_model_rmse[name], dtype=np.float64) for name in MODEL_ORDER]
    if any(array.ndim != 1 for array in arrays) or len({len(array) for array in arrays}) != 1:
        raise ValueError("per-model RMSE arrays must be aligned one-dimensional arrays")
    if len(arrays[0]) < 3 or not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("example selection requires at least three finite records")
    shared_score = np.mean(np.stack(arrays), axis=0)
    order = np.argsort(shared_score, kind="stable")
    return {
        "best": int(order[0]),
        "median": int(order[len(order) // 2]),
        "worst": int(order[-1]),
    }


def _lag_diagnostic(
    reference: np.ndarray,
    generated: np.ndarray,
    max_lag_samples: int,
    sampling_rate: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Find the descriptive per-window shift maximizing Pearson correlation."""

    if reference.shape != generated.shape or reference.ndim != 3 or reference.shape[1] != 1:
        raise ValueError("lag diagnostic requires aligned single-channel 3D arrays")
    if max_lag_samples <= 0 or max_lag_samples >= reference.shape[-1]:
        raise ValueError("max lag must be positive and shorter than the waveform")
    if sampling_rate <= 0:
        raise ValueError("sampling rate must be positive")
    scores = []
    shifts = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=np.int32)
    for shift in shifts:
        if shift > 0:
            score = _pearson_rows(reference[:, :, shift:], generated[:, :, :-shift])
        elif shift < 0:
            score = _pearson_rows(reference[:, :, :shift], generated[:, :, -shift:])
        else:
            score = _pearson_rows(reference, generated)
        scores.append(score)
    score_matrix = np.stack(scores, axis=1)
    if np.any(np.all(~np.isfinite(score_matrix), axis=1)):
        raise ValueError("lag diagnostic found a constant or invalid generated waveform")
    safe_scores = np.where(np.isfinite(score_matrix), score_matrix, -np.inf)
    best_indices = np.argmax(safe_scores, axis=1)
    best_shifts = shifts[best_indices]
    best_correlations = score_matrix[np.arange(len(reference)), best_indices]
    shift_ms = best_shifts.astype(np.float64) * 1000.0 / sampling_rate
    summary = {
        "status": "descriptive_alignment_diagnostic_only",
        "search_range_samples": [-max_lag_samples, max_lag_samples],
        "search_range_ms": [
            -max_lag_samples * 1000.0 / sampling_rate,
            max_lag_samples * 1000.0 / sampling_rate,
        ],
        "shift_definition": "positive values delay the generated waveform",
        "best_correlation_mean": float(np.mean(best_correlations)),
        "best_correlation_median": float(np.median(best_correlations)),
        "best_shift_ms_mean": float(np.mean(shift_ms)),
        "best_shift_ms_median": float(np.median(shift_ms)),
        "fraction_best_correlation_at_least_0_7": float(np.mean(best_correlations >= 0.7)),
        "claim_boundary": "does not replace unshifted paired metrics",
    }
    return summary, {
        "best_lag_samples": best_shifts,
        "best_lag_ms": shift_ms,
        "lag_adjusted_pearson_r": best_correlations,
    }


def _baseline_summary(reference: np.ndarray, generated: np.ndarray) -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        summary, _ = _waveform_metrics(reference, generated)
    correlation = summary["per_record_pearson"]
    if correlation["usable_records"] == 0:
        correlation["mean"] = None
        correlation["median"] = None
    return summary


def _limits(signals: list[np.ndarray]) -> tuple[float, float]:
    lower = min(float(np.min(signal)) for signal in signals)
    upper = max(float(np.max(signal)) for signal in signals)
    margin = max((upper - lower) * 0.08, 0.05)
    return lower - margin, upper + margin


def _plot_stacked(
    conditions: np.ndarray,
    targets: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    selected: Mapping[str, int],
    shared_rmse: np.ndarray,
    output_path: Path,
    sampling_rate: int,
) -> None:
    columns = tuple(selected)
    rows = ("ppg", "target", "zero", *MODEL_ORDER)
    figure, axes = plt.subplots(
        len(rows), len(columns), figsize=(15, 11), sharex=True, squeeze=False
    )
    time = np.arange(targets.shape[-1], dtype=np.float32) / sampling_rate
    for column, label in enumerate(columns):
        index = selected[label]
        ecg_signals = [targets[index, 0], *(predictions[name][index, 0] for name in MODEL_ORDER)]
        ecg_limits = _limits(ecg_signals)
        for row, signal_name in enumerate(rows):
            axis = axes[row, column]
            if signal_name == "ppg":
                signal = conditions[index, 0]
                color = "#8064a2"
                axis.set_ylim(*_limits([signal]))
            elif signal_name == "target":
                signal = targets[index, 0]
                color = "#111111"
                axis.set_ylim(*ecg_limits)
            elif signal_name == "zero":
                signal = np.zeros_like(targets[index, 0])
                color = "#7f7f7f"
                axis.set_ylim(*ecg_limits)
            else:
                signal = predictions[signal_name][index, 0]
                color = COLORS[signal_name]
                axis.set_ylim(*ecg_limits)
            axis.plot(time, signal, color=color, linewidth=0.9)
            axis.grid(alpha=0.18)
            axis.set_xlim(0.0, targets.shape[-1] / sampling_rate)
            if column == 0:
                row_label = {
                    "ppg": "Condition PPG",
                    "target": "Real ECG",
                    "zero": "Zero baseline",
                    **DISPLAY_NAMES,
                }[signal_name]
                axis.set_ylabel(row_label)
            if row == 0:
                axis.set_title(
                    f"{label.capitalize()} shared error | row {index} | mean RMSE {shared_rmse[index]:.3f}"
                )
            if row == len(rows) - 1:
                axis.set_xlabel("Time (s)")
    figure.suptitle(
        "MIMIC-AFib matched deterministic waveform inspection (normalized domain)",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.97))
    figure.savefig(output_path, dpi=240)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def _plot_overlay(
    targets: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    selected: Mapping[str, int],
    output_path: Path,
    sampling_rate: int,
) -> None:
    figure, axes = plt.subplots(len(selected), 1, figsize=(14, 8), sharex=True, squeeze=False)
    time = np.arange(targets.shape[-1], dtype=np.float32) / sampling_rate
    for axis, (label, index) in zip(axes[:, 0], selected.items()):
        axis.plot(time, targets[index, 0], color="#111111", linewidth=1.25, label="Real ECG")
        for name in MODEL_ORDER:
            axis.plot(
                time,
                predictions[name][index, 0],
                color=COLORS[name],
                linewidth=0.8,
                alpha=0.82,
                label=DISPLAY_NAMES[name],
            )
        axis.axhline(0.0, color="#7f7f7f", linewidth=0.7, linestyle="--", label="Zero baseline")
        axis.set_title(f"{label.capitalize()} shared-error example | test row {index}")
        axis.set_ylabel("Normalized amplitude")
        axis.grid(alpha=0.18)
        axis.set_xlim(0.0, targets.shape[-1] / sampling_rate)
    axes[0, 0].legend(ncol=5, loc="upper right")
    axes[-1, 0].set_xlabel("Time (s)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=240)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def _read_validation_metrics(checkpoint_path: Path) -> dict[str, float]:
    path = checkpoint_path.resolve().parent / "validation_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing validation metrics beside checkpoint: {path}")
    values = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[row["metric"]] = float(row["value"])
    required = ("val/rmse", "val/mae", "val/waveform_fd", "val/num_samples")
    if any(name not in values for name in required):
        raise ValueError(f"validation metrics are incomplete: {path}")
    return values


def _write_per_window_metrics(
    output_path: Path,
    targets: np.ndarray,
    conditions: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    per_model: Mapping[str, Mapping[str, np.ndarray]],
    lag_per_model: Mapping[str, Mapping[str, np.ndarray]],
    selected: Mapping[str, int],
) -> None:
    selected_by_index = {index: label for label, index in selected.items()}
    fields = ["test_row", "selection", "target_rms", "ppg_rms"]
    for name in MODEL_ORDER:
        fields.extend(
            [
                f"{name}_rmse",
                f"{name}_mae",
                f"{name}_pearson_r",
                f"{name}_rms",
                f"{name}_best_lag_samples",
                f"{name}_best_lag_ms",
                f"{name}_lag_adjusted_pearson_r",
            ]
        )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(targets)):
            row = {
                "test_row": index,
                "selection": selected_by_index.get(index, ""),
                "target_rms": float(np.sqrt(np.mean(targets[index].astype(np.float64) ** 2))),
                "ppg_rms": float(np.sqrt(np.mean(conditions[index].astype(np.float64) ** 2))),
            }
            for name in MODEL_ORDER:
                row.update(
                    {
                        f"{name}_rmse": per_model[name]["rmse"][index],
                        f"{name}_mae": per_model[name]["mae"][index],
                        f"{name}_pearson_r": per_model[name]["pearson_r"][index],
                        f"{name}_rms": float(
                            np.sqrt(np.mean(predictions[name][index].astype(np.float64) ** 2))
                        ),
                        f"{name}_best_lag_samples": lag_per_model[name]["best_lag_samples"][index],
                        f"{name}_best_lag_ms": lag_per_model[name]["best_lag_ms"][index],
                        f"{name}_lag_adjusted_pearson_r": lag_per_model[name][
                            "lag_adjusted_pearson_r"
                        ][index],
                    }
                )
            writer.writerow(row)


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {
        "cfm": args.cfm_checkpoint.resolve(),
        "rcfm": args.rcfm_checkpoint.resolve(),
        "rcfm_ot": args.rcfm_ot_checkpoint.resolve(),
    }
    contracts = {name: _checkpoint_contract(path) for name, path in checkpoint_paths.items()}
    _validate_contracts(contracts)
    config = contracts["cfm"]["config"]
    output_spec = contracts["cfm"]["output_spec"]
    _, test_set = build_datasets(
        str(config["task"]),
        config["datasets"],
        str(args.data_root.resolve()),
        int(config["window_size"]),
        normalization_metadata=contracts["cfm"]["normalization"],
        normalization_id=str(config["normalization_id"]),
        condition_lead_index=config.get("condition_lead_index"),
        target_lead_index=config.get("target_lead_index"),
        load_train=False,
        heldout_split="test",
    )
    if len(test_set) != args.expected_records:
        raise ValueError(f"expected {args.expected_records} held-out windows, found {len(test_set)}")
    targets = np.asarray(test_set.target_ecg[:, None, :], dtype=np.float32)
    conditions = np.asarray(test_set.condition_signal[:, None, :], dtype=np.float32)
    if targets.shape != conditions.shape or targets.shape[1:] != (1, 512):
        raise ValueError("MIMIC held-out arrays must be aligned (records,1,512) tensors")
    noise = _validation_noise(
        len(targets),
        int(output_spec["channels"]),
        int(output_spec["length"]),
        args.batch_size,
        args.noise_seed,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch cannot access CUDA")

    predictions = {}
    generation_metadata = {}
    expected_kinds = {
        "cfm": "canonical_multistep_cfm",
        "rcfm": "canonical_multistep_rcfm",
        "rcfm_ot": "canonical_multistep_rcfm",
    }
    for name in MODEL_ORDER:
        prediction, metadata = _generate(
            checkpoint_paths[name],
            expected_kinds[name],
            conditions,
            noise,
            args.batch_size,
            args.steps,
            device,
            args.deterministic_seed,
        )
        predictions[name] = prediction
        generation_metadata[name] = metadata

    summaries = {}
    per_model = {}
    lag_per_model = {}
    validation_reproduction = {}
    for name in MODEL_ORDER:
        summaries[name], per_model[name] = _waveform_metrics(targets, predictions[name])
        expected = _read_validation_metrics(checkpoint_paths[name])
        differences = {
            "rmse": summaries[name]["rmse"] - expected["val/rmse"],
            "mae": summaries[name]["mae"] - expected["val/mae"],
            "waveform_fd": summaries[name]["waveform_fd"] - expected["val/waveform_fd"],
        }
        if max(abs(value) for value in differences.values()) > args.validation_tolerance:
            raise ValueError(f"{name} predictions do not reproduce saved validation metrics: {differences}")
        lag_summary, lag_per_model[name] = _lag_diagnostic(
            targets,
            predictions[name],
            args.max_lag_samples,
            args.sampling_rate,
        )
        summaries[name]["lag_adjusted_correlation_diagnostic"] = lag_summary
        validation_reproduction[name] = {
            "expected": {
                "rmse": expected["val/rmse"],
                "mae": expected["val/mae"],
                "waveform_fd": expected["val/waveform_fd"],
            },
            "difference": differences,
        }
    baselines = {}
    for name, values in (("zero", np.zeros_like(targets)), ("ppg_copy", conditions)):
        baselines[name] = _baseline_summary(targets, values)

    shared_rmse = np.mean(
        np.stack([np.asarray(per_model[name]["rmse"]) for name in MODEL_ORDER]), axis=0
    )
    selected = _select_examples({name: per_model[name]["rmse"] for name in MODEL_ORDER})
    _write_per_window_metrics(
        output_dir / "per_window_metrics.csv",
        targets,
        conditions,
        predictions,
        per_model,
        lag_per_model,
        selected,
    )
    _plot_stacked(
        conditions,
        targets,
        predictions,
        selected,
        shared_rmse,
        output_dir / "waveforms_best_median_worst.png",
        args.sampling_rate,
    )
    _plot_overlay(
        targets,
        predictions,
        selected,
        output_dir / "waveforms_overlay_best_median_worst.png",
        args.sampling_rate,
    )
    np.savez_compressed(
        output_dir / "mimic_flow_predictions.npz",
        targets=targets,
        conditions=conditions,
        initial_noise=noise,
        cfm_predictions=predictions["cfm"],
        rcfm_predictions=predictions["rcfm"],
        rcfm_ot_predictions=predictions["rcfm_ot"],
    )
    _json(
        output_dir / "waveform_summary.json",
        {
            "models": summaries,
            "baselines": baselines,
            "selection": {
                label: {
                    "test_row": index,
                    "shared_mean_rmse": float(shared_rmse[index]),
                }
                for label, index in selected.items()
            },
            "validation_reproduction": validation_reproduction,
        },
    )

    artifact_names = [
        "waveforms_best_median_worst.png",
        "waveforms_best_median_worst.pdf",
        "waveforms_overlay_best_median_worst.png",
        "waveforms_overlay_best_median_worst.pdf",
        "mimic_flow_predictions.npz",
        "per_window_metrics.csv",
        "waveform_summary.json",
    ]
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "command": shlex.join(sys.argv),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": "MIMIC-AFib",
            "dataset_version": config["dataset_version"],
            "split_hash": config["split_hash"],
            "heldout_windows": len(targets),
            "normalization_id": config["normalization_id"],
            "sampling_rate_hz": args.sampling_rate,
            "window_seconds": int(config["window_size"]),
            "nfe": args.steps,
            "batch_size": args.batch_size,
            "validation_noise_seed": args.noise_seed,
            "deterministic_seed": args.deterministic_seed,
            "noise_sha256": _array_sha256(noise),
            "same_noise_for_all_models": True,
            "selection": "best/median/worst by mean per-window RMSE across all three models",
            "prediction_domain": "RDDM-compatible per-window normalized and NeuroKit-cleaned",
            "lag_diagnostic": (
                f"descriptive per-window Pearson-maximizing shift within +/-{args.max_lag_samples} "
                "samples; never substitutes for unshifted paired metrics"
            ),
        },
        "checkpoints": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                **generation_metadata[name],
            }
            for name, path in checkpoint_paths.items()
        },
        "execution": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "claim_boundary": (
            "Four-second windows lack subject IDs, continuity, certified AF labels, and a physical "
            "inverse transform. Results are normalized-domain waveform diagnostics only."
        ),
        "artifact_sha256": {name: _sha256(output_dir / name) for name in artifact_names},
        "outputs": [*artifact_names, "protocol.json"],
    }
    _json(output_dir / "protocol.json", protocol)
    print(json.dumps({"output_dir": str(output_dir), "selected": selected}, sort_keys=True))
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfm_checkpoint", type=Path, required=True)
    parser.add_argument("--rcfm_checkpoint", type=Path, required=True)
    parser.add_argument("--rcfm_ot_checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--noise_seed", type=int, default=2025)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    parser.add_argument("--sampling_rate", type=int, default=128)
    parser.add_argument("--max_lag_samples", type=int, default=64)
    parser.add_argument("--expected_records", type=int, default=1800)
    parser.add_argument("--validation_tolerance", type=float, default=2e-5)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
