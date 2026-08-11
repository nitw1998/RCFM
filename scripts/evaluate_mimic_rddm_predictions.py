"""Deterministically evaluate the reproduced RDDM on frozen MIMIC-AFib rows."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import random
import shlex
import sys
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

from data import get_ppg2ecg_datasets
from diffusion import RDDM
from model import ConditionNet, DiffusionUNetCrossAttention
from scripts.evaluate_cpsc_zscore_paired import (
    _array_sha256,
    _json,
    _sha256,
    _waveform_metrics,
)
from scripts.visualize_mimic_flow_predictions import (
    _baseline_summary,
    _lag_diagnostic,
    _limits,
)


EXPECTED_UPSTREAM_COMMIT = "7d5348843c3985c211a23ae5105a2d9497d5156a"
EXPECTED_SPLIT_HASH = "a7e388293adaa7b48d3493efc505dd8750520730cae9fd7649157866efa86a51"


def _validate_checkpoint(payload: Mapping[str, object]) -> None:
    required = {
        "schema_version",
        "kind",
        "epoch",
        "global_step",
        "rddm_state",
        "condition_1_state",
        "condition_2_state",
        "config",
        "normalization",
        "provenance",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("RDDM checkpoint is missing: " + ", ".join(missing))
    if payload["schema_version"] != 1 or payload["kind"] != "independent_rddm_reproduction":
        raise ValueError("checkpoint must be the schema-1 independent RDDM reproduction")
    if int(payload["epoch"]) != 500 or int(payload["global_step"]) != 33000:
        raise ValueError("RDDM evaluation requires the frozen epoch-500/step-33000 endpoint")
    config = payload["config"]
    if not isinstance(config, Mapping):
        raise ValueError("RDDM checkpoint config must be a mapping")
    expected = {
        "task": "ppg2ecg",
        "datasets": ["MIMIC-AFib"],
        "dataset_version": "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1",
        "split_hash": EXPECTED_SPLIT_HASH,
        "normalization_id": "rddm_window_minmax_neg1_1_v1",
        "alignment_id": "paired_array_row_rddm_contract_zero_ppg_qc_v1",
        "window_size": 4,
        "expected_train_windows": 8400,
        "expected_test_windows": 1800,
        "nT": 10,
        "attention_heads": 8,
        "seed": 31,
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
    }
    mismatched = [key for key, value in expected.items() if config.get(key) != value]
    if mismatched:
        raise ValueError("RDDM checkpoint violates the frozen contract: " + ", ".join(mismatched))
    if float(config.get("beta_start", -1)) != 1e-4 or float(config.get("beta_end", -1)) != 0.2:
        raise ValueError("RDDM checkpoint must use betas=(1e-4, 0.2)")
    normalization = payload["normalization"]
    if not isinstance(normalization, Mapping):
        raise ValueError("RDDM normalization metadata must be a mapping")
    normalization_expected = {
        "method": "rddm_window_minmax_neg1_1",
        "stats_scope": "per_window_per_modality",
        "feature_range": [-1.0, 1.0],
        "generated_inverse_policy": "normalized_domain_only",
    }
    normalization_mismatches = [
        key for key, value in normalization_expected.items() if normalization.get(key) != value
    ]
    if normalization_mismatches:
        raise ValueError(
            "RDDM checkpoint normalization violates the frozen contract: "
            + ", ".join(normalization_mismatches)
        )
    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping) or provenance.get("upstream_commit") != EXPECTED_UPSTREAM_COMMIT:
        raise ValueError("RDDM checkpoint provenance has the wrong upstream commit")


def _validate_dataset_manifest(path: Path, config: Mapping[str, object]) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != config["dataset_version"]:
        raise ValueError("dataset manifest version differs from checkpoint")
    if manifest.get("split_membership_hash") != config["split_hash"]:
        raise ValueError("dataset manifest split hash differs from checkpoint")
    if manifest.get("splits", {}).get("test", {}).get("retained_windows") != 1800:
        raise ValueError("dataset manifest must declare 1,800 retained test windows")
    if manifest.get("filter", {}).get("predicate") != "all_512_samples_equal_zero":
        raise ValueError("dataset manifest does not declare the frozen all-zero PPG filter")
    return manifest


def _set_deterministic(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def _generate_batches(
    model: torch.nn.Module,
    condition_net_1: torch.nn.Module,
    condition_net_2: torch.nn.Module,
    conditions: np.ndarray,
    batch_size: int,
    sampling_seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if conditions.ndim != 3 or conditions.shape[1:] != (1, 512):
        raise ValueError("RDDM conditions must have shape (records,1,512)")
    if batch_size <= 0 or sampling_seed < 0:
        raise ValueError("batch size must be positive and sampling seed nonnegative")
    predictions = np.empty_like(conditions, dtype=np.float32)
    batch_indices = np.empty(len(conditions), dtype=np.int32)
    batch_seeds = []
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    for batch_index, start in enumerate(range(0, len(conditions), batch_size)):
        stop = min(start + batch_size, len(conditions))
        seed = sampling_seed + batch_index
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            condition = torch.from_numpy(conditions[start:stop]).to(device=device)
            encoded_1 = condition_net_1(condition)
            encoded_2 = condition_net_2(condition)
            generated = model(
                cond1=encoded_1,
                cond2=encoded_2,
                mode="sample",
                window_size=conditions.shape[-1],
            )
        predictions[start:stop] = generated.detach().cpu().numpy()
        batch_indices[start:stop] = batch_index
        batch_seeds.append(seed)
        print(f"RDDM: generated {stop}/{len(conditions)} with batch_seed={seed}", flush=True)
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("RDDM predictions contain NaN or Inf")
    return predictions, batch_indices, np.asarray(batch_seeds, dtype=np.int64)


def _select_examples(rmse: np.ndarray) -> dict[str, int]:
    values = np.asarray(rmse, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3 or not np.all(np.isfinite(values)):
        raise ValueError("example selection requires at least three finite RMSE values")
    order = np.argsort(values, kind="stable")
    return {"best": int(order[0]), "median": int(order[len(order) // 2]), "worst": int(order[-1])}


def _write_per_window_metrics(
    path: Path,
    original_rows: np.ndarray,
    targets: np.ndarray,
    conditions: np.ndarray,
    predictions: np.ndarray,
    per_record: Mapping[str, np.ndarray],
    lag_values: Mapping[str, np.ndarray],
    batch_indices: np.ndarray,
    batch_seeds: np.ndarray,
    selected: Mapping[str, int],
) -> None:
    selected_by_row = {row: label for label, row in selected.items()}
    fields = [
        "test_row",
        "source_test_row_before_zero_filter",
        "selection",
        "sampling_batch_index",
        "sampling_batch_seed",
        "target_rms",
        "ppg_rms",
        "prediction_rms",
        "rmse",
        "mae",
        "bias",
        "pearson_r",
        "best_lag_samples",
        "best_lag_ms",
        "lag_adjusted_pearson_r",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in range(len(targets)):
            batch_index = int(batch_indices[row])
            writer.writerow(
                {
                    "test_row": row,
                    "source_test_row_before_zero_filter": int(original_rows[row]),
                    "selection": selected_by_row.get(row, ""),
                    "sampling_batch_index": batch_index,
                    "sampling_batch_seed": int(batch_seeds[batch_index]),
                    "target_rms": float(np.sqrt(np.mean(targets[row].astype(np.float64) ** 2))),
                    "ppg_rms": float(np.sqrt(np.mean(conditions[row].astype(np.float64) ** 2))),
                    "prediction_rms": float(
                        np.sqrt(np.mean(predictions[row].astype(np.float64) ** 2))
                    ),
                    "rmse": per_record["rmse"][row],
                    "mae": per_record["mae"][row],
                    "bias": per_record["bias"][row],
                    "pearson_r": per_record["pearson_r"][row],
                    "best_lag_samples": lag_values["best_lag_samples"][row],
                    "best_lag_ms": lag_values["best_lag_ms"][row],
                    "lag_adjusted_pearson_r": lag_values["lag_adjusted_pearson_r"][row],
                }
            )


def _plot_examples(
    output_path: Path,
    targets: np.ndarray,
    conditions: np.ndarray,
    predictions: np.ndarray,
    selected: Mapping[str, int],
    rmse: np.ndarray,
    sampling_rate: int,
) -> None:
    rows = ("ppg", "target", "zero", "prediction")
    figure, axes = plt.subplots(len(rows), len(selected), figsize=(15, 8), sharex=True, squeeze=False)
    time = np.arange(targets.shape[-1]) / sampling_rate
    for column, (label, index) in enumerate(selected.items()):
        ecg_limits = _limits([targets[index, 0], predictions[index, 0]])
        for row, name in enumerate(rows):
            axis = axes[row, column]
            if name == "ppg":
                values, color, title = conditions[index, 0], "#8064a2", "Condition PPG"
                axis.set_ylim(*_limits([values]))
            elif name == "target":
                values, color, title = targets[index, 0], "#111111", "Real ECG"
                axis.set_ylim(*ecg_limits)
            elif name == "zero":
                values, color, title = np.zeros_like(targets[index, 0]), "#7f7f7f", "Zero baseline"
                axis.set_ylim(*ecg_limits)
            else:
                values, color, title = predictions[index, 0], "#d17a00", "RDDM"
                axis.set_ylim(*ecg_limits)
            axis.plot(time, values, color=color, linewidth=0.9)
            axis.grid(alpha=0.18)
            if column == 0:
                axis.set_ylabel(title)
            if row == 0:
                axis.set_title(f"{label.capitalize()} | row {index} | RMSE {rmse[index]:.3f}")
            if row == len(rows) - 1:
                axis.set_xlabel("Time (s)")
    figure.suptitle("MIMIC-AFib deterministic RDDM waveform inspection (normalized domain)")
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))
    figure.savefig(output_path, dpi=240)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    _validate_checkpoint(checkpoint)
    config = dict(checkpoint["config"])
    normalization = dict(checkpoint["normalization"])
    dataset_dir = args.data_root.resolve() / "MIMIC-AFib"
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = _validate_dataset_manifest(manifest_path, config)

    _, test_set = get_ppg2ecg_datasets(
        DATA_PATH=str(args.data_root.resolve()),
        datasets=["MIMIC-AFib"],
        window_size=int(config["window_size"]),
        normalization_metadata=normalization,
        normalization_id=str(config["normalization_id"]),
        load_train=False,
    )
    if len(test_set) != args.expected_records:
        raise ValueError(f"expected {args.expected_records} held-out windows, found {len(test_set)}")
    targets = np.asarray(test_set.target_ecg[:, None, :], dtype=np.float32)
    conditions = np.asarray(test_set.condition_signal[:, None, :], dtype=np.float32)
    original_rows = np.load(dataset_dir / "kept_indices_test.npy", allow_pickle=False)
    if targets.shape != conditions.shape or targets.shape != (args.expected_records, 1, 512):
        raise ValueError("MIMIC held-out arrays must be aligned (1800,1,512) tensors")
    if original_rows.shape != (args.expected_records,) or len(np.unique(original_rows)) != len(original_rows):
        raise ValueError("kept test-row mapping is invalid")
    evaluated_records = args.max_records or args.expected_records
    if evaluated_records <= 0 or evaluated_records > args.expected_records:
        raise ValueError("max_records must be within the frozen held-out record count")
    targets = targets[:evaluated_records]
    conditions = conditions[:evaluated_records]
    original_rows = original_rows[:evaluated_records]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch cannot access CUDA")
    _set_deterministic(args.deterministic_seed, device)
    model = RDDM(
        eps_model=DiffusionUNetCrossAttention(
            512, 1, str(device), num_heads=int(config["attention_heads"])
        ),
        region_model=DiffusionUNetCrossAttention(
            512, 1, str(device), num_heads=int(config["attention_heads"])
        ),
        betas=(float(config["beta_start"]), float(config["beta_end"])),
        n_T=int(config["nT"]),
    ).to(device)
    condition_net_1 = ConditionNet().to(device)
    condition_net_2 = ConditionNet().to(device)
    model.load_state_dict(checkpoint["rddm_state"], strict=True)
    condition_net_1.load_state_dict(checkpoint["condition_1_state"], strict=True)
    condition_net_2.load_state_dict(checkpoint["condition_2_state"], strict=True)
    model.eval()
    condition_net_1.eval()
    condition_net_2.eval()
    checkpoint_metadata = {
        "kind": checkpoint["kind"],
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
    }
    del checkpoint
    gc.collect()

    predictions, batch_indices, batch_seeds = _generate_batches(
        model,
        condition_net_1,
        condition_net_2,
        conditions,
        args.batch_size,
        args.sampling_seed,
        device,
    )
    del model, condition_net_1, condition_net_2
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    summary, per_record = _waveform_metrics(targets, predictions)
    lag_summary, lag_values = _lag_diagnostic(
        targets, predictions, args.max_lag_samples, args.sampling_rate
    )
    summary["lag_adjusted_correlation_diagnostic"] = lag_summary
    baselines = {
        "zero": _baseline_summary(targets, np.zeros_like(targets)),
        "ppg_copy": _baseline_summary(targets, conditions),
    }
    selected = _select_examples(per_record["rmse"])
    _write_per_window_metrics(
        output_dir / "per_window_metrics.csv",
        original_rows,
        targets,
        conditions,
        predictions,
        per_record,
        lag_values,
        batch_indices,
        batch_seeds,
        selected,
    )
    _plot_examples(
        output_dir / "waveforms_best_median_worst.png",
        targets,
        conditions,
        predictions,
        selected,
        per_record["rmse"],
        args.sampling_rate,
    )
    np.savez_compressed(
        output_dir / "mimic_rddm_predictions.npz",
        targets=targets,
        conditions=conditions,
        rddm_predictions=predictions,
        source_test_rows_before_zero_filter=original_rows,
        sampling_batch_index=batch_indices,
        sampling_batch_seeds=batch_seeds,
    )
    _json(
        output_dir / "waveform_summary.json",
        {
            "model": {"rddm": summary},
            "baselines": baselines,
            "selection": {
                label: {
                    "test_row": row,
                    "source_test_row_before_zero_filter": int(original_rows[row]),
                    "rmse": float(per_record["rmse"][row]),
                }
                for label, row in selected.items()
            },
        },
    )

    artifact_names = [
        "mimic_rddm_predictions.npz",
        "per_window_metrics.csv",
        "waveform_summary.json",
        "waveforms_best_median_worst.png",
        "waveforms_best_median_worst.pdf",
    ]
    status = "completed" if evaluated_records == args.expected_records else "smoke_completed"
    protocol = {
        "schema_version": 1,
        "status": status,
        "command": shlex.join(sys.argv),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": "MIMIC-AFib",
            "dataset_version": config["dataset_version"],
            "split_hash": config["split_hash"],
            "heldout_windows_available": args.expected_records,
            "heldout_windows_evaluated": evaluated_records,
            "row_order": "QC artifact order with original pre-filter test rows saved",
            "normalization_id": config["normalization_id"],
            "sampling_rate_hz": args.sampling_rate,
            "window_seconds": int(config["window_size"]),
            "diffusion_steps": int(config["nT"]),
            "batch_size": args.batch_size,
            "sampling_seed_base": args.sampling_seed,
            "batch_seed_rule": "sampling_seed_base + zero_based_batch_index",
            "sampling_batch_seeds": batch_seeds.tolist(),
            "deterministic_seed": args.deterministic_seed,
            "prediction_sha256": _array_sha256(predictions),
            "target_sha256": _array_sha256(targets),
            "condition_sha256": _array_sha256(conditions),
            "source_row_mapping_sha256": _array_sha256(original_rows),
            "prediction_domain": "RDDM-compatible per-window normalized and NeuroKit-cleaned",
            "stochastic_protocol_boundary": (
                "exact replay requires the recorded batch size, row order, seed rule, software, "
                "device class, and checkpoint"
            ),
            "dataset_manifest_sha256": _sha256(manifest_path),
            "dataset_test_array_sha256": manifest["output_sha256"],
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            **checkpoint_metadata,
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
    print(json.dumps({"output_dir": str(output_dir), "status": status, "metrics": summary}, sort_keys=True))
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sampling_seed", type=int, default=2025)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    parser.add_argument("--sampling_rate", type=int, default=128)
    parser.add_argument("--max_lag_samples", type=int, default=64)
    parser.add_argument("--expected_records", type=int, default=1800)
    parser.add_argument("--max_records", type=int, default=None, help="Smoke-only evaluation cap.")
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
