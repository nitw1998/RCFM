"""Deterministically evaluate PTB-XL CFM, RCFM, RCFM-OT, and adapted RDDM.

The primary protocol uses the synchronized, unshifted fold-10 lead pairs.  A
bounded target-informed lag search is saved only as a timing-error diagnostic.
"""

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

import numpy as np
import scipy
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion import RDDM
from model import ConditionNet, DiffusionUNetCrossAttention
from scripts.evaluate_cpsc_zscore_paired import (
    _array_sha256,
    _checkpoint_contract,
    _generate,
    _json,
    _pearson_rows,
    _sha256,
)
from src.rcfm.metrics.bland_altman import bland_altman, paired_correlation
from src.rcfm.metrics.waveform import waveform_frechet_distance
from train_rcfm import build_datasets


MODEL_ORDER = ("cfm", "rcfm", "rcfm_ot", "rddm")
TARGET_INDICES = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
TARGET_LEADS = ("I", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
EXPECTED_SPLIT_HASH = "7784c98a2c8daccc23fc7cb0d47dc1933eeee77f120c05f8a24ac149cd8474f7"
EXPECTED_DATASET_VERSION = "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1"
EXPECTED_UPSTREAM_COMMIT = "7d5348843c3985c211a23ae5105a2d9497d5156a"


def _set_deterministic(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _validate_flow_contracts(contracts: Mapping[str, Mapping[str, object]]) -> None:
    if set(contracts) != {"cfm", "rcfm", "rcfm_ot"}:
        raise ValueError("flow contracts must contain cfm, rcfm, and rcfm_ot")
    expected_kinds = {
        "cfm": "canonical_multistep_cfm",
        "rcfm": "canonical_multistep_rcfm",
        "rcfm_ot": "canonical_multistep_rcfm",
    }
    reference = contracts["cfm"]
    matched_config = (
        "task", "datasets", "dataset_version", "split_hash", "normalization_id",
        "alignment_id", "condition_lead_index", "target_lead_indices", "window_size",
        "attention_heads", "flow_matcher", "sigma", "seed",
    )
    for name, contract in contracts.items():
        if contract["kind"] != expected_kinds[name]:
            raise ValueError(f"{name} has the wrong checkpoint kind")
        mismatched = [
            key for key in matched_config
            if contract["config"].get(key) != reference["config"].get(key)
        ]
        if mismatched:
            raise ValueError(f"{name} flow contract differs on: {', '.join(mismatched)}")
        if contract["normalization"] != reference["normalization"]:
            raise ValueError(f"{name} normalization metadata differs")
        if contract["output_spec"] != reference["output_spec"]:
            raise ValueError(f"{name} output specification differs")
    config = reference["config"]
    output = reference["output_spec"]
    expected = {
        "task": "ecg2ecg", "datasets": ["PTBXL"],
        "dataset_version": EXPECTED_DATASET_VERSION, "split_hash": EXPECTED_SPLIT_HASH,
        "normalization_id": "record_minmax_neg1_1_v1", "condition_lead_index": 1,
        "target_lead_indices": list(TARGET_INDICES), "window_size": 4,
    }
    bad = [key for key, value in expected.items() if config.get(key) != value]
    if bad or int(output.get("channels", 0)) != 11 or int(output.get("length", 0)) != 512:
        raise ValueError("flow checkpoint violates the frozen PTB-XL contract: " + ", ".join(bad))
    if tuple(output.get("target_leads", ())) != TARGET_LEADS:
        raise ValueError("flow checkpoint has the wrong target-lead order")
    cfm, rcfm, rcfm_ot = (contracts[name]["config"] for name in ("cfm", "rcfm", "rcfm_ot"))
    if float(cfm.get("region_weight", -1)) != 0 or bool(cfm.get("use_minibatch_ot")):
        raise ValueError("CFM must have region_weight=0 and OT disabled")
    if float(rcfm.get("region_weight", 0)) <= 0 or bool(rcfm.get("use_minibatch_ot")):
        raise ValueError("RCFM must have positive region weight and OT disabled")
    if float(rcfm_ot.get("region_weight", 0)) != float(rcfm["region_weight"]):
        raise ValueError("RCFM and RCFM-OT region weights must match")
    if not bool(rcfm_ot.get("use_minibatch_ot")) or rcfm_ot.get("ot_method") != "exact":
        raise ValueError("RCFM-OT must use exact minibatch OT")


def _validate_rddm_checkpoint(checkpoint: Mapping[str, object]) -> None:
    required = {
        "schema_version", "kind", "epoch", "global_step", "rddm_state",
        "condition_1_state", "condition_2_state", "config", "normalization", "provenance",
    }
    if missing := sorted(required - set(checkpoint)):
        raise ValueError("RDDM checkpoint is missing: " + ", ".join(missing))
    if checkpoint["schema_version"] != 1 or checkpoint["kind"] != "independent_rddm_reproduction":
        raise ValueError("invalid RDDM checkpoint schema or kind")
    config = checkpoint["config"]
    expected = {
        "task": "ecg2ecg", "datasets": ["PTBXL"],
        "dataset_version": EXPECTED_DATASET_VERSION, "split_hash": EXPECTED_SPLIT_HASH,
        "normalization_id": "record_minmax_neg1_1_v1", "alignment_id":
        "ptbxl_official_folds_first4s_same_record_lead_II_to_other11_v1",
        "condition_lead_index": 1, "target_lead_indices": list(TARGET_INDICES),
        "target_channels": 11, "window_size": 4, "nT": 10, "attention_heads": 8,
        "seed": 31, "reproduction_label": "RDDM-ECG (adapted)",
    }
    bad = [key for key, value in expected.items() if config.get(key) != value]
    if bad:
        raise ValueError("RDDM checkpoint violates the frozen PTB-XL contract: " + ", ".join(bad))
    if int(checkpoint["epoch"]) != 500 or int(checkpoint["global_step"]) != 68500:
        raise ValueError("RDDM evaluation requires the predeclared epoch-500 endpoint")
    if float(config.get("beta_start", -1)) != 1e-4 or float(config.get("beta_end", -1)) != 0.2:
        raise ValueError("RDDM beta schedule changed")
    if checkpoint["provenance"].get("upstream_commit") != EXPECTED_UPSTREAM_COMMIT:
        raise ValueError("RDDM upstream commit changed")


def _validate_dataset_manifest(path: Path, expected_records: int) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset_version": EXPECTED_DATASET_VERSION,
        "split_hash": EXPECTED_SPLIT_HASH,
        "split_method": "official_ptbxl_strat_fold_1_8_train_9_val_10_test",
        "patient_disjoint_verified": True,
    }
    bad = [key for key, value in expected.items() if manifest.get(key) != value]
    test = manifest.get("splits", {}).get("test", {})
    if bad or test.get("folds") != [10] or int(test.get("records", -1)) != expected_records:
        raise ValueError("dataset manifest violates the frozen fold-10 contract")
    return manifest


@torch.no_grad()
def _sample_rddm(model: RDDM, cond1: Mapping[str, object], cond2: Mapping[str, object], shape: tuple[int, int, int]) -> torch.Tensor:
    """Apply the upstream sampler with an explicit multi-lead tensor shape."""

    batch, channels, length = shape
    feature = cond1["down_conditions"][-1]
    device = feature.device
    if int(feature.shape[0]) != batch or min(shape) <= 0:
        raise ValueError("RDDM condition batch and requested output shape disagree")
    sample = torch.randn(shape, device=device)
    for step in range(model.n_T, 0, -1):
        noise = torch.randn(shape, device=device) if step > 1 else 0
        time = torch.full((batch,), step / model.n_T, device=device)
        sample = model.region_model(sample, cond2, time)
        epsilon = model.eps_model(sample, cond1, time)
        sample = (
            model.oneover_sqrta[step] * (sample - epsilon * model.mab_over_sqrtmab[step])
            + model.sqrt_beta_t[step] * noise
        )
    if tuple(sample.shape) != shape:
        raise ValueError("RDDM sampler returned the wrong output shape")
    return sample


@torch.no_grad()
def _generate_rddm(
    checkpoint_path: Path,
    conditions: np.ndarray,
    batch_size: int,
    sampling_seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    _validate_rddm_checkpoint(checkpoint)
    config = checkpoint["config"]
    channels, length = int(config["target_channels"]), int(config["window_size"]) * 128
    model = RDDM(
        eps_model=DiffusionUNetCrossAttention(length, channels, str(device), num_heads=int(config["attention_heads"])),
        region_model=DiffusionUNetCrossAttention(length, channels, str(device), num_heads=int(config["attention_heads"])),
        betas=(float(config["beta_start"]), float(config["beta_end"])), n_T=int(config["nT"]),
    ).to(device)
    condition_1, condition_2 = ConditionNet().to(device), ConditionNet().to(device)
    model.load_state_dict(checkpoint["rddm_state"], strict=True)
    condition_1.load_state_dict(checkpoint["condition_1_state"], strict=True)
    condition_2.load_state_dict(checkpoint["condition_2_state"], strict=True)
    model.eval(); condition_1.eval(); condition_2.eval()
    metadata = {"kind": checkpoint["kind"], "epoch": int(checkpoint["epoch"]), "global_step": int(checkpoint["global_step"])}
    del checkpoint
    gc.collect()
    predictions = np.empty((len(conditions), channels, length), dtype=np.float32)
    batch_seeds = []
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    for batch_index, start in enumerate(range(0, len(conditions), batch_size)):
        stop = min(start + batch_size, len(conditions))
        seed = sampling_seed + batch_index
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            condition = torch.from_numpy(conditions[start:stop]).to(device)
            generated = _sample_rddm(
                model, condition_1(condition), condition_2(condition),
                (stop - start, channels, length),
            )
        predictions[start:stop] = generated.cpu().numpy()
        batch_seeds.append(seed)
        print(f"rddm: generated {stop}/{len(conditions)} with batch_seed={seed}", flush=True)
    del model, condition_1, condition_2
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("RDDM predictions contain NaN or Inf")
    metadata["sampling_batch_seeds"] = batch_seeds
    return predictions, metadata


def _metric_summary(reference: np.ndarray, generated: np.ndarray, include_fd: bool = True) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if reference.shape != generated.shape or reference.ndim != 3:
        raise ValueError("metrics require matching (records, leads, samples) arrays")
    error = generated.astype(np.float64) - reference.astype(np.float64)
    per_record = {
        "rmse": np.sqrt(np.mean(error ** 2, axis=(1, 2))),
        "mae": np.mean(np.abs(error), axis=(1, 2)),
        "bias": np.mean(error, axis=(1, 2)),
        "pearson_r": _pearson_rows(reference, generated),
    }
    point_correlation = paired_correlation(reference.reshape(-1), generated.reshape(-1))
    point_correlation.update({"p_value": None, "inference_status": "blocked_autocorrelated_time_samples"})
    agreement = bland_altman(reference.reshape(-1), generated.reshape(-1))
    agreement.pop("pair_means"); agreement.pop("differences")
    per_lead = {}
    lead_fds = []
    for index, lead in enumerate(TARGET_LEADS):
        lead_error = error[:, index]
        lead_correlation = paired_correlation(reference[:, index].reshape(-1), generated[:, index].reshape(-1))
        lead_correlation.update({"p_value": None, "inference_status": "blocked_autocorrelated_time_samples"})
        lead_agreement = bland_altman(reference[:, index].reshape(-1), generated[:, index].reshape(-1))
        lead_agreement.pop("pair_means"); lead_agreement.pop("differences")
        fd = waveform_frechet_distance(reference[:, index], generated[:, index]) if include_fd else None
        if fd is not None:
            lead_fds.append(fd)
        row_correlation = _pearson_rows(reference[:, index:index + 1], generated[:, index:index + 1])
        finite_row_correlation = row_correlation[np.isfinite(row_correlation)]
        per_lead[lead] = {
            "rmse": float(np.sqrt(np.mean(lead_error ** 2))), "mae": float(np.mean(np.abs(lead_error))),
            "bias": float(np.mean(lead_error)), "waveform_fd": fd,
            "pointwise_correlation_descriptive_only": lead_correlation,
            "pointwise_bland_altman_descriptive_only": lead_agreement,
            "per_record_pearson_mean": float(np.mean(finite_row_correlation)) if len(finite_row_correlation) else None,
            "per_record_pearson_median": float(np.median(finite_row_correlation)) if len(finite_row_correlation) else None,
        }
    finite_record_correlation = per_record["pearson_r"][np.isfinite(per_record["pearson_r"])]
    summary = {
        "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)), "waveform_fd_macro_lead": float(np.mean(lead_fds)) if lead_fds else None,
        "waveform_fd_definition": "mean of 11 independent 512-sample lead FDs",
        "pointwise_correlation_descriptive_only": point_correlation,
        "pointwise_bland_altman_descriptive_only": agreement,
        "per_record_pearson_mean": float(np.mean(finite_record_correlation)) if len(finite_record_correlation) else None,
        "per_record_pearson_median": float(np.median(finite_record_correlation)) if len(finite_record_correlation) else None,
        "target_rms": float(np.sqrt(np.mean(reference.astype(np.float64) ** 2))),
        "prediction_rms": float(np.sqrt(np.mean(generated.astype(np.float64) ** 2))),
        "prediction_min": float(np.min(generated)), "prediction_max": float(np.max(generated)),
        "prediction_outside_neg1_1_fraction": float(np.mean(np.abs(generated) > 1.0)),
        "per_lead": per_lead,
    }
    return summary, per_record


def _lag_diagnostic(reference: np.ndarray, generated: np.ndarray, max_lag: int) -> dict[str, object]:
    """Summarize oracle shifts without producing corrected evaluation arrays."""

    if reference.shape != generated.shape or max_lag <= 0 or 2 * max_lag >= reference.shape[-1]:
        raise ValueError("invalid lag diagnostic inputs")
    shifts, raw_r, best_r = [], [], []
    left, right = max_lag, reference.shape[-1] - max_lag
    for record in range(len(reference)):
        for lead in range(reference.shape[1]):
            target = reference[record, lead].astype(np.float64)
            prediction = generated[record, lead].astype(np.float64)
            correlations = []
            for shift in range(-max_lag, max_lag + 1):
                correlations.append(_pearson_rows(target[None, None, left:right], prediction[None, None, left-shift:right-shift])[0])
            values = np.asarray(correlations)
            best_index = int(np.nanargmax(values)) if np.any(np.isfinite(values)) else max_lag
            shifts.append(best_index - max_lag)
            raw_r.append(values[max_lag]); best_r.append(values[best_index])
    shifts_array = np.asarray(shifts, dtype=np.int16)
    return {
        "status": "diagnostic_only_target_informed_not_applied_to_primary_metrics",
        "max_lag_samples": max_lag,
        "pairs": len(shifts),
        "median_signed_lag_samples": float(np.median(shifts_array)),
        "median_absolute_lag_samples": float(np.median(np.abs(shifts_array))),
        "boundary_hit_fraction": float(np.mean(np.abs(shifts_array) == max_lag)),
        "raw_fixed_support_pearson_median": float(np.nanmedian(raw_r)),
        "oracle_best_pearson_median": float(np.nanmedian(best_r)),
    }


def _write_per_record(path: Path, record_ids: np.ndarray, values: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    fields = ["record_id"] + [f"{model}_{metric}" for model in MODEL_ORDER for metric in ("rmse", "mae", "bias", "pearson_r")]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index, record_id in enumerate(record_ids):
            row = {"record_id": str(record_id)}
            for model in MODEL_ORDER:
                for metric in ("rmse", "mae", "bias", "pearson_r"):
                    row[f"{model}_{metric}"] = float(values[model][metric][index])
            writer.writerow(row)


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    flow_paths = {"cfm": args.cfm_checkpoint, "rcfm": args.rcfm_checkpoint, "rcfm_ot": args.rcfm_ot_checkpoint}
    contracts = {name: _checkpoint_contract(path) for name, path in flow_paths.items()}
    _validate_flow_contracts(contracts)
    rddm_checkpoint = torch.load(args.rddm_checkpoint, map_location="cpu")
    _validate_rddm_checkpoint(rddm_checkpoint)
    rddm_contract = {"epoch": int(rddm_checkpoint["epoch"]), "global_step": int(rddm_checkpoint["global_step"]), "config": dict(rddm_checkpoint["config"])}
    del rddm_checkpoint; gc.collect()
    dataset_dir = args.data_root.resolve() / "PTBXL"
    manifest = _validate_dataset_manifest(dataset_dir / "dataset_manifest.json", args.expected_records)
    config = contracts["cfm"]["config"]
    _, test_set = build_datasets(
        config["task"], config["datasets"], str(args.data_root.resolve()), int(config["window_size"]),
        normalization_metadata=contracts["cfm"]["normalization"], normalization_id=config["normalization_id"],
        condition_lead_index=1, target_lead_indices=list(TARGET_INDICES), load_train=False, heldout_split="test",
    )
    if len(test_set) != args.expected_records:
        raise ValueError(f"expected {args.expected_records} fold-10 records, found {len(test_set)}")
    count = args.expected_records if args.max_records is None else min(args.max_records, args.expected_records)
    targets = np.asarray(test_set.target_ecg[:count], dtype=np.float32)
    conditions = np.asarray(test_set.condition_signal[:count, None, :], dtype=np.float32)
    record_ids = np.asarray(test_set.record_ids[:count])
    patient_ids = np.load(dataset_dir / "patient_ids_test.npy", allow_pickle=False)[:count]
    if targets.shape != (count, 11, 512) or conditions.shape != (count, 1, 512):
        raise ValueError("PTB-XL test tensors have the wrong shape")
    if len(np.unique(record_ids)) != count:
        raise ValueError("PTB-XL test record IDs are not unique")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    _set_deterministic(args.deterministic_seed, device)
    noise_generator = torch.Generator(device="cpu"); noise_generator.manual_seed(args.noise_seed)
    initial_noise = torch.randn((count, 11, 512), generator=noise_generator).numpy()
    np.savez_compressed(output_dir / "paired_reference.npz", targets=targets, conditions=conditions, record_ids=record_ids, patient_ids=patient_ids, initial_flow_noise=initial_noise)
    predictions: dict[str, np.ndarray] = {}
    generation: dict[str, object] = {}
    kinds = {"cfm": "canonical_multistep_cfm", "rcfm": "canonical_multistep_rcfm", "rcfm_ot": "canonical_multistep_rcfm"}
    for name in ("cfm", "rcfm", "rcfm_ot"):
        predictions[name], metadata = _generate(flow_paths[name], kinds[name], conditions, initial_noise, args.batch_size, args.steps, device, args.deterministic_seed)
        np.save(output_dir / f"{name}_predictions.npy", predictions[name], allow_pickle=False)
        generation[name] = {**metadata, "prediction_sha256": _array_sha256(predictions[name])}
    predictions["rddm"], metadata = _generate_rddm(args.rddm_checkpoint, conditions, args.batch_size, args.rddm_sampling_seed, device)
    np.save(output_dir / "rddm_predictions.npy", predictions["rddm"], allow_pickle=False)
    generation["rddm"] = {**metadata, "prediction_sha256": _array_sha256(predictions["rddm"]), "selection_policy": "predeclared_epoch_500_endpoint_without_validation_selection"}
    summaries, per_record, lag = {}, {}, {}
    for name in MODEL_ORDER:
        summaries[name], per_record[name] = _metric_summary(targets, predictions[name])
        lag[name] = _lag_diagnostic(targets, predictions[name], args.max_lag_samples)
    baselines = {}
    for name, values in {"zero": np.zeros_like(targets), "lead_ii_copy": np.repeat(conditions, 11, axis=1)}.items():
        baselines[name], _ = _metric_summary(targets, values, include_fd=True)
    _write_per_record(output_dir / "per_record_metrics.csv", record_ids, per_record)
    _json(output_dir / "waveform_summary.json", {"models": summaries, "baselines": baselines, "lag_diagnostic": lag})
    status = "completed" if count == args.expected_records else "smoke_completed"
    protocol = {
        "schema_version": 1, "status": status, "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv),
        "protocol": {
            "dataset": "PTB-XL", "dataset_version": EXPECTED_DATASET_VERSION,
            "split": "official_fold_10", "split_hash": EXPECTED_SPLIT_HASH,
            "available_records": args.expected_records, "evaluated_records": count,
            "patient_count": int(len(np.unique(patient_ids))), "sampling_rate_hz": 128,
            "window_seconds": 4, "condition_lead": "II", "target_leads": list(TARGET_LEADS),
            "normalization_id": "record_minmax_neg1_1_v1", "primary_alignment": "raw_synchronized_same_record_same_start_no_shift",
            "phase_correction_applied": False, "lag_diagnostic": "target-informed descriptive only",
            "flow_nfe": args.steps, "flow_noise_seed": args.noise_seed,
            "rddm_steps": int(rddm_contract["config"]["nT"]), "rddm_sampling_seed": args.rddm_sampling_seed,
            "same_flow_noise_across_flow_models": True,
        },
        "selection": {
            "cfm": "fold_9_best_rmse_epoch_450", "rcfm": "fold_9_best_rmse_epoch_400",
            "rcfm_ot": "fold_9_best_rmse_epoch_425", "rddm": "epoch_500_endpoint_no_validation_selection",
        },
        "clinical_boundaries": {
            "hrv": "blocked_four_second_records", "physical_inverse": "ground_truth_target_min_range_is_oracle_only",
            "intervals": "pending_separate_per_lead_delineation", "ptbxl_plus_labels": "not_used",
        },
        "checkpoints": {name: {"path": str(path.resolve()), "sha256": _sha256(path), **generation[name]} for name, path in {**flow_paths, "rddm": args.rddm_checkpoint}.items()},
        "artifacts": {
            "paired_reference_sha256": _sha256(output_dir / "paired_reference.npz"),
            **{f"{name}_predictions_sha256": _sha256(output_dir / f"{name}_predictions.npy") for name in MODEL_ORDER},
        },
        "manifest_source": {"path": str((dataset_dir / "dataset_manifest.json").resolve()), "sha256": _sha256(dataset_dir / "dataset_manifest.json"), "eligible_records": manifest["eligible_records"]},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
    }
    _json(output_dir / "protocol.json", protocol)
    print(f"PTB-XL four-way evaluation {status}: {output_dir}", flush=True)
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfm_checkpoint", type=Path, required=True)
    parser.add_argument("--rcfm_checkpoint", type=Path, required=True)
    parser.add_argument("--rcfm_ot_checkpoint", type=Path, required=True)
    parser.add_argument("--rddm_checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--noise_seed", type=int, default=2025)
    parser.add_argument("--rddm_sampling_seed", type=int, default=2025)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    parser.add_argument("--expected_records", type=int, default=2203)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
