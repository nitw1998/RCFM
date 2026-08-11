"""Evaluate PTB-XL conditional-OT, VP, target, and SB epoch-500 endpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import scipy
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import (
    _array_sha256,
    _checkpoint_contract,
    _sha256,
)
from scripts.evaluate_ptbxl_fourway import (
    EXPECTED_DATASET_VERSION,
    EXPECTED_SPLIT_HASH,
    TARGET_INDICES,
    TARGET_LEADS,
    _lag_diagnostic,
    _metric_summary,
    _set_deterministic,
    _validate_dataset_manifest,
)
from model import ConditionNet, DiffusionUNetCrossAttention
from rcfm import RegionAwareConditionalFlowMatching
from src.rcfm.checkpoint import load_checkpoint
from train_rcfm import build_datasets


MODEL_ORDER = ("conditional_ot", "vp", "target", "sb")
EXPECTED_EPOCH = 500
EXPECTED_GLOBAL_STEP = 68500


@torch.no_grad()
def _generate_ablation(
    checkpoint_path: Path,
    expected_kind: str,
    conditions: np.ndarray,
    initial_noise: np.ndarray,
    batch_size: int,
    steps: int,
    device: torch.device,
    deterministic_seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint["kind"] != expected_kind:
        raise ValueError(f"checkpoint kind changed while loading {checkpoint_path.name}")
    config, output_spec = checkpoint["config"], checkpoint["output_spec"]
    signal_length = int(output_spec["length"])
    output_channels = int(output_spec["channels"])
    if initial_noise.shape != (len(conditions), output_channels, signal_length):
        raise ValueError("saved initial noise disagrees with checkpoint output shape")

    _set_deterministic(deterministic_seed, device)
    condition_net = ConditionNet().to(device)
    flow_network = DiffusionUNetCrossAttention(
        signal_length,
        output_channels,
        device=str(device),
        num_heads=int(config["attention_heads"]),
    ).to(device)
    model = RegionAwareConditionalFlowMatching(
        flow_model=flow_network,
        flow_matcher_type=str(config["flow_matcher"]),
        sigma=float(config["sigma"]),
        region_weight=float(config["region_weight"]),
        use_minibatch_ot=bool(config["use_minibatch_ot"]),
        ot_method=str(config["ot_method"]),
        ot_reg=float(config["ot_reg"]),
        ot_normalize_cost=bool(config["ot_normalize_cost"]),
        ot_diagnostics=bool(config["ot_diagnostics"]),
        ot_strict_mode=bool(config["ot_strict_mode"]),
        ot_sampling_strategy=str(config["ot_sampling_strategy"]),
        association_debug=bool(config.get("association_debug", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    condition_net.load_state_dict(checkpoint["condition_state"], strict=True)
    model.eval()
    condition_net.eval()
    metadata = {
        "kind": checkpoint["kind"],
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "best_metrics": dict(checkpoint["best_metrics"]),
        "inference_ot_calls": 0,
    }
    del checkpoint
    gc.collect()

    predictions = np.empty_like(initial_noise)
    for start in range(0, len(conditions), batch_size):
        stop = min(start + batch_size, len(conditions))
        condition_batch = torch.from_numpy(conditions[start:stop]).to(device=device)
        noise_batch = torch.from_numpy(initial_noise[start:stop]).to(device=device)
        prediction = model.sample(
            conditions=condition_net(condition_batch),
            shape=tuple(noise_batch.shape),
            steps=steps,
            device=device,
            initial_noise=noise_batch,
        )
        predictions[start:stop] = prediction.cpu().numpy()
        print(f"{config['flow_matcher']}: generated {stop}/{len(conditions)}", flush=True)
    if model._last_source_indices is not None or model._last_target_indices is not None:
        raise RuntimeError("inference unexpectedly invoked minibatch OT coupling")
    del model, flow_network, condition_net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("generated predictions contain NaN or Inf")
    return predictions, metadata


def _validate_contracts(contracts: Mapping[str, Mapping[str, object]]) -> None:
    if tuple(contracts) != MODEL_ORDER:
        raise ValueError("contracts must follow conditional_ot, vp, target, sb order")
    reference = contracts["conditional_ot"]
    common_fields = (
        "task",
        "datasets",
        "dataset_version",
        "split_hash",
        "normalization_id",
        "alignment_id",
        "condition_lead_index",
        "target_lead_indices",
        "window_size",
        "attention_heads",
        "region_weight",
        "seed",
    )
    for name, contract in contracts.items():
        if int(contract["epoch"]) != EXPECTED_EPOCH or int(contract["global_step"]) != EXPECTED_GLOBAL_STEP:
            raise ValueError(f"{name} is not the frozen epoch-500 endpoint")
        mismatched = [
            field
            for field in common_fields
            if contract["config"].get(field) != reference["config"].get(field)
        ]
        if mismatched:
            raise ValueError(f"{name} differs on common protocol fields: {', '.join(mismatched)}")
        if contract["normalization"] != reference["normalization"]:
            raise ValueError(f"{name} normalization metadata differs")
        if contract["output_spec"] != reference["output_spec"]:
            raise ValueError(f"{name} output specification differs")

    config = reference["config"]
    expected = {
        "task": "ecg2ecg",
        "datasets": ["PTBXL"],
        "dataset_version": EXPECTED_DATASET_VERSION,
        "split_hash": EXPECTED_SPLIT_HASH,
        "normalization_id": "record_minmax_neg1_1_v1",
        "alignment_id": "ptbxl_official_folds_first4s_same_record_lead_II_to_other11_v1",
        "condition_lead_index": 1,
        "target_lead_indices": list(TARGET_INDICES),
        "window_size": 4,
        "region_weight": 0.01,
        "seed": 31,
    }
    bad = [field for field, value in expected.items() if config.get(field) != value]
    output = reference["output_spec"]
    if bad or int(output.get("channels", 0)) != 11 or int(output.get("length", 0)) != 512:
        raise ValueError("checkpoints violate the frozen PTB-XL contract: " + ", ".join(bad))
    if tuple(output.get("target_leads", ())) != TARGET_LEADS:
        raise ValueError("checkpoints have the wrong target-lead order")

    expected_factors = {
        "conditional_ot": {
            "kind": "canonical_multistep_rcfm",
            "experiment_role": None,
            "flow_matcher": "conditional",
            "sigma": 0.0,
            "use_minibatch_ot": True,
            "ot_method": "exact",
            "ot_sampling_strategy": "assignment",
        },
        "vp": {
            "kind": "path_ablation_rcfm",
            "experiment_role": "path_ablation",
            "flow_matcher": "vp",
            "sigma": 0.1,
            "use_minibatch_ot": False,
            "ot_method": "exact",
            "ot_sampling_strategy": "multinomial",
        },
        "target": {
            "kind": "path_ablation_rcfm",
            "experiment_role": "path_ablation",
            "flow_matcher": "target",
            "sigma": 0.1,
            "use_minibatch_ot": False,
            "ot_method": "exact",
            "ot_sampling_strategy": "multinomial",
        },
        "sb": {
            "kind": "path_ablation_rcfm",
            "experiment_role": "path_ablation",
            "flow_matcher": "sb",
            "sigma": 0.1,
            "use_minibatch_ot": True,
            "ot_method": "exact",
            "ot_sampling_strategy": "multinomial",
        },
    }
    for name, factors in expected_factors.items():
        contract, model_config = contracts[name], contracts[name]["config"]
        if contract["kind"] != factors["kind"]:
            raise ValueError(f"{name} checkpoint kind changed")
        mismatched = [
            field
            for field, value in factors.items()
            if field != "kind" and model_config.get(field) != value
        ]
        if mismatched:
            raise ValueError(f"{name} path/coupling factors changed: {', '.join(mismatched)}")


def _write_per_record(
    path: Path,
    record_ids: np.ndarray,
    patient_ids: np.ndarray,
    values: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    metrics = ("rmse", "mae", "bias", "pearson_r")
    fields = ["record_id", "patient_id"] + [
        f"{model}_{metric}" for model in MODEL_ORDER for metric in metrics
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (record_id, patient_id) in enumerate(zip(record_ids, patient_ids)):
            row: dict[str, object] = {
                "record_id": str(record_id),
                "patient_id": str(patient_id),
            }
            for model in MODEL_ORDER:
                for metric in metrics:
                    row[f"{model}_{metric}"] = float(values[model][metric][index])
            writer.writerow(row)


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    checkpoint_paths = {
        "conditional_ot": args.conditional_ot_checkpoint,
        "vp": args.vp_checkpoint,
        "target": args.target_checkpoint,
        "sb": args.sb_checkpoint,
    }
    contracts = {name: _checkpoint_contract(path) for name, path in checkpoint_paths.items()}
    _validate_contracts(contracts)

    dataset_dir = args.data_root.resolve() / "PTBXL"
    manifest = _validate_dataset_manifest(
        dataset_dir / "dataset_manifest.json", args.expected_records
    )
    config = contracts["conditional_ot"]["config"]
    _, test_set = build_datasets(
        config["task"],
        config["datasets"],
        str(args.data_root.resolve()),
        int(config["window_size"]),
        normalization_metadata=contracts["conditional_ot"]["normalization"],
        normalization_id=config["normalization_id"],
        condition_lead_index=1,
        target_lead_indices=list(TARGET_INDICES),
        load_train=False,
        heldout_split="test",
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
    generator = torch.Generator(device="cpu").manual_seed(args.noise_seed)
    initial_noise = torch.randn((count, 11, 512), generator=generator).numpy()
    np.savez_compressed(
        output_dir / "paired_reference.npz",
        targets=targets,
        conditions=conditions,
        record_ids=record_ids,
        patient_ids=patient_ids,
        initial_flow_noise=initial_noise,
    )

    predictions: dict[str, np.ndarray] = {}
    generation: dict[str, object] = {}
    for name in MODEL_ORDER:
        predictions[name], metadata = _generate_ablation(
            checkpoint_paths[name],
            str(contracts[name]["kind"]),
            conditions,
            initial_noise,
            args.batch_size,
            args.steps,
            device,
            args.deterministic_seed,
        )
        np.save(output_dir / f"{name}_predictions.npy", predictions[name], allow_pickle=False)
        generation[name] = {
            **metadata,
            "prediction_sha256": _array_sha256(predictions[name]),
            "selection_policy": "predeclared_epoch_500_endpoint",
        }

    summaries: dict[str, object] = {}
    per_record: dict[str, Mapping[str, np.ndarray]] = {}
    lag: dict[str, object] = {}
    for name in MODEL_ORDER:
        summaries[name], per_record[name] = _metric_summary(targets, predictions[name])
        lag[name] = _lag_diagnostic(targets, predictions[name], args.max_lag_samples)
    baselines = {}
    for name, values in {
        "zero": np.zeros_like(targets),
        "lead_ii_copy": np.repeat(conditions, 11, axis=1),
    }.items():
        baselines[name], _ = _metric_summary(targets, values)
    _write_per_record(
        output_dir / "per_record_metrics.csv", record_ids, patient_ids, per_record
    )
    (output_dir / "waveform_summary.json").write_text(
        json.dumps(
            {"models": summaries, "baselines": baselines, "lag_diagnostic": lag},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    status = "completed" if count == args.expected_records else "smoke_completed"
    protocol = {
        "schema_version": 1,
        "status": status,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "dataset": "PTB-XL",
            "dataset_version": EXPECTED_DATASET_VERSION,
            "split": "official_fold_10",
            "split_hash": EXPECTED_SPLIT_HASH,
            "available_records": args.expected_records,
            "evaluated_records": count,
            "patient_count": int(len(np.unique(patient_ids))),
            "sampling_rate_hz": 128,
            "window_seconds": 4,
            "condition_lead": "II",
            "target_leads": list(TARGET_LEADS),
            "normalization_id": "record_minmax_neg1_1_v1",
            "primary_alignment": "raw_synchronized_same_record_same_start_no_shift",
            "phase_correction_applied": False,
            "flow_nfe": args.steps,
            "flow_noise_seed": args.noise_seed,
            "same_initial_noise_across_all_models": True,
            "ablation_scope": "probability_path_and_endpoint_coupling_not_four_OT_solvers",
        },
        "selection": {model: "predeclared_epoch_500_endpoint" for model in MODEL_ORDER},
        "factor_warning": (
            "Conditional-OT uses conditional path, sigma=0, exact assignment coupling; "
            "VP and target use sigma=0.1 without OT; SB uses sigma=0.1 and exact "
            "multinomial coupling. Path, sigma, and coupling sampling are not fully isolated."
        ),
        "clinical_boundaries": {
            "hrv": "blocked_four_second_records",
            "physical_inverse": "ground_truth_target_min_range_is_oracle_only",
            "ptbxl_plus_labels": "not_used",
        },
        "checkpoints": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "factors": {
                    key: contracts[name]["config"].get(key)
                    for key in (
                        "flow_matcher",
                        "sigma",
                        "use_minibatch_ot",
                        "ot_method",
                        "ot_sampling_strategy",
                    )
                },
                **generation[name],
            }
            for name, path in checkpoint_paths.items()
        },
        "artifacts": {
            "paired_reference_sha256": _sha256(output_dir / "paired_reference.npz"),
            **{
                f"{name}_predictions_sha256": _sha256(
                    output_dir / f"{name}_predictions.npy"
                )
                for name in MODEL_ORDER
            },
        },
        "manifest_source": {
            "path": str((dataset_dir / "dataset_manifest.json").resolve()),
            "sha256": _sha256(dataset_dir / "dataset_manifest.json"),
            "eligible_records": manifest["eligible_records"],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(f"PTB-XL path/coupling ablation evaluation {status}: {output_dir}", flush=True)
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditional_ot_checkpoint", type=Path, required=True)
    parser.add_argument("--vp_checkpoint", type=Path, required=True)
    parser.add_argument("--target_checkpoint", type=Path, required=True)
    parser.add_argument("--sb_checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--noise_seed", type=int, default=2025)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    parser.add_argument("--expected_records", type=int, default=2203)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
