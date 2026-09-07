"""Deterministically evaluate CPSC2018 CFM, RCFM, RCFM-OT, and adapted RDDM."""

from __future__ import annotations

import argparse
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

from diffusion import RDDM
from model import ConditionNet, DiffusionUNetCrossAttention
from scripts.evaluate_cpsc_zscore_paired import (
    _array_sha256,
    _checkpoint_contract,
    _generate,
    _json,
    _sha256,
)
from scripts.evaluate_ptbxl_fourway import (
    MODEL_ORDER,
    TARGET_INDICES,
    TARGET_LEADS,
    _metric_summary,
    _sample_rddm,
    _set_deterministic,
    _write_per_record,
)
from train_rcfm import build_datasets


EXPECTED_DATASET_VERSION = "cpsc2018-source-derived-all12lead-qc-v3-record-minmax-neg1-1"
EXPECTED_SPLIT_HASH = "35e0a796a60e4d6979b1f050048495fdf1826eedda7d40e47a56d4dcd5874223"
EXPECTED_ALIGNMENT = "same_record_simultaneous_channels_first_4s_lead_II_to_other_11_v3"
EXPECTED_UPSTREAM_COMMIT = "7d5348843c3985c211a23ae5105a2d9497d5156a"


def _validate_flow_contracts(
    contracts: Mapping[str, Mapping[str, object]], expected_training_seed: int = 31
) -> None:
    expected_kinds = {
        "cfm": "canonical_multistep_cfm",
        "rcfm": "canonical_multistep_rcfm",
        "rcfm_ot": "canonical_multistep_rcfm",
    }
    if set(contracts) != set(expected_kinds):
        raise ValueError("flow contracts must contain cfm, rcfm, and rcfm_ot")
    reference = contracts["cfm"]
    matched = (
        "task", "datasets", "dataset_version", "split_hash", "normalization_id",
        "alignment_id", "condition_lead_index", "target_lead_indices", "window_size",
        "attention_heads", "flow_matcher", "sigma", "seed",
    )
    for name, contract in contracts.items():
        if contract["kind"] != expected_kinds[name]:
            raise ValueError(f"{name} has the wrong checkpoint kind")
        mismatched = [key for key in matched if contract["config"].get(key) != reference["config"].get(key)]
        if mismatched:
            raise ValueError(f"{name} flow contract differs on: {', '.join(mismatched)}")
        if contract["normalization"] != reference["normalization"]:
            raise ValueError(f"{name} normalization metadata differs")
        if contract["output_spec"] != reference["output_spec"]:
            raise ValueError(f"{name} output specification differs")
        if int(contract["epoch"]) != 500:
            raise ValueError(f"{name} evaluation requires the epoch-500 endpoint")
    config, output = reference["config"], reference["output_spec"]
    expected = {
        "task": "ecg2ecg", "datasets": ["CPSC2018"],
        "dataset_version": EXPECTED_DATASET_VERSION, "split_hash": EXPECTED_SPLIT_HASH,
        "normalization_id": "record_minmax_neg1_1_v1", "alignment_id": EXPECTED_ALIGNMENT,
        "condition_lead_index": 1, "target_lead_indices": list(TARGET_INDICES), "window_size": 4,
    }
    bad = [key for key, value in expected.items() if config.get(key) != value]
    if int(config.get("seed", -1)) != expected_training_seed:
        bad.append("seed")
    if bad or int(output.get("channels", 0)) != 11 or int(output.get("length", 0)) != 512:
        raise ValueError("flow checkpoint violates the frozen CPSC2018 contract: " + ", ".join(bad))
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


def _validate_rddm_checkpoint(
    checkpoint: Mapping[str, object], expected_training_seed: int = 31
) -> None:
    if checkpoint.get("kind") != "independent_rddm_reproduction" or int(checkpoint.get("epoch", -1)) != 500:
        raise ValueError("RDDM evaluation requires the reproduced epoch-500 checkpoint")
    config = checkpoint["config"]
    expected = {
        "task": "ecg2ecg", "datasets": ["CPSC2018"],
        "dataset_version": EXPECTED_DATASET_VERSION, "split_hash": EXPECTED_SPLIT_HASH,
        "normalization_id": "record_minmax_neg1_1_v1", "alignment_id": EXPECTED_ALIGNMENT,
        "condition_lead_index": 1, "target_lead_indices": list(TARGET_INDICES),
        "target_channels": 11, "window_size": 4, "nT": 10, "attention_heads": 8,
        "seed": expected_training_seed, "reproduction_label": "RDDM-ECG (adapted)",
    }
    bad = [key for key, value in expected.items() if config.get(key) != value]
    if bad or int(checkpoint.get("global_step", -1)) != 21500:
        raise ValueError("RDDM checkpoint violates the frozen CPSC2018 contract: " + ", ".join(bad))
    if checkpoint.get("provenance", {}).get("upstream_commit") != EXPECTED_UPSTREAM_COMMIT:
        raise ValueError("RDDM upstream commit changed")


def _validate_manifest(path: Path, expected_records: int) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    # The source-derived v3 manifest predates the checkpoint-level dataset_version
    # field. Its immutable split hash, schema, and stored record count bind the data.
    if int(manifest.get("schema_version", -1)) != 2 or manifest.get("split_hash") != EXPECTED_SPLIT_HASH:
        raise ValueError("CPSC2018 dataset manifest contract changed")
    split = manifest.get("splits", {}).get("val", {})
    records = split.get("records", split.get("count"))
    if int(records if records is not None else -1) != expected_records:
        raise ValueError("CPSC2018 validation record count changed")
    return manifest


@torch.no_grad()
def _generate_rddm(checkpoint_path: Path, conditions: np.ndarray, batch_size: int,
                   sampling_seed: int, device: torch.device,
                   expected_training_seed: int = 31) -> tuple[np.ndarray, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    _validate_rddm_checkpoint(checkpoint, expected_training_seed)
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
    for start in range(0, len(conditions), batch_size):
        stop = min(start + batch_size, len(conditions))
        seed = sampling_seed + start // batch_size
        batch_seeds.append(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        source = torch.as_tensor(conditions[start:stop], device=device)
        generated = _sample_rddm(model, condition_1(source), condition_2(source), (stop - start, channels, length))
        predictions[start:stop] = generated.cpu().numpy()
    del model, condition_1, condition_2
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("RDDM predictions contain NaN or Inf")
    metadata["sampling_batch_seeds"] = batch_seeds
    return predictions, metadata


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    flow_paths = {"cfm": args.cfm_checkpoint, "rcfm": args.rcfm_checkpoint, "rcfm_ot": args.rcfm_ot_checkpoint}
    contracts = {name: _checkpoint_contract(path) for name, path in flow_paths.items()}
    _validate_flow_contracts(contracts, args.expected_training_seed)
    rddm = torch.load(args.rddm_checkpoint, map_location="cpu")
    _validate_rddm_checkpoint(rddm, args.expected_training_seed)
    rddm_steps = int(rddm["config"]["nT"])
    del rddm
    gc.collect()
    dataset_dir = args.data_root.resolve() / "CPSC2018"
    manifest = _validate_manifest(dataset_dir / "dataset_manifest.json", args.expected_records)
    config = contracts["cfm"]["config"]
    _, dataset = build_datasets(
        config["task"], config["datasets"], str(args.data_root.resolve()), int(config["window_size"]),
        normalization_metadata=contracts["cfm"]["normalization"], normalization_id=config["normalization_id"],
        condition_lead_index=1, target_lead_indices=list(TARGET_INDICES), load_train=False, heldout_split="val",
    )
    if len(dataset) != args.expected_records:
        raise ValueError(f"expected {args.expected_records} validation records, found {len(dataset)}")
    count = len(dataset) if args.max_records is None else min(args.max_records, len(dataset))
    targets = np.asarray(dataset.target_ecg[:count], dtype=np.float32)
    conditions = np.asarray(dataset.condition_signal[:count, None, :], dtype=np.float32)
    record_ids = np.asarray(dataset.record_ids[:count])
    if targets.shape != (count, 11, 512) or conditions.shape != (count, 1, 512):
        raise ValueError("CPSC2018 validation tensors have the wrong shape")
    if len(np.unique(record_ids)) != count:
        raise ValueError("CPSC2018 validation record IDs are not unique")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    _set_deterministic(args.deterministic_seed, device)
    generator = torch.Generator(device="cpu"); generator.manual_seed(args.noise_seed)
    initial_noise = torch.randn((count, 11, 512), generator=generator).numpy()
    np.savez_compressed(output / "paired_reference.npz", targets=targets, conditions=conditions,
                        record_ids=record_ids, initial_flow_noise=initial_noise)
    predictions, generation = {}, {}
    kinds = {"cfm": "canonical_multistep_cfm", "rcfm": "canonical_multistep_rcfm", "rcfm_ot": "canonical_multistep_rcfm"}
    for name in ("cfm", "rcfm", "rcfm_ot"):
        predictions[name], metadata = _generate(flow_paths[name], kinds[name], conditions, initial_noise,
                                                args.batch_size, args.steps, device, args.deterministic_seed)
        np.save(output / f"{name}_predictions.npy", predictions[name], allow_pickle=False)
        generation[name] = {**metadata, "prediction_sha256": _array_sha256(predictions[name])}
    predictions["rddm"], metadata = _generate_rddm(
        args.rddm_checkpoint, conditions, args.batch_size, args.rddm_sampling_seed, device,
        args.expected_training_seed,
    )
    np.save(output / "rddm_predictions.npy", predictions["rddm"], allow_pickle=False)
    generation["rddm"] = {**metadata, "prediction_sha256": _array_sha256(predictions["rddm"])}
    summaries, per_record = {}, {}
    for name in MODEL_ORDER:
        summaries[name], per_record[name] = _metric_summary(targets, predictions[name])
    baselines = {}
    for name, values in {"zero": np.zeros_like(targets), "lead_ii_copy": np.repeat(conditions, 11, axis=1)}.items():
        baselines[name], _ = _metric_summary(targets, values)
    _write_per_record(output / "per_record_metrics.csv", record_ids, per_record)
    _json(output / "waveform_summary.json", {"models": summaries, "baselines": baselines})
    status = "completed" if count == args.expected_records else "smoke_completed"
    protocol = {
        "schema_version": 1, "status": status, "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv),
        "protocol": {
            "dataset": "CPSC2018", "dataset_version": EXPECTED_DATASET_VERSION,
            "split": "source_derived_validation", "split_hash": EXPECTED_SPLIT_HASH,
            "available_records": args.expected_records, "evaluated_records": count,
            "sampling_rate_hz": 128, "window_seconds": 4, "condition_lead": "II",
            "target_leads": list(TARGET_LEADS), "normalization_id": "record_minmax_neg1_1_v1",
            "primary_alignment": "raw_synchronized_same_record_same_start_no_shift",
            "phase_correction_applied": False, "phase_diagnostic_applied": False,
            "flow_nfe": args.steps, "flow_noise_seed": args.noise_seed,
            "training_seed": args.expected_training_seed,
            "rddm_steps": rddm_steps, "rddm_sampling_seed": args.rddm_sampling_seed,
            "same_flow_noise_across_flow_models": True,
        },
        "selection": {model: "epoch_500_endpoint_no_validation_metric_selection" for model in MODEL_ORDER},
        "claim_boundaries": {
            "validation_only": True, "patient_disjoint": "unverified_no_patient_identifiers",
            "physical_amplitudes": "blocked_unknown_source_unit",
            "hrv": "blocked_independent_four_second_records",
        },
        "checkpoints": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path), **generation[name]}
            for name, path in {**flow_paths, "rddm": args.rddm_checkpoint}.items()
        },
        "artifacts": {
            "paired_reference_sha256": _sha256(output / "paired_reference.npz"),
            **{f"{name}_predictions_sha256": _sha256(output / f"{name}_predictions.npy") for name in MODEL_ORDER},
        },
        "manifest_source": {"path": str((dataset_dir / "dataset_manifest.json").resolve()),
                            "sha256": _sha256(dataset_dir / "dataset_manifest.json"),
                            "eligible_records": manifest.get("eligible_records")},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "torch": torch.__version__, "device": str(device),
                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
    }
    _json(output / "protocol.json", protocol)
    print(f"CPSC2018 four-way evaluation {status}: {output}", flush=True)
    return output


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
    parser.add_argument("--expected_training_seed", type=int, default=31)
    parser.add_argument("--expected_records", type=int, default=686)
    parser.add_argument("--max_records", type=int, default=None)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
