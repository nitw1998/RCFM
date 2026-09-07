"""Evaluate one WESAD training seed against the frozen seed-31 reference rows."""

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
    _sha256,
    _waveform_metrics,
)
from scripts.evaluate_mimic_phase_corrected import _fixed_support_align
from scripts.evaluate_mmecg_fourway import _write_csv
from scripts.evaluate_ptbxl_fourway import _sample_rddm, _set_deterministic
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic
from train_rcfm import build_datasets


MODELS = ("cfm", "rcfm", "rcfm_ot", "rddm")
EXPECTED_DATASET_VERSION = "wesad-subject-fold1-linear-resample-window-minmax-v1"
EXPECTED_SPLIT_HASH = "0b90bffe7c3032243c803e534de3592618284d26792c2b612ce7af17a81a85cd"
EXPECTED_ALIGNMENT = "native_common_start_same_window_no_delay_correction_subject_fold1_v1"


def _validate_flow_contracts(
    contracts: Mapping[str, Mapping[str, object]], expected_training_seed: int
) -> None:
    kinds = {
        "cfm": "canonical_multistep_cfm",
        "rcfm": "canonical_multistep_rcfm",
        "rcfm_ot": "canonical_multistep_rcfm",
    }
    if set(contracts) != set(kinds):
        raise ValueError("flow contracts must contain cfm, rcfm, and rcfm_ot")
    reference = contracts["cfm"]
    matched = (
        "task", "datasets", "dataset_version", "split_hash", "normalization_id",
        "alignment_id", "window_size", "attention_heads", "flow_matcher", "sigma", "seed",
    )
    for model, contract in contracts.items():
        if contract.get("kind") != kinds[model]:
            raise ValueError(f"{model} has the wrong checkpoint kind")
        if int(contract.get("epoch", -1)) != 500 or int(contract.get("global_step", -1)) != 68500:
            raise ValueError(f"{model} is not the frozen WESAD epoch-500 endpoint")
        config = contract["config"]
        bad = [key for key in matched if config.get(key) != reference["config"].get(key)]
        if bad or contract["normalization"] != reference["normalization"]:
            raise ValueError(f"{model} differs on frozen fields: {', '.join(bad)}")
        if contract["output_spec"] != reference["output_spec"]:
            raise ValueError(f"{model} output specification differs")
    expected = {
        "task": "ppg2ecg", "datasets": ["WESAD"],
        "dataset_version": EXPECTED_DATASET_VERSION, "split_hash": EXPECTED_SPLIT_HASH,
        "normalization_id": "window_minmax_neg1_1_v1", "alignment_id": EXPECTED_ALIGNMENT,
        "window_size": 4, "attention_heads": 8, "flow_matcher": "conditional",
        "sigma": 0.0, "seed": expected_training_seed,
    }
    config = reference["config"]
    bad = [key for key, value in expected.items() if config.get(key) != value]
    if bad or reference["output_spec"].get("channels") != 1 or reference["output_spec"].get("length") != 512:
        raise ValueError("flow checkpoints violate the frozen WESAD contract: " + ", ".join(bad))
    cfm, rcfm, rcfm_ot = (contracts[name]["config"] for name in ("cfm", "rcfm", "rcfm_ot"))
    if float(cfm.get("region_weight", -1)) != 0 or bool(cfm.get("use_minibatch_ot")):
        raise ValueError("CFM must have region_weight=0 and OT disabled")
    if float(rcfm.get("region_weight", 0)) <= 0 or bool(rcfm.get("use_minibatch_ot")):
        raise ValueError("RCFM must have positive region weight and OT disabled")
    if float(rcfm_ot.get("region_weight", 0)) != float(rcfm.get("region_weight")):
        raise ValueError("RCFM and RCFM-OT region weights must match")
    if not bool(rcfm_ot.get("use_minibatch_ot")) or rcfm_ot.get("ot_method") != "exact":
        raise ValueError("RCFM-OT must use exact minibatch OT")


def _validate_rddm_checkpoint(checkpoint: Mapping[str, object], expected_training_seed: int) -> None:
    config = checkpoint.get("config", {})
    expected = {
        "task": "ppg2ecg", "datasets": ["WESAD"],
        "dataset_version": EXPECTED_DATASET_VERSION, "split_hash": EXPECTED_SPLIT_HASH,
        "normalization_id": "window_minmax_neg1_1_v1", "alignment_id": EXPECTED_ALIGNMENT,
        "heldout_split": "test", "expected_train_windows": 17494,
        "expected_test_windows": 4213, "window_size": 4, "target_channels": 1,
        "nT": 10, "seed": expected_training_seed,
        "reproduction_label": "RDDM-PPG (matched-protocol reproduction)",
    }
    bad = [key for key, value in expected.items() if config.get(key) != value]
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("kind") != "independent_rddm_reproduction"
        or int(checkpoint.get("epoch", -1)) != 500
        or int(checkpoint.get("global_step", -1)) != 68500
        or checkpoint.get("provenance", {}).get("upstream_commit")
        != "7d5348843c3985c211a23ae5105a2d9497d5156a"
        or bad
    ):
        raise ValueError("RDDM checkpoint violates the frozen WESAD contract: " + ", ".join(bad))


@torch.no_grad()
def _generate_rddm(
    checkpoint_path: Path,
    conditions: np.ndarray,
    batch_size: int,
    sampling_seed: int,
    device: torch.device,
    expected_training_seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    _validate_rddm_checkpoint(checkpoint, expected_training_seed)
    config = checkpoint["config"]
    model = RDDM(
        eps_model=DiffusionUNetCrossAttention(512, 1, str(device), num_heads=int(config["attention_heads"])),
        region_model=DiffusionUNetCrossAttention(512, 1, str(device), num_heads=int(config["attention_heads"])),
        betas=(float(config["beta_start"]), float(config["beta_end"])),
        n_T=int(config["nT"]),
    ).to(device)
    condition_1, condition_2 = ConditionNet().to(device), ConditionNet().to(device)
    model.load_state_dict(checkpoint["rddm_state"], strict=True)
    condition_1.load_state_dict(checkpoint["condition_1_state"], strict=True)
    condition_2.load_state_dict(checkpoint["condition_2_state"], strict=True)
    model.eval(); condition_1.eval(); condition_2.eval()
    metadata = {"kind": checkpoint["kind"], "epoch": int(checkpoint["epoch"]), "global_step": int(checkpoint["global_step"])}
    del checkpoint
    predictions = np.empty((len(conditions), 1, 512), dtype=np.float32)
    seeds = []
    for batch_index, start in enumerate(range(0, len(conditions), batch_size)):
        stop = min(start + batch_size, len(conditions)); seed = sampling_seed + batch_index
        seeds.append(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        source = torch.as_tensor(conditions[start:stop], device=device)
        generated = _sample_rddm(model, condition_1(source), condition_2(source), (stop - start, 1, 512))
        predictions[start:stop] = generated.cpu().numpy()
    metadata["sampling_batch_seeds"] = seeds
    del model, condition_1, condition_2
    torch.cuda.empty_cache(); gc.collect()
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("RDDM predictions contain NaN or Inf")
    return predictions, metadata


def _load_reference(path: Path) -> tuple[np.ndarray, ...]:
    with np.load(path, allow_pickle=False) as artifact:
        required = {"targets", "conditions", "subject_ids", "labels", "initial_flow_noise"}
        if missing := sorted(required - set(artifact.files)):
            raise ValueError("frozen WESAD reference is missing: " + ", ".join(missing))
        arrays = tuple(np.asarray(artifact[key]) for key in ("targets", "conditions", "subject_ids", "labels", "initial_flow_noise"))
    targets, conditions, subjects, labels, noise = arrays
    if any(value.shape != (4213, 1, 512) for value in (targets, conditions, noise)):
        raise ValueError("frozen WESAD waveform arrays have the wrong shape")
    if subjects.shape != (4213,) or labels.shape != (4213,):
        raise ValueError("frozen WESAD identity arrays have the wrong shape")
    return targets.astype(np.float32), conditions.astype(np.float32), subjects.astype(str), labels.astype(np.int16), noise.astype(np.float32)


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.max_lag_samples != 16 or args.sampling_rate != 128 or args.inference_steps != 50:
        raise ValueError("response protocol requires NFE=50 and +/-16 samples at 128 Hz")
    paths = {name: getattr(args, f"{name}_checkpoint").resolve() for name in MODELS}
    contracts = {name: _checkpoint_contract(paths[name]) for name in ("cfm", "rcfm", "rcfm_ot")}
    _validate_flow_contracts(contracts, args.expected_training_seed)
    rddm = torch.load(paths["rddm"], map_location="cpu")
    _validate_rddm_checkpoint(rddm, args.expected_training_seed); rddm_steps = int(rddm["config"]["nT"]); del rddm
    targets, conditions, subjects, labels, noise = _load_reference(args.reference_artifact.resolve())

    _, heldout = build_datasets(
        task="ppg2ecg", datasets=["WESAD"], data_root=str(args.data_root.resolve()),
        window_size=4, normalization_metadata=contracts["cfm"]["normalization"],
        normalization_id="window_minmax_neg1_1_v1", load_train=False, heldout_split="test",
    )
    loaded = (
        np.asarray(heldout.target_ecg[:, None, :], dtype=np.float32),
        np.asarray(heldout.condition_signal[:, None, :], dtype=np.float32),
        np.load(args.data_root / "WESAD/subject_ids_test.npy", allow_pickle=False).astype(str),
        np.load(args.data_root / "WESAD/labels_test.npy", allow_pickle=False).astype(np.int16),
    )
    if not all(np.array_equal(a, b) for a, b in zip((targets, conditions, subjects, labels), loaded)):
        raise ValueError("frozen WESAD reference rows do not match the held-out dataset")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal WESAD multiseed evaluation requires CUDA")
    _set_deterministic(args.deterministic_seed, device)
    predictions: dict[str, np.ndarray] = {}
    generation: dict[str, object] = {}
    kinds = {"cfm": "canonical_multistep_cfm", "rcfm": "canonical_multistep_rcfm", "rcfm_ot": "canonical_multistep_rcfm"}
    for model in ("cfm", "rcfm", "rcfm_ot"):
        predictions[model], generation[model] = _generate(
            paths[model], kinds[model], conditions, noise, args.batch_size,
            args.inference_steps, device, args.deterministic_seed,
        )
        torch.cuda.empty_cache(); gc.collect()
    predictions["rddm"], generation["rddm"] = _generate_rddm(
        paths["rddm"], conditions, args.batch_size, args.rddm_sampling_seed,
        device, args.expected_training_seed,
    )

    raw_summary, phase_summary, phase_arrays = {}, {}, {
        "targets": targets[:, :, 16:-16], "subject_ids": subjects, "labels": labels,
    }
    for model in MODELS:
        raw_summary[model], _ = _waveform_metrics(targets, predictions[model])
        lag_summary, lag_values = _lag_diagnostic(targets, predictions[model], 16, 128)
        shifts = lag_values["best_lag_samples"].astype(np.int32)
        center, before, after = _fixed_support_align(targets, predictions[model], shifts, 16)
        before_summary, _ = _waveform_metrics(center, before)
        after_summary, _ = _waveform_metrics(center, after)
        phase_summary[model] = {
            "lag": lag_summary, "unshifted_fixed_support": before_summary,
            "oracle_aligned_fixed_support": after_summary,
            "median_absolute_shift_samples": float(np.median(np.abs(shifts))),
            "boundary_hit_fraction": float(np.mean(np.abs(shifts) == 16)),
        }
        phase_arrays[f"{model}_oracle_shifts"] = shifts
        phase_arrays[f"{model}_unshifted_predictions"] = before
        phase_arrays[f"{model}_oracle_aligned_predictions"] = after

    raw_path, phase_path = output / "raw_predictions.npz", output / "phase_predictions_maxlag16.npz"
    np.savez_compressed(
        raw_path, targets=targets, conditions=conditions, subject_ids=subjects, labels=labels,
        initial_flow_noise=noise, **{f"{model}_predictions": predictions[model] for model in MODELS},
    )
    np.savez_compressed(phase_path, **phase_arrays)
    summary_path = output / "waveform_summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": 1, "training_seed": args.expected_training_seed,
        "generation": generation, "raw_full_window": raw_summary, "phase_sensitivity": phase_summary,
    }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    outputs = (raw_path, phase_path, summary_path)
    protocol = {
        "schema_version": 1, "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv), "training_seed": args.expected_training_seed,
        "inputs": {"reference": {"path": str(args.reference_artifact.resolve()), "sha256": _sha256(args.reference_artifact)},
                   "checkpoints": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()}},
        "array_sha256": {"targets": _array_sha256(targets), "conditions": _array_sha256(conditions),
                         "subject_ids": _array_sha256(subjects), "labels": _array_sha256(labels),
                         "initial_flow_noise": _array_sha256(noise)},
        "protocol": {"dataset": "WESAD", "windows": 4213, "heldout_subjects": sorted(set(subjects)),
                     "flow_nfe": 50, "rddm_steps": rddm_steps, "flow_noise_seed": 2025,
                     "rddm_sampling_seed": args.rddm_sampling_seed, "phase_correction_target_informed": True,
                     "max_lag_samples": 16, "fixed_support_samples": 480},
        "execution": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                      "torch": torch.__version__, "device": str(device)},
        "outputs": [path.name for path in outputs],
        "artifact_sha256": {path.name: _sha256(path) for path in outputs},
        "claim_boundary": "Three held-out subjects; target-informed phase correction is diagnostic and not deployable.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    for model in MODELS:
        parser.add_argument(f"--{model}_checkpoint", type=Path, required=True)
    parser.add_argument("--reference_artifact", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_training_seed", type=int, choices=(32, 33), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    parser.add_argument("--rddm_sampling_seed", type=int, default=2025)
    parser.add_argument("--sampling_rate", type=int, default=128)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
