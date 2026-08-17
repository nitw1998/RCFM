"""Generate CFM+OT on an existing deterministic five-dataset reference artifact."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import (
    _checkpoint_contract,
    _generate,
    _json,
    _sha256,
)


DATASETS = {
    "mimic_afib": {
        "name": "MIMIC-AFib", "task": "ppg2ecg", "records": 1800, "channels": 1,
        "dataset_version": "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1",
        "split_hash": "a7e388293adaa7b48d3493efc505dd8750520730cae9fd7649157866efa86a51",
        "normalization_id": "rddm_window_minmax_neg1_1_v1",
        "alignment_id": "paired_array_row_rddm_contract_zero_ppg_qc_v1",
        "noise_key": "initial_noise",
    },
    "cpsc2018": {
        "name": "CPSC2018", "task": "ecg2ecg", "records": 686, "channels": 11,
        "dataset_version": "cpsc2018-source-derived-all12lead-qc-v3-record-minmax-neg1-1",
        "split_hash": "35e0a796a60e4d6979b1f050048495fdf1826eedda7d40e47a56d4dcd5874223",
        "normalization_id": "record_minmax_neg1_1_v1",
        "alignment_id": "same_record_simultaneous_channels_first_4s_lead_II_to_other_11_v3",
        "noise_key": "initial_flow_noise",
    },
    "wesad": {
        "name": "WESAD", "task": "ppg2ecg", "records": 4213, "channels": 1,
        "dataset_version": "wesad-subject-fold1-linear-resample-window-minmax-v1",
        "split_hash": "0b90bffe7c3032243c803e534de3592618284d26792c2b612ce7af17a81a85cd",
        "normalization_id": "window_minmax_neg1_1_v1",
        "alignment_id": "native_common_start_same_window_no_delay_correction_subject_fold1_v1",
        "noise_key": "initial_flow_noise",
    },
    "mmecg": {
        "name": "mmECG", "task": "rcg2ecg", "records": 2877, "channels": 1,
        "dataset_version": "mmecg-public-20221108-subject-split-window-minmax-v1",
        "split_hash": "e26fc81121cfd3b0e457608e37a7aa496ac7c48a0e0e7583a465bd069cc9da9f",
        "normalization_id": "window_minmax_neg1_1_v1",
        "alignment_id": "same_record_same_window_no_additional_phase_correction_subject_split_v1",
        "noise_key": "initial_flow_noise",
    },
}


def _validate_checkpoint(contract: dict[str, object], dataset_key: str) -> None:
    expected = DATASETS[dataset_key]
    if contract.get("kind") != "canonical_multistep_cfm_ot":
        raise ValueError("checkpoint must be canonical_multistep_cfm_ot")
    config = contract["config"]
    required = {
        "task": expected["task"], "datasets": [expected["name"]],
        "dataset_version": expected["dataset_version"], "split_hash": expected["split_hash"],
        "normalization_id": expected["normalization_id"], "alignment_id": expected["alignment_id"],
        "window_size": 4, "attention_heads": 8, "flow_matcher": "conditional",
        "sigma": 0.0, "seed": 31, "region_weight": 0.0,
        "use_minibatch_ot": True, "ot_method": "exact",
    }
    bad = [key for key, value in required.items() if config.get(key) != value]
    output = contract["output_spec"]
    if output.get("channels") != expected["channels"] or output.get("length") != 512:
        bad.append("output_shape")
    if contract["normalization"].get("normalization_id") != expected["normalization_id"]:
        bad.append("normalization")
    if bad:
        raise ValueError("CFM+OT checkpoint violates frozen contract: " + ", ".join(bad))


def _read_reference(path: Path, dataset_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = DATASETS[dataset_key]
    with np.load(path, allow_pickle=False) as artifact:
        required = {"targets", "conditions", str(expected["noise_key"])}
        if missing := sorted(required - set(artifact.files)):
            raise ValueError("frozen reference is missing: " + ", ".join(missing))
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        conditions = np.asarray(artifact["conditions"], dtype=np.float32)
        noise = np.asarray(artifact[str(expected["noise_key"])], dtype=np.float32)
    output_shape = (int(expected["records"]), int(expected["channels"]), 512)
    if targets.shape != output_shape or noise.shape != output_shape:
        raise ValueError("frozen targets/noise have the wrong shape")
    if conditions.shape != (int(expected["records"]), 1, 512):
        raise ValueError("frozen conditions have the wrong shape")
    if any(not np.all(np.isfinite(value)) for value in (targets, conditions, noise)):
        raise FloatingPointError("frozen reference contains NaN or Inf")
    return targets, conditions, noise


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.steps != 50 or args.deterministic_seed != 31:
        raise ValueError("frozen protocol requires NFE=50 and deterministic seed 31")
    reference = args.reference_artifact.resolve()
    _, conditions, noise = _read_reference(reference, args.dataset)
    contract = _checkpoint_contract(args.checkpoint.resolve())
    _validate_checkpoint(contract, args.dataset)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    predictions, generation = _generate(
        args.checkpoint.resolve(), "canonical_multistep_cfm_ot", conditions, noise,
        args.batch_size, args.steps, device, args.deterministic_seed,
    )
    prediction_path = output / "cfm_ot_predictions.npy"
    np.save(prediction_path, predictions, allow_pickle=False)
    protocol = {
        "schema_version": 1, "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv),
        "dataset": args.dataset,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": _sha256(args.checkpoint), **generation},
        "reference": {"path": str(reference), "sha256": _sha256(reference)},
        "inference": {
            "records": DATASETS[args.dataset]["records"], "nfe": args.steps,
            "shared_noise_from_reference": True, "deterministic_seed": args.deterministic_seed,
            "ot_used_at_inference": False, "phase_correction_applied": False,
        },
        "prediction": {"path": str(prediction_path), "sha256": _sha256(prediction_path)},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                     "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
    }
    _json(output / "protocol.json", protocol)
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference_artifact", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    return parser


if __name__ == "__main__":
    result = run(build_argparser().parse_args())
    print(f"CFM+OT frozen-reference prediction saved to {result}")
