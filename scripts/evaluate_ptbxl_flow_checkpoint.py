"""Generate one PTB-XL fold-10 flow prediction on a frozen paired reference."""

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
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import (
    _array_sha256,
    _checkpoint_contract,
    _generate,
    _json,
    _sha256,
)
from scripts.evaluate_ptbxl_fourway import (
    EXPECTED_DATASET_VERSION,
    EXPECTED_SPLIT_HASH,
    TARGET_INDICES,
    TARGET_LEADS,
)


MODEL_CONTRACTS = {
    "cfm": {
        "kind": "canonical_multistep_cfm",
        "region_weight": 0.0,
        "use_minibatch_ot": False,
        "mask_method": None,
    },
    "cfm_ot": {
        "kind": "canonical_multistep_cfm_ot",
        "region_weight": 0.0,
        "use_minibatch_ot": True,
        "mask_method": None,
    },
    "diag": {
        "kind": "canonical_multistep_rcfm",
        "region_weight": 0.01,
        "use_minibatch_ot": False,
        "mask_method": "xresnet1d101_all_positive_statements_gradcam_l5_sample_center_v1",
    },
    "diag_ot": {
        "kind": "canonical_multistep_rcfm",
        "region_weight": 0.01,
        "use_minibatch_ot": True,
        "mask_method": "xresnet1d101_all_positive_statements_gradcam_l5_sample_center_v1",
    },
    "semantic": {
        "kind": "canonical_multistep_rcfm",
        "region_weight": 0.01,
        "use_minibatch_ot": False,
        "mask_method": "ptbxl_plus_resunet_p_qrs_t_lead_ii_soft_max_epoch13_v1",
    },
    "ecgmamba_diag": {
        "kind": "canonical_multistep_rcfm",
        "region_weight": 0.01,
        "use_minibatch_ot": False,
        "mask_method": "ecgmamba_fca_mgda_s42_positive_diagnostic_head_gradcam_fullres_v1",
    },
    "ecgmamba_semantic": {
        "kind": "canonical_multistep_rcfm",
        "region_weight": 0.01,
        "use_minibatch_ot": False,
        "mask_method": "ecgmamba_fca_mgda_s42_p_qrs_t_semantic_head_gradcam_fullres_v1",
    },
    "diag_l003": {
        "kind": "canonical_multistep_rcfm",
        "region_weight": 0.03,
        "use_minibatch_ot": False,
        "mask_method": "xresnet1d101_all_positive_statements_gradcam_l5_sample_center_v1",
    },
    "diag_l010": {
        "kind": "canonical_multistep_rcfm",
        "region_weight": 0.1,
        "use_minibatch_ot": False,
        "mask_method": "xresnet1d101_all_positive_statements_gradcam_l5_sample_center_v1",
    },
    "diag_l030": {
        "kind": "canonical_multistep_rcfm",
        "region_weight": 0.3,
        "use_minibatch_ot": False,
        "mask_method": "xresnet1d101_all_positive_statements_gradcam_l5_sample_center_v1",
    },
}


def _validate_reference(
    reference_path: Path,
    source_protocol: Mapping[str, object],
    expected_records: int,
) -> dict[str, np.ndarray]:
    protocol = source_protocol.get("protocol", {})
    expected = {
        "dataset": "PTB-XL",
        "dataset_version": EXPECTED_DATASET_VERSION,
        "split": "official_fold_10",
        "split_hash": EXPECTED_SPLIT_HASH,
        "evaluated_records": expected_records,
        "normalization_id": "record_minmax_neg1_1_v1",
        "phase_correction_applied": False,
        "flow_nfe": 50,
        "flow_noise_seed": 2025,
        "same_flow_noise_across_flow_models": True,
    }
    bad = [key for key, value in expected.items() if protocol.get(key) != value]
    if source_protocol.get("status") != "completed" or bad:
        raise ValueError("source reference violates the frozen fold-10 protocol: " + ", ".join(bad))
    expected_hash = source_protocol.get("artifacts", {}).get("paired_reference_sha256")
    if not expected_hash or _sha256(reference_path) != expected_hash:
        raise ValueError("paired reference hash disagrees with its source protocol")
    with np.load(reference_path, allow_pickle=False) as artifact:
        required = {"targets", "conditions", "record_ids", "patient_ids", "initial_flow_noise"}
        if set(artifact.files) != required:
            raise ValueError("paired reference keys changed")
        arrays = {name: np.asarray(artifact[name]) for name in required}
    shape = (expected_records, 11, 512)
    if arrays["targets"].shape != shape or arrays["initial_flow_noise"].shape != shape:
        raise ValueError("paired target/noise shape changed")
    if arrays["conditions"].shape != (expected_records, 1, 512):
        raise ValueError("paired condition shape changed")
    if len(arrays["record_ids"]) != expected_records or len(np.unique(arrays["record_ids"])) != expected_records:
        raise ValueError("paired record IDs are incomplete or non-unique")
    return arrays


def _validate_checkpoint(
    contract: Mapping[str, object], model_name: str, expected_training_seed: int = 31
) -> None:
    expected_model = MODEL_CONTRACTS[model_name]
    if contract.get("kind") != expected_model["kind"]:
        raise ValueError(f"{model_name} checkpoint kind is not {expected_model['kind']}")
    config = contract["config"]
    expected_config = {
        "task": "ecg2ecg",
        "datasets": ["PTBXL"],
        "dataset_version": EXPECTED_DATASET_VERSION,
        "split_hash": EXPECTED_SPLIT_HASH,
        "normalization_id": "record_minmax_neg1_1_v1",
        "alignment_id": "ptbxl_official_folds_first4s_same_record_lead_II_to_other11_v1",
        "condition_lead_index": 1,
        "target_lead_indices": list(TARGET_INDICES),
        "window_size": 4,
        "flow_matcher": "conditional",
        "sigma": 0.0,
        "seed": expected_training_seed,
        "region_weight": expected_model["region_weight"],
        "use_minibatch_ot": expected_model["use_minibatch_ot"],
    }
    bad = [key for key, value in expected_config.items() if config.get(key) != value]
    if expected_model["use_minibatch_ot"] and config.get("ot_method") != "exact":
        bad.append("ot_method")
    expected_mask = expected_model["mask_method"]
    if expected_mask is None:
        if config.get("region_mask_path") or config.get("region_mask_manifest"):
            bad.append("unexpected_region_mask")
    else:
        provenance = config.get("region_mask_provenance", {})
        if config.get("mask_method") != expected_mask or provenance.get("method") != expected_mask:
            bad.append("mask_method")
        if provenance.get("test_mask_generated") is not False:
            bad.append("test_mask_generated")
    output = contract["output_spec"]
    if output.get("channels") != 11 or output.get("length") != 512:
        bad.append("output_shape")
    if tuple(output.get("target_leads", ())) != TARGET_LEADS:
        bad.append("target_leads")
    normalization = contract["normalization"]
    if normalization.get("normalization_id") != "record_minmax_neg1_1_v1":
        bad.append("normalization")
    if bad:
        raise ValueError(f"{model_name} checkpoint violates frozen contract: " + ", ".join(bad))


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_protocol_path = args.source_protocol.resolve()
    reference_path = args.reference_artifact.resolve()
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    arrays = _validate_reference(reference_path, source_protocol, args.expected_records)
    contract = _checkpoint_contract(args.checkpoint.resolve())
    _validate_checkpoint(contract, args.model_name, args.expected_training_seed)
    if args.steps != 50:
        raise ValueError("the frozen comparison requires exactly 50 flow steps")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    predictions, generation = _generate(
        args.checkpoint.resolve(),
        str(MODEL_CONTRACTS[args.model_name]["kind"]),
        arrays["conditions"].astype(np.float32, copy=False),
        arrays["initial_flow_noise"].astype(np.float32, copy=False),
        args.batch_size,
        args.steps,
        device,
        args.deterministic_seed,
    )
    prediction_path = output_dir / "predictions.npy"
    np.save(prediction_path, predictions, allow_pickle=False)
    del predictions
    gc.collect()
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "model_name": args.model_name,
        "training_seed": args.expected_training_seed,
        "selection": "fold_9_best_rmse_checkpoint",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": _sha256(args.checkpoint.resolve()),
            **generation,
        },
        "paired_reference": {
            "path": str(reference_path),
            "sha256": _sha256(reference_path),
            "source_protocol_path": str(source_protocol_path),
            "source_protocol_sha256": _sha256(source_protocol_path),
        },
        "inference": {
            "records": args.expected_records,
            "split": "official_fold_10",
            "flow_nfe": args.steps,
            "shared_initial_noise_seed": 2025,
            "deterministic_seed": args.deterministic_seed,
            "mask_used_at_inference": False,
            "ot_used_at_inference": False,
            "phase_correction_applied": False,
        },
        "prediction": {"path": str(prediction_path), "sha256": _sha256(prediction_path)},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    _json(output_dir / "protocol.json", protocol)
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model_name", choices=tuple(MODEL_CONTRACTS), required=True)
    parser.add_argument("--reference_artifact", type=Path, required=True)
    parser.add_argument("--source_protocol", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    parser.add_argument("--expected_training_seed", type=int, default=31)
    parser.add_argument("--expected_records", type=int, default=2203)
    return parser


if __name__ == "__main__":
    output = run(build_argparser().parse_args())
    print(f"PTB-XL {build_argparser().prog} prediction saved to {output}")
