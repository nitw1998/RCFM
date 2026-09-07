"""Evaluate one PTB-XL or CPSC2018 RDDM training seed on a frozen reference."""

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

from scripts.evaluate_cpsc_fourway import (
    _generate_rddm as _generate_cpsc_rddm,
    _validate_rddm_checkpoint as _validate_cpsc_rddm,
)
from scripts.evaluate_cpsc_zscore_paired import _json, _sha256
from scripts.evaluate_ptbxl_fourway import (
    _generate_rddm as _generate_ptbxl_rddm,
    _metric_summary,
    _set_deterministic,
    _validate_rddm_checkpoint as _validate_ptbxl_rddm,
)


DATASETS = {
    "ptbxl": {"protocol_name": "PTB-XL", "records": 2203, "split": "official_fold_10"},
    "cpsc2018": {
        "protocol_name": "CPSC2018",
        "records": 686,
        "split": "source_derived_validation",
    },
}


def _load_reference(
    path: Path, source_protocol_path: Path, dataset: str
) -> tuple[np.ndarray, np.ndarray, str, dict[str, object]]:
    protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    contract = DATASETS[dataset]
    details = protocol.get("protocol", {})
    if (
        protocol.get("status") != "completed"
        or details.get("dataset") != contract["protocol_name"]
        or details.get("split") != contract["split"]
        or details.get("phase_correction_applied") is not False
    ):
        raise ValueError("source protocol is not the frozen raw evaluation contract")
    reference_hash = _sha256(path)
    declared_hash = protocol.get("artifacts", {}).get("paired_reference_sha256")
    if declared_hash != reference_hash:
        raise ValueError("paired reference hash differs from its source protocol")
    with np.load(path, allow_pickle=False) as arrays:
        required = {"targets", "conditions", "record_ids"}
        if not required.issubset(arrays.files):
            raise ValueError("paired reference is missing required arrays")
        targets = np.asarray(arrays["targets"], dtype=np.float32)
        conditions = np.asarray(arrays["conditions"], dtype=np.float32)
        record_ids = np.asarray(arrays["record_ids"])
    expected = (int(contract["records"]), 11, 512)
    if targets.shape != expected or conditions.shape != (expected[0], 1, 512):
        raise ValueError("paired reference tensor shape changed")
    if record_ids.shape != (expected[0],) or not np.all(np.isfinite(targets)):
        raise ValueError("paired reference identifiers or values are invalid")
    return targets, conditions, reference_hash, protocol


def run(args: argparse.Namespace) -> Path:
    dataset = args.dataset
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    validator = _validate_ptbxl_rddm if dataset == "ptbxl" else _validate_cpsc_rddm
    validator(checkpoint, args.expected_training_seed)
    checkpoint_seed = int(checkpoint["config"]["seed"])
    del checkpoint
    targets, conditions, reference_hash, source_protocol = _load_reference(
        args.reference_artifact.resolve(), args.source_protocol.resolve(), dataset
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    _set_deterministic(args.deterministic_seed, device)
    generator = _generate_ptbxl_rddm if dataset == "ptbxl" else _generate_cpsc_rddm
    predictions, generation = generator(
        args.checkpoint,
        conditions,
        args.batch_size,
        args.sampling_seed,
        device,
        args.expected_training_seed,
    )
    summary, _ = _metric_summary(targets, predictions)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.npy"
    np.save(prediction_path, predictions, allow_pickle=False)
    _json(output / "waveform_summary.json", {"model": "rddm", "metrics": summary})
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "model_name": "rddm",
        "training_seed": checkpoint_seed,
        "selection": "predeclared_epoch_500_endpoint_without_validation_selection",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "dataset": dataset,
        "paired_reference": {
            "path": str(args.reference_artifact.resolve()),
            "sha256": reference_hash,
            "source_protocol": str(args.source_protocol.resolve()),
            "source_protocol_sha256": _sha256(args.source_protocol.resolve()),
        },
        "inference": {
            "records": int(len(targets)),
            "split": DATASETS[dataset]["split"],
            "rddm_steps": 10,
            "sampling_seed": args.sampling_seed,
            "deterministic_seed": args.deterministic_seed,
            "phase_correction_applied": False,
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": _sha256(args.checkpoint.resolve()),
            **generation,
        },
        "prediction": {
            "path": str(prediction_path.resolve()),
            "sha256": _sha256(prediction_path),
            "shape": list(predictions.shape),
        },
        "source_evaluation_status": source_protocol["status"],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
    }
    _json(output / "protocol.json", protocol)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference_artifact", type=Path, required=True)
    parser.add_argument("--source_protocol", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_training_seed", type=int, choices=(31, 32, 33), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sampling_seed", type=int, default=2025)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
