"""Build paired ECGMamba diagnostic- and semantic-head Grad-CAM training masks."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.interpretability.ecgmambaformer_compat import (
    diagnostic_class_names,
    ecgmamba_task_head_gradcam,
    load_ecgmambaformer_fca_mgda,
    record_global_zscore,
)
from src.rcfm.interpretability.gradcam import normalize_soft_mask, resample_mask_to_sample_grid


DIAG_METHOD = "ecgmamba_fca_mgda_s42_positive_diagnostic_head_gradcam_fullres_v1"
SEMANTIC_METHOD = "ecgmamba_fca_mgda_s42_p_qrs_t_semantic_head_gradcam_fullres_v1"
DATASET_VERSION = "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1"
SPLIT_HASH = "7784c98a2c8daccc23fc7cb0d47dc1933eeee77f120c05f8a24ac149cd8474f7"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _positive_targets(
    metadata: pd.DataFrame,
    record_ids: np.ndarray,
    classes: list[str],
) -> list[tuple[int, ...]]:
    class_to_index = {name: index for index, name in enumerate(classes)}
    output: list[tuple[int, ...]] = []
    for record_id in record_ids:
        codes = ast.literal_eval(str(metadata.loc[int(record_id), "scp_codes"]))
        if not isinstance(codes, dict):
            raise ValueError("PTB-XL scp_codes must decode to a dictionary")
        output.append(
            tuple(sorted(class_to_index[code] for code in codes if code in class_to_index))
        )
    return output


def _project(values: np.ndarray) -> tuple[np.ndarray, bool]:
    cam = np.asarray(values, dtype=np.float32)
    if cam.shape != (5000,) or not np.all(np.isfinite(cam)):
        raise ValueError("ECGMamba CAM must have shape (5000,) and be finite")
    projected = resample_mask_to_sample_grid(cam[:2000], 500, 128, output_length=512)
    return normalize_soft_mask(projected)


def _manifest(
    *,
    method: str,
    mask_path: Path,
    source_path: Path,
    record_ids_path: Path,
    records: int,
    dependencies: dict[str, str],
    checkpoint_path: Path,
    classes: list[str],
    degenerate_records: int,
    missing_target_records: int,
    claim_boundary: str,
) -> dict[str, object]:
    masks = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    flat = np.asarray(masks).reshape(-1)
    return {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "PTBXL",
            "dataset_version": DATASET_VERSION,
            "split_hash": SPLIT_HASH,
            "train_records": records,
            "folds": list(range(1, 9)),
            "sampling_rate_hz": 128,
            "window": "first 512 samples (4 seconds)",
            "source_ecg": {"file_name": source_path.name, "sha256": _sha256(source_path)},
            "record_ids_sha256": _sha256(record_ids_path),
        },
        "mask": {
            "method": method,
            "file_name": mask_path.name,
            "shape": list(masks.shape),
            "dtype": str(masks.dtype),
            "sha256": _sha256(mask_path),
            "soft_occupancy_mean": float(flat.mean()),
            "quantiles": [float(value) for value in np.quantile(flat, [0, 0.25, 0.5, 0.75, 1])],
            "degenerate_records": degenerate_records,
            "missing_task_target_records": missing_target_records,
            "broadcast_policy": "one 12-lead temporal CAM broadcast to all 11 target leads",
        },
        "ecgmamba_protocol": {
            "architecture": "local unpublished ECGMambaImproved + FCA-MGDA multitask checkpoint",
            "checkpoint_sha256": _sha256(checkpoint_path),
            "strict_checkpoint_load": True,
            "diagnostic_classes": classes,
            "native_input": "10-second 12-lead PTB-XL record resampled from 128 to 500 Hz",
            "normalization": "per-record global z-score across all time samples and leads",
            "target_layer": "shared full-resolution encoder output (1024 x 5000)",
            "projection": "first 2000 model samples projected to 512 samples by sample-center interpolation",
            "channel_weight": "temporal mean activation gradient",
            "positive_evidence": "ReLU",
            "per_record_normalization": "min-max [0,1]",
            "fold10_used_for_checkpoint_or_mask_generation": False,
        },
        "dependencies": dependencies,
        "test_mask_generated": False,
        "mask_usage": "training_loss_weight_only",
        "claim_boundary": claim_boundary,
    }


def run(args: argparse.Namespace) -> Path:
    if args.expected_records <= 0 or args.progress_every <= 0:
        raise ValueError("record and progress counts must be positive")
    dataset_root = args.data_root.resolve() / "PTBXL"
    source_path = dataset_root / "X_train_resampled.npy"
    record_ids_path = dataset_root / "record_ids_train.npy"
    checkpoint_path = args.checkpoint.resolve()
    metadata_path = args.metadata.resolve()
    scp_path = args.scp_statements.resolve()
    source_root = args.ecgmambaformer_root.resolve()
    required = (source_path, record_ids_path, checkpoint_path, metadata_path, scp_path)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("a required ECGMamba mask input is missing")
    waveforms = np.load(source_path, mmap_mode="r", allow_pickle=False)
    record_ids = np.load(record_ids_path, allow_pickle=False).astype(np.int64)
    if waveforms.shape != (args.expected_records, 1280, 12):
        raise ValueError("PTB-XL training waveform shape violates the frozen contract")
    if record_ids.shape != (args.expected_records,) or len(np.unique(record_ids)) != len(record_ids):
        raise ValueError("PTB-XL training record IDs are invalid")
    records = len(record_ids) if args.max_records is None else min(args.max_records, len(record_ids))
    if records <= 0:
        raise ValueError("max_records must retain at least one record")
    record_ids = record_ids[:records]

    classes = diagnostic_class_names(scp_path)
    if len(classes) != 44 or "AFIB" in classes or "NORM" not in classes:
        raise ValueError("expected the ECGMamba 44-class diagnostic head")
    metadata = pd.read_csv(metadata_path, usecols=["ecg_id", "strat_fold", "scp_codes"])
    if metadata.ecg_id.duplicated().any():
        raise ValueError("PTB-XL metadata contains duplicate record IDs")
    metadata = metadata.set_index("ecg_id")
    if any(int(metadata.loc[int(record_id), "strat_fold"]) not in range(1, 9) for record_id in record_ids):
        raise ValueError("mask construction is restricted to official training folds 1-8")
    diagnostic_targets = _positive_targets(metadata, record_ids, classes)

    output_dir = args.output_dir.resolve()
    diag_dir, semantic_dir = output_dir / "diag", output_dir / "semantic"
    diag_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    diag_path = diag_dir / "region_masks_train.npy"
    semantic_path = semantic_dir / "region_masks_train.npy"
    diag_manifest_path = diag_dir / "manifest.json"
    semantic_manifest_path = semantic_dir / "manifest.json"
    progress_path = output_dir / "progress.json"
    dependencies = {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "metadata_sha256": _sha256(metadata_path),
        "scp_statements_sha256": _sha256(scp_path),
        "record_ids_sha256": _sha256(record_ids_path),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "compat_module_sha256": _sha256(
            REPO_ROOT / "src/rcfm/interpretability/ecgmambaformer_compat.py"
        ),
        "audit_protocol_sha256": _sha256(args.audit_protocol.resolve()),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "methods": [DIAG_METHOD, SEMANTIC_METHOD],
                "source_sha256": _sha256(source_path),
                "records": records,
                "dependencies": dependencies,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if diag_manifest_path.is_file() or semantic_manifest_path.is_file():
        if not diag_manifest_path.is_file() or not semantic_manifest_path.is_file():
            raise ValueError("only one completed task-head manifest exists")
        diag_manifest = json.loads(diag_manifest_path.read_text(encoding="utf-8"))
        semantic_manifest = json.loads(semantic_manifest_path.read_text(encoding="utf-8"))
        valid = (
            diag_manifest.get("status") == semantic_manifest.get("status") == "completed"
            and diag_manifest.get("build_fingerprint") == fingerprint
            and semantic_manifest.get("build_fingerprint") == fingerprint
            and diag_path.is_file()
            and semantic_path.is_file()
            and diag_manifest.get("mask", {}).get("sha256") == _sha256(diag_path)
            and semantic_manifest.get("mask", {}).get("sha256") == _sha256(semantic_path)
        )
        if valid:
            print(output_dir)
            return output_dir
        raise ValueError("completed ECGMamba task-head cache does not match this request")

    shape = (records, 1, 512)
    if progress_path.is_file() or diag_path.is_file() or semantic_path.is_file():
        if not progress_path.is_file() or not diag_path.is_file() or not semantic_path.is_file():
            raise ValueError("incomplete task-head cache is missing progress or an array")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("build_fingerprint") != fingerprint:
            raise ValueError("incomplete task-head cache provenance differs")
        completed = int(progress["completed_records"])
        diag_degenerate = int(progress.get("diagnostic_degenerate_records", 0))
        semantic_degenerate = int(progress.get("semantic_degenerate_records", 0))
        diag_masks = np.lib.format.open_memmap(diag_path, mode="r+", dtype=np.float32)
        semantic_masks = np.lib.format.open_memmap(semantic_path, mode="r+", dtype=np.float32)
        if diag_masks.shape != shape or semantic_masks.shape != shape:
            raise ValueError("incomplete task-head cache has an invalid shape")
    else:
        completed = 0
        diag_degenerate = 0
        semantic_degenerate = 0
        diag_masks = np.lib.format.open_memmap(diag_path, mode="w+", dtype=np.float32, shape=shape)
        semantic_masks = np.lib.format.open_memmap(
            semantic_path, mode="w+", dtype=np.float32, shape=shape
        )
        _atomic_json(
            progress_path,
            {"status": "running", "build_fingerprint": fingerprint, "completed_records": 0},
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = load_ecgmambaformer_fca_mgda(
        source_root, checkpoint_path, len(classes), map_location="cpu"
    ).eval().to(device)
    for index in range(completed, records):
        waveform_500hz = resample_poly(
            np.asarray(waveforms[index], dtype=np.float32), 500, 128, axis=0, padtype="line"
        ).astype(np.float32)
        if waveform_500hz.shape != (5000, 12):
            raise ValueError("ECGMamba resampling produced an invalid waveform")
        normalized, _mean, _scale = record_global_zscore(waveform_500hz)
        inputs = torch.from_numpy(normalized.T.copy()).unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        targets = diagnostic_targets[index]
        if targets:
            diag_cam, _outputs, _target = ecgmamba_task_head_gradcam(
                model, inputs, task="diag", diagnostic_target_indices=targets
            )
            diag_mask, degenerate = _project(diag_cam)
            diag_degenerate += int(degenerate)
        else:
            diag_mask = np.zeros(512, dtype=np.float32)
            diag_degenerate += 1
        try:
            semantic_cam, _outputs, _target = ecgmamba_task_head_gradcam(
                model, inputs, task="semantic"
            )
            semantic_mask, degenerate = _project(semantic_cam)
            semantic_degenerate += int(degenerate)
        except ValueError as error:
            if "predicts no P/QRS/T foreground" not in str(error):
                raise
            semantic_mask = np.zeros(512, dtype=np.float32)
            semantic_degenerate += 1
        diag_masks[index, 0] = diag_mask
        semantic_masks[index, 0] = semantic_mask
        completed = index + 1
        if completed % args.progress_every == 0 or completed == records:
            diag_masks.flush()
            semantic_masks.flush()
            _atomic_json(
                progress_path,
                {
                    "status": "running",
                    "build_fingerprint": fingerprint,
                    "completed_records": completed,
                    "diagnostic_degenerate_records": diag_degenerate,
                    "semantic_degenerate_records": semantic_degenerate,
                },
            )
            print(f"ECGMamba paired task-head masks: {completed}/{records}", flush=True)

    del model, diag_masks, semantic_masks
    diag_manifest = _manifest(
        method=DIAG_METHOD,
        mask_path=diag_path,
        source_path=source_path,
        record_ids_path=record_ids_path,
        records=records,
        dependencies=dependencies,
        checkpoint_path=checkpoint_path,
        classes=classes,
        degenerate_records=diag_degenerate,
        missing_target_records=sum(not targets for targets in diagnostic_targets),
        claim_boundary=(
            "Failed-localization negative control: the fold-9 gate found heterogeneous lag and "
            "negative median zero-lag correlation. This mask is not validated as a P/QRS/T region."
        ),
    )
    semantic_manifest = _manifest(
        method=SEMANTIC_METHOD,
        mask_path=semantic_path,
        source_path=source_path,
        record_ids_path=record_ids_path,
        records=records,
        dependencies=dependencies,
        checkpoint_path=checkpoint_path,
        classes=classes,
        degenerate_records=semantic_degenerate,
        missing_target_records=semantic_degenerate,
        claim_boundary=(
            "Training-only semantic-task Grad-CAM selected by a 64-record fold-9 localization gate; "
            "the unpublished model and algorithm-generated ECGdeli reference limit claims."
        ),
    )
    diag_manifest["build_fingerprint"] = fingerprint
    semantic_manifest["build_fingerprint"] = fingerprint
    diag_manifest["ecgmamba_protocol"]["task_head_score"] = (
        "mean pre-sigmoid logit over known-positive diagnostic classes; zero mask when absent"
    )
    semantic_manifest["ecgmamba_protocol"]["task_head_score"] = (
        "mean pre-softmax P/QRS/T logit on each class's own argmax support"
    )
    _atomic_json(diag_manifest_path, diag_manifest)
    _atomic_json(semantic_manifest_path, semantic_manifest)
    progress_path.unlink()
    print(output_dir)
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--scp_statements", type=Path, required=True)
    parser.add_argument("--ecgmambaformer_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit_protocol", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected_records", type=int, default=17440)
    parser.add_argument("--max_records", type=int, default=None, help="Smoke-test cap only.")
    parser.add_argument("--progress_every", type=int, default=25)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
