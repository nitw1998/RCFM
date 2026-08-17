"""Build PTB-XL training-only P/QRS/T masks with a frozen PTB-XL+ ResUNet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.interpretability.delineation_dataset import (
    PTBXL_LEAD_ORDER,
    WAVE_NAMES,
    robust_window_normalize,
)
from src.rcfm.interpretability.delineation_unet import DelineationResUNet1D


METHOD = "ptbxl_plus_resunet_p_qrs_t_lead_ii_soft_max_epoch13_v1"
DATASET_VERSION = "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1"
SPLIT_HASH = "7784c98a2c8daccc23fc7cb0d47dc1933eeee77f120c05f8a24ac149cd8474f7"
SOURCE_LEAD = "II"
SOURCE_LEAD_INDEX = PTBXL_LEAD_ORDER.index(SOURCE_LEAD)


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


def semantic_soft_mask(region_logits: torch.Tensor) -> torch.Tensor:
    """Collapse P/QRS/T probabilities into one temporal soft mask."""

    if region_logits.ndim != 3 or region_logits.shape[1] != len(WAVE_NAMES):
        raise ValueError("region logits must have shape (batch, 3, samples)")
    probabilities = torch.sigmoid(region_logits.float())
    masks = probabilities.amax(dim=1, keepdim=True)
    if not torch.isfinite(masks).all():
        raise ValueError("semantic masks contain non-finite values")
    return masks


def _validate_checkpoint(
    checkpoint: dict[str, object],
    checkpoint_path: Path,
    sidecar_manifest_path: Path,
    expected_epoch: int,
) -> tuple[dict[str, object], dict[str, object]]:
    if checkpoint.get("kind") != "ptbxl_plus_delineation_resunet1d":
        raise ValueError("checkpoint is not a PTB-XL+ delineation ResUNet")
    if int(checkpoint.get("epoch", -1)) != expected_epoch:
        raise ValueError("checkpoint epoch does not match the frozen fold-9 selection")
    config = checkpoint.get("config")
    provenance = checkpoint.get("provenance")
    output_spec = checkpoint.get("output_spec")
    if not isinstance(config, dict) or not isinstance(provenance, dict) or not isinstance(output_spec, dict):
        raise ValueError("delineation checkpoint is missing config/provenance/output_spec")
    sidecar = json.loads(sidecar_manifest_path.read_text(encoding="utf-8"))
    required = {
        "sample_rate_hz": 128,
        "window_samples": 512,
        "waveform_split_hash": SPLIT_HASH,
    }
    mismatched = [name for name, value in required.items() if config.get(name) != value]
    if mismatched:
        raise ValueError(f"delineation checkpoint contract mismatch: {mismatched}")
    if output_spec.get("region_classes") != list(WAVE_NAMES):
        raise ValueError("delineation checkpoint region classes are not P/QRS/T")
    if sidecar.get("waveform_split_hash") != SPLIT_HASH:
        raise ValueError("delineation sidecar does not match the PTB-XL waveform split")
    if provenance.get("dataset_manifest_sha256") != _sha256(sidecar_manifest_path):
        raise ValueError("delineation sidecar checksum differs from checkpoint provenance")
    if provenance.get("split_and_eligibility_hash") != sidecar.get("split_and_eligibility_hash"):
        raise ValueError("delineation sidecar eligibility hash differs from checkpoint provenance")
    if not isinstance(checkpoint.get("model_state"), dict):
        raise ValueError(f"checkpoint has no model state: {checkpoint_path}")
    return config, sidecar


def _plot_examples(waveforms: np.ndarray, masks: np.ndarray, output_dir: Path) -> list[str]:
    indices = np.linspace(0, len(waveforms) - 1, min(4, len(waveforms)), dtype=np.int64)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(len(indices), 1, figsize=(7.16, 1.25 * len(indices)), squeeze=False)
    time = np.arange(512) / 128.0
    for axis, index in zip(axes[:, 0], indices):
        signal = robust_window_normalize(
            np.asarray(waveforms[int(index), :512, SOURCE_LEAD_INDEX], dtype=np.float32)
        )
        mask = np.asarray(masks[int(index), 0], dtype=np.float32)
        axis.plot(time, signal, color="#222222", linewidth=0.7, label="Lead II")
        axis.fill_between(time, -1.0, 2.0 * mask - 1.0, color="#2A9D8F", alpha=0.22)
        axis.plot(time, 2.0 * mask - 1.0, color="#0072B2", linewidth=0.65, label="SemanticMask")
        axis.set(xlim=(0, 4), ylim=(-1.05, 1.05), ylabel=f"Row {int(index)}")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1, 0].set_xlabel("Time (s)")
    figure.tight_layout(pad=0.6)
    names = ["semantic_resunet_examples.png", "semantic_resunet_examples.pdf"]
    figure.savefig(output_dir / names[0], dpi=600)
    figure.savefig(output_dir / names[1])
    plt.close(figure)
    return names


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--delineation_sidecar_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--expected_records", type=int, default=17440)
    parser.add_argument("--expected_epoch", type=int, default=13)
    parser.add_argument("--max_records", type=int, default=None, help="Smoke-test cap only.")
    parser.add_argument("--progress_every", type=int, default=1024)
    return parser


@torch.inference_mode()
def run(args: argparse.Namespace) -> Path:
    if args.batch_size <= 0 or args.progress_every <= 0:
        raise ValueError("batch_size and progress_every must be positive")
    dataset_root = args.data_root.resolve() / "PTBXL"
    source_path = dataset_root / "X_train_resampled.npy"
    record_ids_path = dataset_root / "record_ids_train.npy"
    checkpoint_path = args.checkpoint.resolve()
    sidecar_manifest_path = args.delineation_sidecar_root.resolve() / "dataset_manifest.json"
    for path in (source_path, record_ids_path, checkpoint_path, sidecar_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"required SemanticMask input is missing: {path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config, sidecar = _validate_checkpoint(
        checkpoint, checkpoint_path, sidecar_manifest_path, args.expected_epoch
    )
    waveforms = np.load(source_path, mmap_mode="r", allow_pickle=False)
    record_ids = np.load(record_ids_path, allow_pickle=False)
    if waveforms.shape != (args.expected_records, 1280, 12):
        raise ValueError("PTB-XL training waveform shape does not match the frozen contract")
    if record_ids.shape != (args.expected_records,) or len(np.unique(record_ids)) != len(record_ids):
        raise ValueError("PTB-XL training record IDs are missing, duplicated, or misaligned")
    records = len(waveforms) if args.max_records is None else min(args.max_records, len(waveforms))
    if records <= 0:
        raise ValueError("max_records must leave at least one training record")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dependencies = {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "sidecar_manifest_sha256": _sha256(sidecar_manifest_path),
        "record_ids_sha256": _sha256(record_ids_path),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "model_module_sha256": _sha256(REPO_ROOT / "src/rcfm/interpretability/delineation_unet.py"),
        "normalization_module_sha256": _sha256(
            REPO_ROOT / "src/rcfm/interpretability/delineation_dataset.py"
        ),
    }
    fingerprint_payload = {
        "method": METHOD,
        "source_sha256": _sha256(source_path),
        "records": records,
        "checkpoint_epoch": args.expected_epoch,
        "dependencies": dependencies,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    final_path = output_dir / "region_masks_train.npy"
    partial_path = output_dir / "region_masks_train.partial.npy"
    progress_path = output_dir / "progress.json"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "completed"
            and manifest.get("build_fingerprint") == fingerprint
            and final_path.is_file()
            and manifest.get("mask", {}).get("sha256") == _sha256(final_path)
        ):
            print(final_path)
            return final_path
        raise ValueError("existing completed SemanticMask artifact does not match this request")

    completed = 0
    if partial_path.is_file() or progress_path.is_file():
        if not partial_path.is_file() or not progress_path.is_file():
            raise ValueError("incomplete SemanticMask cache is missing its array or progress record")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("build_fingerprint") != fingerprint:
            raise ValueError("incomplete SemanticMask cache provenance does not match this request")
        completed = int(progress.get("completed_records", 0))
        masks = np.lib.format.open_memmap(partial_path, mode="r+", dtype=np.float32)
        if masks.shape != (records, 1, 512):
            raise ValueError("incomplete SemanticMask cache has an invalid shape")
    else:
        masks = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float32, shape=(records, 1, 512)
        )
        _atomic_json(
            progress_path,
            {"status": "running", "build_fingerprint": fingerprint, "completed_records": 0},
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = DelineationResUNet1D(int(config["base_channels"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    clip_z = float(config["normalization_clip_z"])
    next_report = ((completed // args.progress_every) + 1) * args.progress_every
    for start in range(completed, records, args.batch_size):
        stop = min(start + args.batch_size, records)
        normalized = np.stack(
            [
                robust_window_normalize(
                    np.asarray(waveforms[index, :512, SOURCE_LEAD_INDEX], dtype=np.float32),
                    clip_z,
                )
                for index in range(start, stop)
            ]
        )
        inputs = torch.from_numpy(normalized[:, None]).to(device=device, dtype=torch.float32)
        outputs = model(inputs)
        batch_masks = semantic_soft_mask(outputs["region_logits"]).cpu().numpy()
        masks[start:stop] = batch_masks
        if stop >= next_report or stop == records:
            masks.flush()
            _atomic_json(
                progress_path,
                {"status": "running", "build_fingerprint": fingerprint, "completed_records": stop},
            )
            print(f"Semantic ResUNet masks: {stop}/{records}", flush=True)
            next_report = ((stop // args.progress_every) + 1) * args.progress_every

    masks.flush()
    del masks, model
    os.replace(partial_path, final_path)
    complete_masks = np.load(final_path, mmap_mode="r", allow_pickle=False)
    if not np.isfinite(complete_masks).all() or complete_masks.min() < 0 or complete_masks.max() > 1:
        raise ValueError("completed SemanticMask cache failed finite/range validation")
    figures = _plot_examples(waveforms[:records], complete_masks, output_dir)
    flat = np.asarray(complete_masks).reshape(-1)
    degenerate = int(
        np.sum(np.ptp(np.asarray(complete_masks), axis=-1).reshape(-1) <= np.finfo(np.float32).eps)
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_fingerprint": fingerprint,
        "dataset": {
            "name": "PTBXL",
            "dataset_version": DATASET_VERSION,
            "split_hash": SPLIT_HASH,
            "train_records": records,
            "source_ecg": {"file_name": source_path.name, "sha256": _sha256(source_path)},
            "record_ids_sha256": _sha256(record_ids_path),
            "sampling_rate_hz": 128,
            "folds": list(range(1, 9)),
            "window": "first 512 samples (4 seconds)",
        },
        "mask": {
            "method": METHOD,
            "file_name": final_path.name,
            "shape": list(complete_masks.shape),
            "dtype": str(complete_masks.dtype),
            "sha256": _sha256(final_path),
            "soft_occupancy_mean": float(flat.mean()),
            "quantiles": [float(value) for value in np.quantile(flat, [0, 0.25, 0.5, 0.75, 1])],
            "degenerate_records": degenerate,
            "broadcast_policy": "one Lead-II temporal mask broadcast to all 11 target leads",
        },
        "semantic_protocol": {
            "architecture": "DelineationResUNet1D",
            "training_labels": "PTB-XL+ ECGdeli algorithm-generated P/QRS/T regions",
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_selection": "maximum fold-9 model_selection_score",
            "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
            "checkpoint_best_score": float(checkpoint["best_score"]),
            "source_lead": SOURCE_LEAD,
            "source_lead_index": SOURCE_LEAD_INDEX,
            "normalization": f"per-window robust median/IQR, clip_z={clip_z:g}",
            "region_classes": list(WAVE_NAMES),
            "soft_aggregation": "maximum sigmoid probability over P/QRS/T at each sample",
            "thresholding": "none",
            "fold10_used_for_checkpoint_or_mask_generation": False,
        },
        "delineation_sidecar": {
            "dataset_version": sidecar.get("dataset_version"),
            "split_and_eligibility_hash": sidecar.get("split_and_eligibility_hash"),
            "annotation_provenance": sidecar.get("annotation_provenance"),
        },
        "dependencies": dependencies,
        "figures": {name: _sha256(output_dir / name) for name in figures},
        "test_mask_generated": False,
        "mask_usage": "training_loss_weight_only",
        "claim_boundary": (
            "Source-derived P/QRS/T soft mask trained against algorithm-generated PTB-XL+ ECGdeli "
            "annotations; not manual clinical ground truth and not used during validation or inference."
        ),
    }
    _atomic_json(manifest_path, manifest)
    progress_path.unlink()
    print(final_path)
    return final_path


if __name__ == "__main__":
    run(build_argparser().parse_args())
