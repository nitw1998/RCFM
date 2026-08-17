#!/usr/bin/env python3
"""Reproduce the historical single-lead PTB-XL CFM evaluation.

This entry point intentionally does not use the newer multi-lead evaluation stack.
It matches the checkpoint-era III-to-V5, first-512-samples, per-record min-max task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from metrics import calculate_FD  # noqa: E402
from model import ConditionNet, DiffusionUNetCrossAttention  # noqa: E402
from train_cfm_basic import MinimalFlowMatching  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-root", type=Path, default=REPO / "data/PTBXL")
    parser.add_argument("--official-root", type=Path, default=REPO.parent / "runs/preprocessing/ptbxl_official_minmax_v1/PTBXL")
    parser.add_argument("--checkpoint-root", type=Path, default=REPO / "saved/PTBXL")
    parser.add_argument("--split-source", choices=("historical_val", "official_fold9"), default="historical_val")
    parser.add_argument("--epochs", type=int, nargs="+", default=(979, 999))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--legacy-fd-batch-sizes", type=int, nargs="+", default=(64, 256))
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _minmax_per_record(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32))
    low = values.min(axis=1, keepdims=True)
    span = values.max(axis=1, keepdims=True) - low
    if np.any(span <= 0) or not np.all(np.isfinite(span)):
        raise ValueError("legacy PTB-XL records must have finite nonzero range")
    return (2.0 * (values - low) / span - 1.0).astype(np.float32)


def load_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, object], Path]:
    heldout_root = args.historical_root if args.split_source == "historical_val" else args.official_root
    heldout_path = heldout_root / "X_val_resampled.npy"
    heldout = np.load(heldout_path, mmap_mode="r")
    count = len(heldout) if args.max_records is None else min(args.max_records, len(heldout))
    stats = {
        "method": "per_record_per_selected_lead_minmax_neg1_1",
        "source_commit": "c366eee",
        "source_expression": "sklearn.preprocessing.minmax_scale(..., (-1, 1), axis=1)",
    }
    target = _minmax_per_record(heldout[:count, :512, 10])
    source = _minmax_per_record(heldout[:count, :512, 2])
    return target[:, None], source[:, None], stats, heldout_path


def legacy_batch_fd(target: np.ndarray, prediction: np.ndarray, batch_size: int) -> float:
    values = []
    for start in range(0, len(target), batch_size):
        stop = min(start + batch_size, len(target))
        if stop - start < 2:
            continue
        values.append(calculate_FD(torch.from_numpy(target[start:stop]), torch.from_numpy(prediction[start:stop])))
    return float(np.mean(values))


@torch.inference_mode()
def predict(args: argparse.Namespace, epoch: int, source: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
    device = torch.device(args.device)
    flow_path = args.checkpoint_root / f"minimal_cfm_epoch_{epoch}.pth"
    condition_path = args.checkpoint_root / f"condition_net_epoch_{epoch}.pth"
    model = MinimalFlowMatching(DiffusionUNetCrossAttention(512, 1, device, num_heads=4)).to(device)
    condition = ConditionNet().to(device)
    model.load_state_dict(torch.load(flow_path, map_location="cpu"), strict=True)
    condition.load_state_dict(torch.load(condition_path, map_location="cpu"), strict=True)
    model.eval(); condition.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(source), torch.from_numpy(noise)), batch_size=args.batch_size, shuffle=False, num_workers=0)
    chunks = []
    for source_batch, noise_batch in loader:
        source_batch, state = source_batch.to(device), noise_batch.to(device)
        cond = condition(source_batch)
        for step in range(50):
            time = torch.full((len(state),), step / 50, device=device)
            state = state + model.flow_model(state, cond, time) / 50
        chunks.append(state.cpu().numpy())
    return np.concatenate(chunks).astype(np.float32), {"flow": sha256(flow_path), "condition": sha256(condition_path)}


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=False)
    target, source, stats, heldout_path = load_data(args)
    noise = np.random.default_rng(args.seed).standard_normal(target.shape, dtype=np.float32)
    np.savez_compressed(args.output / "paired_reference.npz", targets=target, conditions=source, initial_noise=noise)
    summary = {
        "protocol": "legacy_ptbxl_single_lead_cfm_v1", "split_source": args.split_source,
        "split_caveat": "historical_val provenance cannot be independently matched to the current official fold9" if args.split_source == "historical_val" else "cross-artifact sensitivity using historical training scaler",
        "records": len(target), "condition_lead": "III", "target_lead": "V5", "samples": 512,
        "normalization": "checkpoint-era per-record per-selected-lead min-max to [-1,1]",
        "normalization_provenance": stats, "seed": args.seed, "sampling_steps": 50,
        "heldout_path": str(heldout_path.resolve()), "heldout_sha256": sha256(heldout_path), "epochs": {},
    }
    for epoch in args.epochs:
        prediction, checkpoint_hashes = predict(args, epoch, source, noise)
        prediction_path = args.output / f"predictions_epoch_{epoch}.npy"
        np.save(prediction_path, prediction, allow_pickle=False)
        metrics = {
            "rmse": float(np.sqrt(np.mean((prediction - target) ** 2, dtype=np.float64))),
            "mae": float(np.mean(np.abs(prediction - target), dtype=np.float64)),
            "full_fold_waveform_fd": float(calculate_FD(torch.from_numpy(target), torch.from_numpy(prediction))),
            "legacy_contiguous_batch_mean_fd": {str(size): legacy_batch_fd(target, prediction, size) for size in args.legacy_fd_batch_sizes},
        }
        summary["epochs"][str(epoch)] = {"metrics": metrics, "checkpoint_sha256": checkpoint_hashes, "prediction_sha256": sha256(prediction_path)}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(epoch, json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
