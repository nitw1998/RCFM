"""Evaluate one deterministic CAT-PPG checkpoint on frozen MIMIC-AFib pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import neurokit2 as nk
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _waveform_metrics
from src.rcfm.baselines.cat_checkpoint import load_cat_checkpoint
from src.rcfm.baselines.cat_data import load_mimic_cat_datasets
from src.rcfm.baselines.catransformer import CATransformer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model(config: dict[str, object]) -> CATransformer:
    return CATransformer(
        input_length=int(config["window_size"]) * int(config["sampling_rate"]),
        output_channels=1, cat_layers=int(config["cat_layers"]), top_k=int(config["top_k"]),
        patch_width=int(config["patch_width"]), d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]), encoder_layers=int(config["encoder_layers"]),
        ff_dim=int(config["ff_dim"]), dropout=float(config["dropout"]),
    )


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = load_cat_checkpoint(args.checkpoint, map_location="cpu")
    if not args.allow_nonfinal_checkpoint and int(checkpoint["epoch"]) != 500:
        raise ValueError("formal CAT evaluation requires the predeclared epoch-500 endpoint")
    config = dict(checkpoint["config"])
    _, test_set, normalization = load_mimic_cat_datasets(
        args.data_root, max_train_records=1, max_test_records=args.max_records
    )
    if normalization != checkpoint["normalization"]:
        raise ValueError("CAT evaluation normalization differs from its checkpoint")
    expected = 1800 if args.max_records is None else min(args.max_records, 1800)
    if len(test_set) != expected:
        raise ValueError("CAT test record count changed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = _model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    predictions, targets, sources = [], [], []
    cycle_fallbacks, reconstruction_fallbacks = [], []
    for target, source in loader:
        prediction, diagnostics = model(source.to(device), return_diagnostics=True)
        predictions.append(prediction.cpu().numpy()); targets.append(target.numpy()); sources.append(source.numpy())
        cycle_fallbacks.extend(
            torch.stack([value.float() for key, value in diagnostics.items() if key.endswith("cycle_fallback")])
            .any(dim=0).cpu().numpy().tolist()
        )
        reconstruction_fallbacks.extend(
            torch.stack([value.float() for key, value in diagnostics.items() if key.endswith("reconstruction_fallback")])
            .any(dim=0).cpu().numpy().tolist()
        )
    prediction = np.concatenate(predictions).astype(np.float32)
    target = np.concatenate(targets).astype(np.float32)
    source = np.concatenate(sources).astype(np.float32)
    np.savez_compressed(output / "paired_predictions.npz", predictions=prediction, targets=target, sources=source)
    summary, per_record = _waveform_metrics(target, prediction)
    summary.update({
        "cycle_extraction_fallback_records": int(np.sum(cycle_fallbacks)),
        "cycle_extraction_fallback_rate": float(np.mean(cycle_fallbacks)),
        "reconstruction_padding_records": int(np.sum(reconstruction_fallbacks)),
        "reconstruction_padding_rate": float(np.mean(reconstruction_fallbacks)),
        "nfe": 1,
    })
    (output / "waveform_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    with (output / "per_record_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["row", "rmse", "mae", "bias", "pearson_r"])
        for row in range(len(target)):
            writer.writerow([row, per_record["rmse"][row], per_record["mae"][row],
                             per_record["bias"][row], per_record["pearson_r"][row]])
    protocol = {
        "schema_version": 1,
        "status": "completed" if len(target) == 1800 and int(checkpoint["epoch"]) == 500 else "smoke_completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv),
        "model": "CAT-PPG (reproduced)", "implementation": "independent_paper_based_reproduction",
        "paper_doi": "10.1109/JBHI.2024.3482853", "deterministic": True, "nfe": 1,
        "cycle_extraction": "source PPG FFT only; no target ECG input",
        "dataset": {"name": "MIMIC-AFib", "records": len(target),
                    "dataset_version": config["dataset_version"], "split_hash": config["split_hash"],
                    "alignment_id": config["alignment_id"], "normalization_id": config["normalization_id"]},
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": _sha256(args.checkpoint),
                       "epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"]},
        "artifacts": {name: _sha256(output / name) for name in
                      ("paired_predictions.npz", "waveform_summary.json", "per_record_metrics.csv")},
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "neurokit2": getattr(nk, "__version__", "unknown"),
                     "device": str(device)},
        "claim_boundary": "Independent paper reproduction, one seed, normalized-domain waveform analysis; test set is final-only.",
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--allow_nonfinal_checkpoint", action="store_true")
    return parser


if __name__ == "__main__":
    result = run(build_argparser().parse_args())
    print(f"CAT-PPG evaluation saved to {result}")
