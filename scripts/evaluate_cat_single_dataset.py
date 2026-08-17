"""Evaluate frozen single-output CAT-PPG/CAT-RCG endpoints."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _array_sha256, _sha256, _waveform_metrics
from scripts.evaluate_mimic_phase_corrected import _fixed_support_align
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic
from src.rcfm.baselines.cat_checkpoint import load_cat_checkpoint
from src.rcfm.baselines.cat_data import load_cat_datasets
from src.rcfm.baselines.catransformer import CATransformer


CONTRACTS = {
    "WESAD": {"records": 4213, "label": "CAT-PPG (reproduced)", "task": "ppg2ecg"},
    "mmECG": {"records": 2877, "label": "CAT-RCG (adapted)", "task": "rcg2ecg"},
}


def _model(config: dict[str, object]) -> CATransformer:
    return CATransformer(
        input_length=int(config["window_size"]) * int(config["sampling_rate"]),
        output_channels=1, cat_layers=int(config["cat_layers"]), top_k=int(config["top_k"]),
        patch_width=int(config["patch_width"]), d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]), encoder_layers=int(config["encoder_layers"]),
        ff_dim=int(config["ff_dim"]), dropout=float(config["dropout"]),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@torch.inference_mode()
def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = load_cat_checkpoint(args.checkpoint.resolve(), map_location="cpu")
    config = dict(checkpoint["config"])
    dataset = str(config["datasets"])
    contract = CONTRACTS.get(dataset)
    if contract is None or config.get("reproduction_label") != contract["label"]:
        raise ValueError("this entry accepts only frozen single-output WESAD/mmECG CAT endpoints")
    if config.get("task") != contract["task"] or int(checkpoint["epoch"]) != 500:
        raise ValueError("CAT endpoint task/epoch violates the frozen protocol")
    if int(checkpoint["output_spec"]["channels"]) != 1:
        raise ValueError("single-dataset CAT evaluation requires one output channel")
    load_config = dict(config)
    load_config.update(data_root=str(args.data_root.resolve()),
                       max_train_records=1, max_test_records=None)
    load_args = argparse.Namespace(**load_config)
    _, heldout, normalization = load_cat_datasets(load_args)
    if len(heldout) != contract["records"] or normalization != checkpoint["normalization"]:
        raise ValueError("CAT held-out data disagree with checkpoint protocol")
    root = args.data_root / dataset
    subjects = np.load(root / "subject_ids_test.npy", allow_pickle=False).astype(str)
    if subjects.shape != (contract["records"],):
        raise ValueError("CAT held-out subject IDs disagree with row count")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    torch.manual_seed(31)
    model = _model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
    loader = DataLoader(heldout, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    targets, sources, predictions = [], [], []
    cycle_fallbacks, reconstruction_fallbacks = [], []
    for target, source in loader:
        prediction, diagnostics = model(source.float().to(device), return_diagnostics=True)
        targets.append(target.numpy()); sources.append(source.numpy()); predictions.append(prediction.cpu().numpy())
        cycle_fallbacks.extend(torch.stack([v for k, v in diagnostics.items() if k.endswith("cycle_fallback")]).any(0).cpu().tolist())
        reconstruction_fallbacks.extend(torch.stack([v for k, v in diagnostics.items() if k.endswith("reconstruction_fallback")]).any(0).cpu().tolist())
    target = np.concatenate(targets).astype(np.float32)
    source = np.concatenate(sources).astype(np.float32)
    prediction = np.concatenate(predictions).astype(np.float32)
    expected_shape = (contract["records"], 1, 512)
    if any(array.shape != expected_shape or not np.all(np.isfinite(array)) for array in (target, source, prediction)):
        raise ValueError("CAT evaluation arrays violate the frozen shape/finite contract")
    raw_summary, raw_rows = _waveform_metrics(target, prediction)
    lag_summary, lag_values = _lag_diagnostic(target, prediction, 16, 128)
    shifts = lag_values["best_lag_samples"].astype(np.int32)
    center, unshifted, aligned = _fixed_support_align(target, prediction, shifts, 16)
    before_summary, before_rows = _waveform_metrics(center, unshifted)
    after_summary, after_rows = _waveform_metrics(center, aligned)
    raw_path = output / "paired_predictions.npz"
    phase_path = output / "phase_predictions_maxlag16.npz"
    np.savez_compressed(raw_path, targets=target, predictions=prediction, sources=source,
                        subject_ids=subjects)
    np.savez_compressed(phase_path, targets=center, cat_unshifted_predictions=unshifted,
                        cat_oracle_aligned_predictions=aligned, cat_oracle_shifts=shifts,
                        subject_ids=subjects)
    rows = []
    for index in range(len(target)):
        rows.append({"row": index, "subject_id": subjects[index],
                     "raw_rmse": raw_rows["rmse"][index], "raw_mae": raw_rows["mae"][index],
                     "raw_pearson_r": raw_rows["pearson_r"][index], "oracle_shift_samples": shifts[index],
                     "unshifted_fixed_rmse": before_rows["rmse"][index],
                     "oracle_aligned_rmse": after_rows["rmse"][index],
                     "oracle_aligned_pearson_r": after_rows["pearson_r"][index]})
    metrics_path = output / "per_window_metrics.csv"; _write_csv(metrics_path, rows)
    summary_path = output / "waveform_summary.json"
    summary = {"schema_version": 1, "dataset": dataset, "model": contract["label"],
               "raw_full_window": raw_summary, "unshifted_fixed_support": before_summary,
               "oracle_aligned_fixed_support": after_summary,
               "oracle_lag_diagnostic": {**lag_summary,
                   "boundary_fraction": float(np.mean(np.abs(shifts) == 16)),
                   "rmse_improved_fraction": float(np.mean(after_rows["rmse"] < before_rows["rmse"]))},
               "diagnostics": {"cycle_fallback_records": int(np.sum(cycle_fallbacks)),
                   "reconstruction_fallback_records": int(np.sum(reconstruction_fallbacks))},
               "interpretation": "Raw full-window results are primary; oracle alignment uses test targets and is diagnostic only."}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    outputs = (raw_path, phase_path, metrics_path, summary_path)
    protocol = {"schema_version": 1, "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": shlex.join(sys.argv), "dataset": dataset, "records": len(target),
                "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": _sha256(args.checkpoint),
                               "epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"]},
                "array_hashes": {"target": _array_sha256(target), "prediction": _array_sha256(prediction)},
                "execution": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": str(device)},
                "outputs": {path.name: _sha256(path) for path in outputs}}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
