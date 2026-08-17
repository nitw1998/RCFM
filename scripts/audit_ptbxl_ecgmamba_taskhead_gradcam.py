"""Audit ECGMamba diagnostic- and semantic-head Grad-CAM on PTB-XL fold 9."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from scipy.special import softmax
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_mimic_region_masks import _soft_alignment_metrics
from src.rcfm.interpretability.delineation_dataset import build_window_targets
from src.rcfm.interpretability.ecgmambaformer_compat import (
    diagnostic_class_names,
    ecgmamba_task_head_gradcam,
    load_ecgmambaformer_fca_mgda,
    record_global_zscore,
)
from src.rcfm.interpretability.gradcam import (
    normalize_soft_mask,
    resample_mask_to_sample_grid,
)


METHODS = ("diag_head_gradcam", "semantic_head_gradcam", "semantic_direct_probability")
METHOD_LABELS = {
    "diag_head_gradcam": "Diagnostic-head Grad-CAM",
    "semantic_head_gradcam": "Semantic-head Grad-CAM",
    "semantic_direct_probability": "Semantic direct probability",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _positive_diagnostic_indices(
    metadata: pd.DataFrame,
    record_id: int,
    class_to_index: dict[str, int],
) -> tuple[int, ...]:
    codes = ast.literal_eval(str(metadata.loc[int(record_id), "scp_codes"]))
    if not isinstance(codes, dict):
        raise ValueError("PTB-XL scp_codes must decode to a dictionary")
    return tuple(sorted(class_to_index[code] for code in codes if code in class_to_index))


def _project_first_four_seconds(values: np.ndarray) -> tuple[np.ndarray, bool]:
    cam = np.asarray(values, dtype=np.float32)
    if cam.shape != (5000,) or not np.all(np.isfinite(cam)):
        raise ValueError("ECGMamba CAM must contain one 10-second 500-Hz vector")
    projected = resample_mask_to_sample_grid(
        cam[:2000], 500, 128, output_length=512
    )
    return normalize_soft_mask(projected)


def _prepare_input(waveform_128hz: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    values = np.asarray(waveform_128hz, dtype=np.float32)
    if values.shape != (1280, 12) or not np.all(np.isfinite(values)):
        raise ValueError("PTB-XL waveform must have shape (1280, 12) and be finite")
    waveform_500hz = resample_poly(values, 500, 128, axis=0, padtype="line").astype(np.float32)
    if waveform_500hz.shape != (5000, 12):
        raise ValueError("500-Hz resampling produced an unexpected shape")
    normalized, _mean, _scale = record_global_zscore(waveform_500hz)
    return torch.from_numpy(normalized.T.copy()).unsqueeze(0), normalized


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    metrics = (
        "mass_in_reference", "inside_minus_outside", "top20_dice",
        "zero_lag_correlation", "best_lag_samples", "best_lag_correlation",
        "mask_degenerate", "soft_occupancy",
    )
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        aggregate: dict[str, object] = {"method": method, "records": len(selected)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            aggregate[f"mean_{metric}"] = float(np.mean(values))
            aggregate[f"median_{metric}"] = float(np.median(values))
        output.append(aggregate)
    return output


def _plot_examples(examples: list[dict[str, object]], output: Path) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif"],
            "font.size": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(len(examples), 3, figsize=(7.16, 1.18 * len(examples)), squeeze=False)
    colors = {
        "diag_head_gradcam": "#b44c97",
        "semantic_head_gradcam": "#008b8b",
        "semantic_direct_probability": "#d17a00",
    }
    time = np.arange(512) / 128.0
    for row_index, example in enumerate(examples):
        signal = np.asarray(example["signal"])
        reference = np.asarray(example["reference"])
        y0, y1 = np.quantile(signal, [0.01, 0.99])
        margin = max(float(y1 - y0) * 0.08, 1e-3)
        for column, method in enumerate(METHODS):
            axis = axes[row_index, column]
            axis.plot(time, signal, color="#222222", linewidth=0.55)
            axis.fill_between(
                time, y0 - margin, y1 + margin, where=reference > 0,
                color="#7f7f7f", alpha=0.12, linewidth=0,
            )
            mask = np.asarray(example["masks"][method])
            twin = axis.twinx()
            twin.fill_between(time, 0, mask, color=colors[method], alpha=0.24, linewidth=0)
            twin.plot(time, mask, color=colors[method], linewidth=0.55)
            twin.set(ylim=(0, 1.03), yticks=[])
            axis.set(xlim=(0, 4), ylim=(y0 - margin, y1 + margin))
            if row_index == 0:
                axis.set_title(METHOD_LABELS[method], fontsize=8)
            if column == 0:
                axis.set_ylabel(f"Row {example['selection_index']}")
            else:
                axis.set_yticklabels([])
            if row_index == len(examples) - 1:
                axis.set_xlabel("Time (s)")
            else:
                axis.set_xticklabels([])
    figure.suptitle(
        "ECGMamba task-head localization on PTB-XL fold 9 (gray: ECGdeli P/QRS/T)",
        fontsize=8.5,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.968), pad=0.45)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    figure.savefig(png, dpi=600)
    figure.savefig(pdf)
    plt.close(figure)
    return [png, pdf]


def run(args: argparse.Namespace) -> Path:
    if args.split != "val":
        raise ValueError("task-head mask selection audit is restricted to fold-9 validation")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)

    waveform_root = args.waveform_root.resolve()
    sidecar_root = args.sidecar_root.resolve()
    source_root = args.ecgmambaformer_root.resolve()
    waveform_path = waveform_root / "X_val_resampled.npy"
    waveform_ids_path = waveform_root / "record_ids_val.npy"
    required = (
        waveform_path, waveform_ids_path, sidecar_root / "dataset_manifest.json",
        sidecar_root / "record_ids_val.npy", sidecar_root / "eligible_leads_val.npy",
        sidecar_root / "wave_valid_val.npy", sidecar_root / "fiducial_positions_val.npy",
        sidecar_root / "fiducial_counts_val.npy", args.metadata.resolve(),
        args.scp_statements.resolve(), args.checkpoint.resolve(),
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("a required waveform, sidecar, metadata, or checkpoint file is missing")

    manifest = json.loads((sidecar_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    lead_names = tuple(str(value) for value in manifest["selected_leads"])
    lead_index = lead_names.index("II")
    waveform_ids = np.load(waveform_ids_path, allow_pickle=False).astype(np.int64)
    sidecar_ids = np.load(sidecar_root / "record_ids_val.npy", allow_pickle=False).astype(np.int64)
    if not np.array_equal(waveform_ids, sidecar_ids):
        raise ValueError("waveform and PTB-XL+ sidecar record order differs")
    waveforms = np.load(waveform_path, mmap_mode="r", allow_pickle=False)
    if waveforms.shape != (len(waveform_ids), 1280, 12):
        raise ValueError("fold-9 waveform array has an unexpected shape")
    eligible = np.load(sidecar_root / "eligible_leads_val.npy", allow_pickle=False).astype(bool)
    wave_valid = np.load(sidecar_root / "wave_valid_val.npy", allow_pickle=False).astype(bool)
    positions = np.load(sidecar_root / "fiducial_positions_val.npy", mmap_mode="r", allow_pickle=False)
    counts = np.load(sidecar_root / "fiducial_counts_val.npy", mmap_mode="r", allow_pickle=False)

    classes = diagnostic_class_names(args.scp_statements.resolve())
    if len(classes) != 44 or "AFIB" in classes or "NORM" not in classes:
        raise ValueError("expected the ECGMamba 44-class diagnostic head")
    class_to_index = {name: index for index, name in enumerate(classes)}
    metadata = pd.read_csv(args.metadata.resolve(), usecols=["ecg_id", "strat_fold", "scp_codes"])
    if metadata.ecg_id.duplicated().any():
        raise ValueError("PTB-XL metadata contains duplicate record IDs")
    metadata = metadata.set_index("ecg_id")
    positive_targets = [
        _positive_diagnostic_indices(metadata, int(record_id), class_to_index)
        for record_id in waveform_ids
    ]
    candidates = np.flatnonzero(
        eligible[:, lead_index]
        & wave_valid[:, lead_index].all(axis=1)
        & np.asarray([bool(value) for value in positive_targets])
    )
    if len(candidates) < args.records:
        raise ValueError(f"only {len(candidates)} records satisfy the frozen audit criteria")
    selected = candidates[np.linspace(0, len(candidates) - 1, args.records, dtype=np.int64)]
    if len(np.unique(selected)) != args.records:
        raise ValueError("deterministic selection produced duplicate records")

    model = load_ecgmambaformer_fca_mgda(
        source_root, args.checkpoint.resolve(), len(classes), map_location="cpu"
    ).eval().to(device)
    rows: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    example_positions = set(
        np.linspace(0, args.records - 1, min(args.figure_records, args.records), dtype=np.int64).tolist()
    )
    for selection_index, record_index in enumerate(selected.tolist()):
        inputs, normalized = _prepare_input(np.asarray(waveforms[record_index]))
        inputs = inputs.to(device=device, dtype=torch.float32)
        diag_cam, _diag_logits, diag_target = ecgmamba_task_head_gradcam(
            model,
            inputs,
            task="diag",
            diagnostic_target_indices=positive_targets[record_index],
        )
        semantic_cam, semantic_logits, semantic_target = ecgmamba_task_head_gradcam(
            model, inputs, task="semantic"
        )
        diag_mask, diag_degenerate = _project_first_four_seconds(diag_cam)
        semantic_mask, semantic_degenerate = _project_first_four_seconds(semantic_cam)
        semantic_probabilities = softmax(semantic_logits, axis=0)
        direct = np.max(semantic_probabilities[1:, :2000], axis=0)
        direct_mask = resample_mask_to_sample_grid(direct, 500, 128, output_length=512)
        direct_mask, direct_degenerate = normalize_soft_mask(direct_mask)
        target = build_window_targets(
            positions[record_index, lead_index],
            counts[record_index, lead_index],
            wave_valid[record_index, lead_index],
            crop_start=0,
            window_samples=512,
        )
        reference = np.max(target["regions"], axis=0)
        masks = {
            "diag_head_gradcam": diag_mask,
            "semantic_head_gradcam": semantic_mask,
            "semantic_direct_probability": direct_mask,
        }
        degeneracy = {
            "diag_head_gradcam": diag_degenerate,
            "semantic_head_gradcam": semantic_degenerate,
            "semantic_direct_probability": direct_degenerate,
        }
        for method, mask in masks.items():
            metrics = _soft_alignment_metrics(mask, reference, max_lag=args.max_lag_samples)
            rows.append(
                {
                    "selection_index": selection_index,
                    "record_index": record_index,
                    "record_id": int(waveform_ids[record_index]),
                    "method": method,
                    "positive_diagnostic_classes": ";".join(
                        classes[index] for index in positive_targets[record_index]
                    ),
                    "semantic_target_classes": ";".join(
                        str(value) for value in semantic_target["semantic_target_classes"]
                    ),
                    **metrics,
                    "mask_degenerate": float(degeneracy[method]),
                    "soft_occupancy": float(np.mean(mask)),
                }
            )
        if selection_index in example_positions:
            signal = normalized[:2000, 1]
            signal = resample_poly(signal, 128, 500, padtype="line").astype(np.float32)[:512]
            examples.append(
                {
                    "selection_index": selection_index,
                    "signal": signal,
                    "reference": reference,
                    "masks": masks,
                    "diag_target": diag_target,
                    "semantic_target": semantic_target,
                }
            )
        if (selection_index + 1) % args.progress_every == 0 or selection_index + 1 == args.records:
            print(f"ECGMamba task-head audit: {selection_index + 1}/{args.records}", flush=True)

    aggregate = _aggregate(rows)
    per_record_path = output_dir / "per_record_localization.csv"
    aggregate_path = output_dir / "localization_summary.csv"
    _write_csv(per_record_path, rows)
    _write_csv(aggregate_path, aggregate)
    figure_paths = _plot_examples(examples, output_dir / "taskhead_gradcam_examples")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": args.records,
                "selection_split": "official_fold_9",
                "reference": "PTB-XL+ ECGdeli Lead-II P/QRS/T union",
                "methods": aggregate,
                "training_mask_gate": "descriptive audit only; no head accepted automatically",
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    outputs = [per_record_path, aggregate_path, summary_path, *figure_paths]
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "scope": "ECGMamba diagnostic-versus-semantic task-head Grad-CAM localization gate",
        "selection": {
            "split": "official_fold_9",
            "records": args.records,
            "rule": "equally spaced among Lead-II eligible records with valid P/QRS/T and >=1 diagnostic label",
            "fold_10_used": False,
        },
        "model": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
            "architecture": "local unpublished ECGMambaImproved + FCA-MGDA multitask checkpoint",
            "diagnostic_classes": classes,
            "strict_checkpoint_load": True,
        },
        "input": {
            "native_waveform": "PTB-XL 10 seconds at 128 Hz, 12 leads",
            "model_grid": "polyphase resampling to 10 seconds at 500 Hz",
            "normalization": "per-record global z-score across time and leads",
            "reported_window": "first 4 seconds projected to the 128-Hz RCFM grid by sample-center interpolation",
        },
        "gradcam": {
            "target_layer": "shared full-resolution encoder output (1024 x 5000)",
            "diag_score": "mean pre-sigmoid logit over known-positive diagnostic classes",
            "semantic_score": "mean pre-softmax P/QRS/T class logit on each class's own argmax support",
            "channel_weight": "temporal mean activation gradient",
            "positive_evidence": "ReLU",
            "per_record_normalization": "min-max [0,1] after first-four-second projection",
        },
        "reference": {
            "source": "PTB-XL+ ECGdeli algorithm-generated fiducials",
            "lead": "II",
            "claim_boundary": "algorithm-generated annotation agreement, not manual clinical ground truth",
        },
        "claim_boundary": (
            "This fold-9 audit selects whether a task-head mask is technically eligible for a later "
            "training-only ablation. It is not generation evidence and does not use fold 10."
        ),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "dependencies": {
            "script_sha256": _sha256(Path(__file__).resolve()),
            "compat_module_sha256": _sha256(
                REPO_ROOT / "src/rcfm/interpretability/ecgmambaformer_compat.py"
            ),
            "sidecar_manifest_sha256": _sha256(sidecar_root / "dataset_manifest.json"),
            "waveform_sha256": _sha256(waveform_path),
        },
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"ECGMamba task-head audit saved to {output_dir}")
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waveform_root", type=Path, required=True)
    parser.add_argument("--sidecar_root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--scp_statements", type=Path, required=True)
    parser.add_argument("--ecgmambaformer_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--records", type=int, default=64)
    parser.add_argument("--figure_records", type=int, default=6)
    parser.add_argument("--max_lag_samples", type=int, default=32)
    parser.add_argument("--progress_every", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
