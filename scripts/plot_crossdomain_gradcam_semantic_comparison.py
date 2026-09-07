#!/usr/bin/env python3
"""Compare higher-resolution AFIB Grad-CAM with direct semantic masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_singlelead_af_faithfulness import batch_gradcam
from scripts.plot_crossdomain_af_gradcam_examples import select_group_examples
from src.rcfm.interpretability.ecgmambaformer_compat import (
    diagnostic_class_names,
    ecgmamba_task_head_gradcam,
    load_ecgmambaformer_fca_mgda,
    record_global_zscore,
)
from src.rcfm.interpretability.gradcam import (
    gradcam_native_multi_1d,
    normalize_soft_mask,
    project_cam_to_sample_grid,
    resample_mask_to_sample_grid,
    stitch_temporal_cams,
    validation_crop_starts,
)
from src.rcfm.interpretability.ptbxl_benchmark_compat import load_xresnet1d101


GROUP_ORDER = ("AF", "non-AF")
CAM_COLOR = "#b44c97"
SEMANTIC_COLOR = "#008b8b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_afib_metadata(mlb_path: Path, scaler_path: Path) -> tuple[int, float, float]:
    with mlb_path.open("rb") as handle:
        classes = np.asarray(pickle.load(handle).classes_, dtype=str)
    matches = np.flatnonzero(classes == "AFIB")
    if matches.tolist() != [4]:
        raise ValueError("expected AFIB at index 4 in the 71-class XResNet head")
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)
    mean = float(np.asarray(scaler.mean_).reshape(-1)[0])
    scale = float(np.asarray(scaler.scale_).reshape(-1)[0])
    if not np.isfinite([mean, scale]).all() or scale <= 0:
        raise ValueError("invalid PTB-XL scalar standardization")
    return int(matches[0]), mean, scale


def cpsc_xres_l5_masks(
    model: torch.nn.Module,
    waveforms: np.ndarray,
    target_index: int,
    mean: float,
    scale: float,
    device: torch.device,
) -> tuple[np.ndarray, list[bool]]:
    """Recompute stride-8 CAMs with explicit feature-center projection."""
    outputs, degeneracy = [], []
    target_layer = model[5][-1]
    for waveform in np.asarray(waveforms, dtype=np.float32):
        standardized = (waveform - mean) / scale
        starts = validation_crop_starts(len(standardized), 250, 125)
        projected = []
        for start in starts:
            crop = torch.from_numpy(standardized[start : start + 250].T.copy())
            inputs = crop.unsqueeze(0).to(device=device, dtype=torch.float32)
            native, _logits = gradcam_native_multi_1d(
                model, {"xres_l5": target_layer}, inputs, target_index
            )
            projected.append(project_cam_to_sample_grid(native["xres_l5"], 250, 8))
        stitched = stitch_temporal_cams(projected, starts, len(standardized))
        mask, degenerate = normalize_soft_mask(stitched)
        outputs.append(mask)
        degeneracy.append(bool(degenerate))
    return np.stack(outputs).astype(np.float32), degeneracy


def load_cpsc(
    diagnostic_npz: Path,
    semantic_npz: Path,
    xresnet_checkpoint: Path,
    benchmark_code_root: Path,
    mlb: Path,
    scaler: Path,
    semantic_checkpoint: Path,
    ecgmamba_root: Path,
    scp_statements: Path,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    with np.load(diagnostic_npz, allow_pickle=False) as diagnostic:
        record_ids = diagnostic["record_ids"].astype(str)
        groups = diagnostic["groups"].astype(str)
        waveforms = diagnostic["waveforms"].astype(np.float32)
    with np.load(semantic_npz, allow_pickle=False) as semantic:
        if not np.array_equal(record_ids, semantic["record_ids"].astype(str)):
            raise ValueError("CPSC diagnostic and semantic artifacts use different records")
        if not np.array_equal(groups, semantic["groups"].astype(str)):
            raise ValueError("CPSC diagnostic and semantic artifacts use different groups")
        normalized_inputs = semantic["normalized_inputs"].astype(np.float32)
        semantic_probabilities = semantic["semantic_probabilities"].astype(np.float32)
    if (waveforms.shape != (6, 1000, 12) or normalized_inputs.shape != (6, 5000, 12)
            or semantic_probabilities.shape != (6, 4, 5000)):
        raise ValueError("unexpected CPSC comparison artifact shapes")
    target_index, mean, scale = _load_afib_metadata(mlb, scaler)
    model = load_xresnet1d101(
        benchmark_code_root.resolve(), xresnet_checkpoint.resolve(), map_location="cpu"
    ).eval().to(device)
    diagnostic_masks, degeneracy = cpsc_xres_l5_masks(
        model, waveforms, target_index, mean, scale, device
    )
    classes = diagnostic_class_names(scp_statements.resolve())
    semantic_model = load_ecgmambaformer_fca_mgda(
        ecgmamba_root.resolve(), semantic_checkpoint.resolve(), len(classes), map_location="cpu"
    ).eval().to(device)
    semantic_gradcams, semantic_direct = [], []
    for normalized, probabilities in zip(normalized_inputs, semantic_probabilities):
        inputs = torch.from_numpy(normalized.T.copy()).unsqueeze(0).to(device=device, dtype=torch.float32)
        raw_cam, _outputs, _target = ecgmamba_task_head_gradcam(
            semantic_model, inputs, task="semantic"
        )
        projected_cam = resample_mask_to_sample_grid(raw_cam, 500, 100, output_length=1000)
        projected_direct = resample_mask_to_sample_grid(
            np.max(probabilities[1:], axis=0), 500, 100, output_length=1000
        )
        semantic_gradcams.append(normalize_soft_mask(projected_cam)[0])
        semantic_direct.append(normalize_soft_mask(projected_direct)[0])
    semantic_gradcams = np.stack(semantic_gradcams).astype(np.float32)
    semantic_direct = np.stack(semantic_direct).astype(np.float32)
    examples = []
    counters = {group: 0 for group in GROUP_ORDER}
    for index, group in enumerate(groups):
        counters[group] += 1
        examples.append({
            "group": group,
            "example_number": counters[group],
            "signal": ((waveforms[index, :, 1] - mean) / scale).astype(np.float32),
            "diagnostic_mask": diagnostic_masks[index],
            "semantic_gradcam": semantic_gradcams[index],
            "semantic_direct": semantic_direct[index],
            "diagnostic_degenerate": degeneracy[index],
            "source_row_index": index,
        })
    return examples, {
        "sampling_rate_hz": 100,
        "diagnostic_classifier": "PTB-XL 12-lead XResNet1D-101, fixed AFIB logit",
        "diagnostic_native_resolution_ms": 80,
        "diagnostic_projection": "xres_l5 stride-8 explicit sample-center projection and overlap-mean stitching",
        "semantic_model": "PTB-XL ECGMamba multitask direct semantic decoder",
        "semantic_gradcam_target": "mean semantic foreground logit on predicted P/QRS/T support",
        "semantic_direct_definition": "per-record normalized max(P,QRS,T) probability",
        "semantic_native_resolution_ms": 2,
        "display_lead": "II",
    }


def _mimic_semantic_masks(
    model: torch.nn.Module, raw_125hz: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    gradcams, direct_probabilities = [], []
    for raw in np.asarray(raw_125hz, dtype=np.float32):
        waveform = resample_poly(raw, 4, 1, padtype="line").astype(np.float32)[:2000]
        twelve_lead = np.repeat(waveform[:, None], 12, axis=1)
        normalized, _mean, _scale = record_global_zscore(twelve_lead)
        inputs = torch.from_numpy(normalized.T.copy()).unsqueeze(0).to(device=device, dtype=torch.float32)
        raw_cam, semantic_logits, _target = ecgmamba_task_head_gradcam(
            model, inputs, task="semantic"
        )
        logits = semantic_logits - np.max(semantic_logits, axis=0, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=0, keepdims=True)
        projected_cam = resample_mask_to_sample_grid(raw_cam, 500, 100, output_length=400)
        projected_direct = resample_mask_to_sample_grid(
            np.max(probabilities[1:], axis=0), 500, 100, output_length=400
        )
        gradcams.append(normalize_soft_mask(projected_cam)[0])
        direct_probabilities.append(normalize_soft_mask(projected_direct)[0])
    return np.stack(gradcams).astype(np.float32), np.stack(direct_probabilities).astype(np.float32)


def load_mimic(
    input_dir: Path,
    singlelead_checkpoint: Path,
    model_code_root: Path,
    semantic_checkpoint: Path,
    ecgmamba_root: Path,
    scp_statements: Path,
    count: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    sys.path.insert(0, str(model_code_root.resolve()))
    from LibMTL.model.singlelead_af import load_singlelead_af_checkpoint

    waveforms = np.load(input_dir / "ecg_ptb_normalized_100hz.npy", mmap_mode="r", allow_pickle=False)
    raw_125hz = np.load(input_dir / "ecg_mV_125hz.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(input_dir / "afib_labels.npy", allow_pickle=False).astype(np.uint8)
    groups = np.load(input_dir / "source_record_names.npy", allow_pickle=False).astype(str)
    if waveforms.shape != (len(labels), 1, 400) or raw_125hz.shape != (len(labels), 500):
        raise ValueError("unexpected MIMIC Lead-II preprocessing shapes")
    selected = select_group_examples(groups, labels, count)
    values_np = np.asarray(waveforms[selected], dtype=np.float32)
    diagnostic_model, diagnostic_checkpoint = load_singlelead_af_checkpoint(
        singlelead_checkpoint.resolve(), map_location=device
    )
    diagnostic_model.to(device).eval()
    values = torch.from_numpy(values_np).to(device)
    _logits, diagnostic_masks, diagnostic_degenerate = batch_gradcam(
        diagnostic_model, diagnostic_model.gradcam_target_layer, values
    )
    classes = diagnostic_class_names(scp_statements.resolve())
    if len(classes) != 44 or "AFIB" in classes:
        raise ValueError("expected the frozen 44-class ECGMamba semantic checkpoint")
    semantic_model = load_ecgmambaformer_fca_mgda(
        ecgmamba_root.resolve(), semantic_checkpoint.resolve(), len(classes), map_location="cpu"
    ).eval().to(device)
    semantic_gradcams, semantic_direct = _mimic_semantic_masks(
        semantic_model, raw_125hz[selected], device
    )
    examples = []
    counters = {group: 0 for group in GROUP_ORDER}
    for local_index, source_index in enumerate(selected):
        group = "AF" if labels[source_index] else "non-AF"
        counters[group] += 1
        examples.append({
            "group": group,
            "example_number": counters[group],
            "signal": values_np[local_index, 0],
            "diagnostic_mask": diagnostic_masks[local_index].float().cpu().numpy(),
            "semantic_gradcam": semantic_gradcams[local_index],
            "semantic_direct": semantic_direct[local_index],
            "diagnostic_degenerate": bool(diagnostic_degenerate[local_index].item()),
            "source_row_index": int(source_index),
            "source_record_hash": hashlib.sha256(groups[source_index].encode("utf-8")).hexdigest()[:12],
        })
    return examples, {
        "sampling_rate_hz": 100,
        "diagnostic_classifier": "PTB-XL Lead-II single-lead ECGMamba, fixed AFIB logit",
        "diagnostic_checkpoint_epoch": int(diagnostic_checkpoint["epoch"]),
        "diagnostic_native_resolution_ms": 10,
        "semantic_model": "PTB-XL ECGMamba multitask direct semantic decoder with Lead-II replication",
        "semantic_gradcam_target": "mean semantic foreground logit on predicted P/QRS/T support",
        "semantic_direct_definition": "per-record normalized max(P,QRS,T) probability",
        "semantic_native_resolution_ms": 2,
        "display_lead": "II",
        "selection": "central window of equally spaced sorted records per label; score and mask excluded",
    }


def plot_comparison(
    examples: list[dict[str, object]], metadata: dict[str, object], output_stem: Path
) -> list[Path]:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman", "Times"],
        "font.size": 7.5, "axes.titlesize": 8.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(len(examples), 3, figsize=(7.16, 1.12 * len(examples)), squeeze=False)
    rate = float(metadata["sampling_rate_hz"])
    columns = (
        ("diagnostic_mask", "AFIB Grad-CAM", CAM_COLOR),
        ("semantic_gradcam", "Semantic-head Grad-CAM", SEMANTIC_COLOR),
        ("semantic_direct", "Semantic direct probability", "#d17a00"),
    )
    for row_index, example in enumerate(examples):
        signal = np.asarray(example["signal"], dtype=np.float32)
        time = np.arange(len(signal), dtype=np.float32) / rate
        y0, y1 = np.quantile(signal, [0.01, 0.99])
        margin = max(float(y1 - y0) * 0.08, 1e-3)
        for column, (mask_key, label, color) in enumerate(columns):
            axis = axes[row_index, column]
            mask = np.asarray(example[mask_key], dtype=np.float32)
            if mask.shape != signal.shape or not np.isfinite(mask).all():
                raise ValueError("waveform and comparison masks must be aligned finite vectors")
            axis.plot(time, signal, color="#222222", linewidth=0.55)
            twin = axis.twinx()
            twin.fill_between(time, 0, mask, color=color, alpha=0.24, linewidth=0)
            twin.plot(time, mask, color=color, linewidth=0.55)
            twin.set(ylim=(0, 1.03), yticks=[])
            axis.set(xlim=(0, len(signal) / rate), ylim=(float(y0 - margin), float(y1 + margin)))
            if row_index == 0:
                axis.set_title(label)
            if row_index == len(examples) - 1:
                axis.set_xlabel("Time (s)")
            else:
                axis.set_xticklabels([])
            if column == 0:
                group = "AF" if example["group"] == "AF" else "non-AF"
                suffix = "*" if example.get("diagnostic_degenerate", False) else ""
                axis.set_ylabel(f"{group} {example['example_number']}{suffix}")
            else:
                axis.set_yticklabels([])
    figure.tight_layout(pad=0.45)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        path = output_stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=600 if suffix == "png" else None)
        outputs.append(path)
    plt.close(figure)
    return outputs


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cpsc_examples, cpsc_metadata = load_cpsc(
        args.cpsc_diagnostic_npz.resolve(), args.cpsc_semantic_npz.resolve(),
        args.xresnet_checkpoint.resolve(), args.benchmark_code_root.resolve(),
        args.mlb.resolve(), args.scaler.resolve(), args.semantic_checkpoint.resolve(),
        args.ecgmamba_root.resolve(), args.scp_statements.resolve(), device,
    )
    mimic_examples, mimic_metadata = load_mimic(
        args.mimic_input_dir.resolve(), args.mimic_singlelead_checkpoint.resolve(),
        args.model_code_root.resolve(), args.semantic_checkpoint.resolve(),
        args.ecgmamba_root.resolve(), args.scp_statements.resolve(),
        args.examples_per_group, device,
    )
    outputs = {
        "cpsc2018": plot_comparison(cpsc_examples, cpsc_metadata, output_dir / "cpsc2018_gradcam_semantic_examples"),
        "mimic_afib": plot_comparison(mimic_examples, mimic_metadata, output_dir / "mimic_afib_gradcam_semantic_examples"),
    }
    published = []
    if args.publish_dir is not None:
        publish_dir = args.publish_dir.resolve()
        publish_dir.mkdir(parents=True, exist_ok=True)
        for paths in outputs.values():
            for source in paths:
                destination = publish_dir / source.name
                shutil.copy2(source, destination)
                published.append(str(destination))
    report = {
        "schema_version": 1, "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure_contract": "six rows by three methods; no suptitle and no footer note",
        "datasets": {"cpsc2018": cpsc_metadata, "mimic_afib": mimic_metadata},
        "selection": {
            "cpsc2018": [{"group": row["group"], "example_number": row["example_number"],
                          "source_row_index": row["source_row_index"], "diagnostic_degenerate": row["diagnostic_degenerate"]}
                         for row in cpsc_examples],
            "mimic_afib": [{"group": row["group"], "example_number": row["example_number"],
                            "source_row_index": row["source_row_index"], "source_record_hash": row["source_record_hash"],
                            "diagnostic_degenerate": row["diagnostic_degenerate"]} for row in mimic_examples],
        },
        "inputs_sha256": {
            "cpsc_diagnostic_npz": sha256(args.cpsc_diagnostic_npz.resolve()),
            "cpsc_semantic_npz": sha256(args.cpsc_semantic_npz.resolve()),
            "xresnet_checkpoint": sha256(args.xresnet_checkpoint.resolve()),
            "mimic_manifest": sha256((args.mimic_input_dir / "manifest.json").resolve()),
            "mimic_singlelead_checkpoint": sha256(args.mimic_singlelead_checkpoint.resolve()),
            "semantic_checkpoint": sha256(args.semantic_checkpoint.resolve()),
        },
        "outputs": [str(path) for paths in outputs.values() for path in paths], "published_outputs": published,
        "claim_boundary": (
            "Direct semantic morphology is a dense decoder probability, not Grad-CAM. CPSC and MIMIC diagnostic CAMs use "
            "different frozen AF classifiers; the MIMIC semantic model requires exploratory single-lead replication."
        ),
    }
    report["output_sha256"] = {path.name: sha256(path) for paths in outputs.values() for path in paths}
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return protocol_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpsc_diagnostic_npz", type=Path, default=Path("runs/gradcam_transfer/cpsc_xresnet1d101_af_vs_nonaf_seed31_v2/gradcam_results.npz"))
    parser.add_argument("--cpsc_semantic_npz", type=Path, default=Path("runs/gradcam_transfer/cpsc_ecgmamba_fca_mgda_s42_norm_semantic_v4/ecgmamba_transfer_results.npz"))
    parser.add_argument("--xresnet_checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark_code_root", type=Path, required=True)
    parser.add_argument("--mlb", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--mimic_input_dir", type=Path, default=Path("runs/preprocessing/mimic_afib_leadii_physical_v1"))
    parser.add_argument("--mimic_singlelead_checkpoint", type=Path, default=Path("runs/training/ptbxl_leadii_af/20260901T063746Z_ptbxl_leadii_af_seed31/best.pt"))
    parser.add_argument("--model_code_root", type=Path, required=True)
    parser.add_argument("--semantic_checkpoint", type=Path, required=True)
    parser.add_argument("--ecgmamba_root", type=Path, required=True)
    parser.add_argument("--scp_statements", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("runs/figures/crossdomain_gradcam_semantic_comparison_v2"))
    parser.add_argument("--publish_dir", type=Path)
    parser.add_argument("--examples_per_group", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
