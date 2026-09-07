#!/usr/bin/env python3
"""Plot fixed AF/non-AF Grad-CAM examples for CPSC2018 and MIMIC-AFib."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_singlelead_af_faithfulness import batch_gradcam


GROUP_ORDER = ("AF", "non-AF")
GROUP_TITLES = {"AF": "AF-positive", "non-AF": "Non-AF"}
CAM_COLOR = "#b44c97"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_group_examples(groups: np.ndarray, labels: np.ndarray, count: int) -> list[int]:
    """Select central windows from equally spaced sorted records, without scores/CAMs."""
    groups = np.asarray(groups).astype(str)
    labels = np.asarray(labels, dtype=np.uint8)
    selected: list[int] = []
    for label in (1, 0):
        names = np.unique(groups[labels == label])
        if len(names) < count:
            raise ValueError(f"only {len(names)} records are available for label {label}")
        positions = np.linspace(0, len(names) - 1, count, dtype=np.int64)
        for name in names[positions]:
            indices = np.flatnonzero(groups == name)
            if len(np.unique(labels[indices])) != 1:
                raise ValueError(f"conflicting labels within source record {name}")
            selected.append(int(indices[len(indices) // 2]))
    return selected


def load_cpsc_examples(npz_path: Path, protocol_path: Path, count: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    standardization = protocol["input"]["standardization"]
    mean, scale = float(standardization["mean"]), float(standardization["scale"])
    if not np.isfinite([mean, scale]).all() or scale <= 0:
        raise ValueError("invalid CPSC classifier standardization")
    with np.load(npz_path, allow_pickle=False) as artifact:
        groups = artifact["groups"].astype(str)
        waveforms = artifact["waveforms"]
        masks = artifact["masks"]
        classes = artifact["class_names"].astype(str)
        probabilities = artifact["aggregate_probabilities"]
        matches = np.flatnonzero(classes == "AFIB")
        if matches.tolist() != [4] or waveforms.shape[1:] != (1000, 12) or masks.shape != (len(groups), 1000):
            raise ValueError("unexpected frozen CPSC Grad-CAM artifact contract")
        examples: list[dict[str, object]] = []
        for group in GROUP_ORDER:
            indices = np.flatnonzero(groups == group)
            if len(indices) < count:
                raise ValueError(f"only {len(indices)} saved CPSC examples for {group}")
            for example_number, index in enumerate(indices[:count], start=1):
                examples.append({
                    "group": group,
                    "example_number": example_number,
                    "signal": ((waveforms[index, :, 1] - mean) / scale).astype(np.float32),
                    "mask": masks[index].astype(np.float32),
                    "probability": float(probabilities[index, 4]),
                    "degenerate": bool(np.ptp(masks[index]) <= 1e-12),
                    "source_row_index": int(index),
                })
    metadata = {
        "classifier": "PTB-XL all-statements XResNet1D-101 (12-lead)",
        "target": "AFIB logit",
        "display_lead": "II",
        "sampling_rate_hz": 100,
        "samples": 1000,
        "selection": "existing seed-31 label-stratified QC-valid examples; prediction and CAM excluded from selection",
    }
    return examples, metadata


def load_mimic_examples(
    input_dir: Path,
    checkpoint_path: Path,
    model_code_root: Path,
    count: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    sys.path.insert(0, str(model_code_root.resolve()))
    from LibMTL.model.singlelead_af import load_singlelead_af_checkpoint

    waveforms = np.load(input_dir / "ecg_ptb_normalized_100hz.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(input_dir / "afib_labels.npy", allow_pickle=False).astype(np.uint8)
    groups = np.load(input_dir / "source_record_names.npy", allow_pickle=False).astype(str)
    if waveforms.shape != (len(labels), 1, 400) or len(groups) != len(labels):
        raise ValueError("unexpected MIMIC Lead-II preprocessing contract")
    selected = select_group_examples(groups, labels, count)
    values_np = np.asarray(waveforms[selected], dtype=np.float32)
    model, checkpoint = load_singlelead_af_checkpoint(checkpoint_path, map_location=device)
    model.to(device).eval()
    values = torch.from_numpy(values_np).to(device)
    logits, masks, degenerate = batch_gradcam(model, model.gradcam_target_layer, values)
    logits_np = logits.float().cpu().numpy()
    masks_np = masks.float().cpu().numpy()
    degenerate_np = degenerate.cpu().numpy()
    examples = []
    counters = {group: 0 for group in GROUP_ORDER}
    for local_index, source_index in enumerate(selected):
        group = "AF" if labels[source_index] else "non-AF"
        counters[group] += 1
        examples.append({
            "group": group,
            "example_number": counters[group],
            "signal": values_np[local_index, 0],
            "mask": masks_np[local_index],
            "probability": float(torch.sigmoid(torch.tensor(logits_np[local_index])).item()),
            "degenerate": bool(degenerate_np[local_index]),
            "source_row_index": int(source_index),
            "source_record_hash": hashlib.sha256(groups[source_index].encode("utf-8")).hexdigest()[:12],
        })
    metadata = {
        "classifier": "PTB-XL Lead-II single-lead ECGMamba AF classifier",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "target": "AFIB logit",
        "display_lead": "II",
        "sampling_rate_hz": 100,
        "samples": 400,
        "selection": "central window of equally spaced sorted records within each label; prediction and CAM excluded from selection",
    }
    return examples, metadata


def plot_examples(examples: list[dict[str, object]], metadata: dict[str, object], output_stem: Path, title: str) -> list[Path]:
    grouped = {group: [row for row in examples if row["group"] == group] for group in GROUP_ORDER}
    rows = max(len(grouped[group]) for group in GROUP_ORDER)
    if any(len(grouped[group]) != rows for group in GROUP_ORDER):
        raise ValueError("AF and non-AF example counts must match")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman", "Times"],
        "font.size": 7.5, "axes.titlesize": 8.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(rows, 2, figsize=(7.16, 1.42 * rows), squeeze=False)
    rate = float(metadata["sampling_rate_hz"])
    for column, group in enumerate(GROUP_ORDER):
        for row_index, example in enumerate(grouped[group]):
            axis = axes[row_index, column]
            signal = np.asarray(example["signal"], dtype=np.float32)
            mask = np.asarray(example["mask"], dtype=np.float32)
            if signal.shape != mask.shape or not np.isfinite(signal).all() or not np.isfinite(mask).all():
                raise ValueError("example waveform and mask must be aligned finite vectors")
            time = np.arange(len(signal), dtype=np.float32) / rate
            y0, y1 = np.quantile(signal, [0.01, 0.99])
            margin = max(float(y1 - y0) * 0.10, 1e-3)
            axis.plot(time, signal, color="#222222", linewidth=0.55)
            twin = axis.twinx()
            twin.fill_between(time, 0, mask, color=CAM_COLOR, alpha=0.24, linewidth=0)
            twin.plot(time, mask, color=CAM_COLOR, linewidth=0.55)
            twin.set(ylim=(0, 1.03), yticks=[])
            axis.set(xlim=(0, len(signal) / rate), ylim=(float(y0 - margin), float(y1 + margin)))
            suffix = " | degenerate CAM" if example["degenerate"] else ""
            axis.set_ylabel(f"Ex. {example['example_number']}\np(AF)={example['probability']:.3f}{suffix}", fontsize=6.5)
            if row_index == 0:
                axis.set_title(GROUP_TITLES[group])
            if row_index == rows - 1:
                axis.set_xlabel("Time (s)")
            else:
                axis.set_xticklabels([])
            if column == 1:
                axis.set_yticklabels([])
    figure.suptitle(title, fontsize=8.5, y=0.995)
    figure.text(
        0.5, 0.008, "Black: PTB-standardized Lead-II ECG; magenta: normalized AFIB Grad-CAM",
        ha="center", fontsize=6.5,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94), pad=0.45)
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
    cpsc_examples, cpsc_metadata = load_cpsc_examples(
        args.cpsc_npz.resolve(), args.cpsc_protocol.resolve(), args.examples_per_group
    )
    mimic_examples, mimic_metadata = load_mimic_examples(
        args.mimic_input_dir.resolve(), args.mimic_checkpoint.resolve(),
        args.model_code_root.resolve(), args.examples_per_group, device,
    )
    outputs = {
        "cpsc2018": plot_examples(
            cpsc_examples, cpsc_metadata, output_dir / "cpsc2018_af_gradcam_examples",
            "Frozen PTB-XL AFIB Grad-CAM on CPSC2018 (Lead II shown)",
        ),
        "mimic_afib": plot_examples(
            mimic_examples, mimic_metadata, output_dir / "mimic_afib_af_gradcam_examples",
            "Frozen PTB-XL Lead-II AFIB Grad-CAM on MIMIC-AFib",
        ),
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
        "reference_figure": str(args.reference_figure.resolve()), "examples_per_group": args.examples_per_group,
        "inputs": {
            "cpsc_npz": {"path": str(args.cpsc_npz.resolve()), "sha256": sha256(args.cpsc_npz.resolve())},
            "cpsc_protocol": {"path": str(args.cpsc_protocol.resolve()), "sha256": sha256(args.cpsc_protocol.resolve())},
            "mimic_manifest": {"path": str((args.mimic_input_dir / 'manifest.json').resolve()),
                               "sha256": sha256((args.mimic_input_dir / 'manifest.json').resolve())},
            "mimic_checkpoint": {"path": str(args.mimic_checkpoint.resolve()), "sha256": sha256(args.mimic_checkpoint.resolve())},
        },
        "datasets": {"cpsc2018": cpsc_metadata, "mimic_afib": mimic_metadata},
        "selected_examples": {
            "cpsc2018": [{key: row[key] for key in ("group", "example_number", "probability", "degenerate", "source_row_index")} for row in cpsc_examples],
            "mimic_afib": [{key: row[key] for key in ("group", "example_number", "probability", "degenerate", "source_row_index", "source_record_hash")} for row in mimic_examples],
        },
        "outputs": [str(path) for paths in outputs.values() for path in paths], "published_outputs": published,
        "claim_boundary": (
            "Qualitative classifier-relative Grad-CAM examples, not anatomical segmentation. "
            "CPSC uses a frozen 12-lead XResNet classifier while MIMIC uses the lead-matched single-lead ECGMamba classifier."
        ),
    }
    for path in [path for paths in outputs.values() for path in paths]:
        report.setdefault("output_sha256", {})[path.name] = sha256(path)
    report_path = output_dir / "protocol.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpsc_npz", type=Path, default=Path("runs/gradcam_transfer/cpsc_xresnet1d101_af_vs_nonaf_seed31_v2/gradcam_results.npz"))
    parser.add_argument("--cpsc_protocol", type=Path, default=Path("runs/gradcam_transfer/cpsc_xresnet1d101_af_vs_nonaf_seed31_v2/protocol.json"))
    parser.add_argument("--mimic_input_dir", type=Path, default=Path("runs/preprocessing/mimic_afib_leadii_physical_v1"))
    parser.add_argument("--mimic_checkpoint", type=Path, default=Path("runs/training/ptbxl_leadii_af/20260901T063746Z_ptbxl_leadii_af_seed31/best.pt"))
    parser.add_argument("--model_code_root", type=Path, required=True)
    parser.add_argument("--reference_figure", type=Path, default=Path("paper/data/figures/ptbxl_ecgmamba_taskhead_gradcam_examples.pdf"))
    parser.add_argument("--output_dir", type=Path, default=Path("runs/figures/crossdomain_af_gradcam_examples_v2"))
    parser.add_argument("--publish_dir", type=Path)
    parser.add_argument("--examples_per_group", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
