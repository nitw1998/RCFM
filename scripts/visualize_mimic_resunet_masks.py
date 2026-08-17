"""Visualize frozen PTB-XL+ ResUNet masks on MIMIC-AFib training ECG windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.experiment import atomic_json
from src.rcfm.interpretability.delineation_dataset import WAVE_NAMES, robust_window_normalize
from src.rcfm.interpretability.delineation_unet import DelineationResUNet1D


COLORS = {"p": "#2A9D8F", "qrs": "#D55E00", "t": "#3572B0"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _region_spans(binary: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.pad(np.asarray(binary, dtype=np.int8), (1, 1)))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _refined_pan_peaks(signal: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, info = nk.ecg_peaks(
            signal,
            sampling_rate=sample_rate_hz,
            method="pantompkins1985",
            correct_artifacts=True,
        )
    peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64)
    radius = max(1, int(round(0.1 * sample_rate_hz)))
    refined: list[int] = []
    for peak in peaks:
        start, stop = max(0, peak - radius), min(len(signal), peak + radius + 1)
        baseline = float(np.median(signal[max(0, peak - 4 * radius) : min(len(signal), peak + 4 * radius + 1)]))
        refined.append(int(start + np.argmax(np.abs(signal[start:stop] - baseline))))
    return np.asarray(sorted(set(refined)), dtype=np.int64)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    return parser


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    if args.examples < 1:
        raise ValueError("examples must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    source_path = args.data_root.resolve() / "MIMIC-AFib" / "ecg_train_4sec.npy"
    checkpoint_path = args.checkpoint.resolve()
    if not source_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("MIMIC training ECG or ResUNet checkpoint is missing")
    raw = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if raw.ndim != 2 or raw.shape[1] != 512:
        raise ValueError("MIMIC training ECG must have shape (records, 512)")
    indices = np.linspace(0, len(raw) - 1, min(args.examples, len(raw)), dtype=np.int64)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("kind") != "ptbxl_plus_delineation_resunet1d":
        raise ValueError("checkpoint is not a PTB-XL+ delineation ResUNet")
    config = checkpoint["config"]
    if int(config["sample_rate_hz"]) != 128 or int(config["window_samples"]) != 512:
        raise ValueError("checkpoint sampling grid does not match MIMIC windows")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = DelineationResUNet1D(base_channels=int(config["base_channels"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval().to(device)

    normalized = np.stack(
        [robust_window_normalize(np.asarray(raw[index]), float(config["normalization_clip_z"])) for index in indices]
    )
    inputs = torch.from_numpy(normalized[:, None, :]).to(device=device, dtype=torch.float32)
    probabilities = torch.sigmoid(model(inputs)["region_logits"]).cpu().numpy()
    threshold = float(config["region_threshold"])
    refined_peaks = [_refined_pan_peaks(signal, 128) for signal in normalized]

    _set_plot_style()
    figure, axes = plt.subplots(len(indices), 2, figsize=(7.16, 1.38 * len(indices)), squeeze=False)
    time = np.arange(512) / 128.0
    for row, (index, signal, masks, peaks) in enumerate(
        zip(indices, normalized, probabilities, refined_peaks)
    ):
        waveform_axis, mask_axis = axes[row]
        waveform_axis.plot(time, signal, color="#222222", linewidth=0.7)
        for wave_index, wave in enumerate(WAVE_NAMES):
            for start, stop in _region_spans(masks[wave_index] >= threshold):
                waveform_axis.axvspan(
                    start / 128.0,
                    stop / 128.0,
                    color=COLORS[wave],
                    alpha=0.18,
                    linewidth=0,
                )
        for peak in peaks:
            waveform_axis.axvline(peak / 128.0, color="#111111", linestyle=":", linewidth=0.65)
        waveform_axis.set_ylabel(f"Window {int(index)}\nnormalized ECG")
        waveform_axis.set_xlim(0, 4)
        waveform_axis.spines[["top", "right"]].set_visible(False)

        for wave_index, wave in enumerate(WAVE_NAMES):
            mask_axis.plot(time, masks[wave_index], color=COLORS[wave], label=f"{wave.upper()} prob.")
        mask_axis.axhline(threshold, color="#777777", linestyle="--", linewidth=0.6)
        for peak in peaks:
            mask_axis.axvline(peak / 128.0, color="#111111", linestyle=":", linewidth=0.65)
        mask_axis.set_xlim(0, 4)
        mask_axis.set_ylim(-0.02, 1.02)
        mask_axis.set_ylabel("Mask probability")
        mask_axis.spines[["top", "right"]].set_visible(False)
        mask_axis.grid(axis="y", color="#EEEEEE", linewidth=0.4)
    axes[0, 1].legend(frameon=False, ncol=3, loc="upper center")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    figure.suptitle(
        "PTB-XL+-trained ResUNet on MIMIC-AFib training ECG (no phase correction)",
        y=0.997,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.982), pad=0.5)
    figure_path = output_dir / "mimic_resunet_mask_examples.png"
    pdf_path = output_dir / "mimic_resunet_mask_examples.pdf"
    figure.savefig(figure_path, dpi=600)
    figure.savefig(pdf_path)
    plt.close(figure)

    occupancy = (probabilities >= threshold).mean(axis=(0, 2))
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "visualization_only_cross_domain_audit",
        "source_split": "MIMIC-AFib training ECG only",
        "test_data_read": False,
        "phase_correction": False,
        "selection": "equally spaced row indices selected before inference",
        "row_indices": [int(value) for value in indices],
        "source_ecg_sha256": _sha256(source_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "normalization": f"robust_window_normalize_clip_z_{float(config['normalization_clip_z'])}",
        "region_threshold": threshold,
        "binary_occupancy": {
            name: float(occupancy[index]) for index, name in enumerate(WAVE_NAMES)
        },
        "outputs": {
            figure_path.name: _sha256(figure_path),
            pdf_path.name: _sha256(pdf_path),
        },
        "claim_boundary": (
            "Anonymous qualitative cross-domain audit only; Pan-Tompkins lines are algorithmic "
            "references, not manual labels, and no transfer accuracy is established."
        ),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(figure_path)
    return figure_path


if __name__ == "__main__":
    run(build_argparser().parse_args())
