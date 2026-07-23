"""Create lightweight paper figures required by the LaTeX source."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT_DIR = Path("figs")


def draw_architecture() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")

    blocks = [
        ("Source\nECG/PPG/RCG", 0.05, 0.58, 0.16, 0.22, "#dbeafe"),
        ("Condition\nEncoder", 0.28, 0.58, 0.16, 0.22, "#dcfce7"),
        ("Gaussian\nNoise", 0.05, 0.20, 0.16, 0.22, "#f3f4f6"),
        ("Region Mask\nPan/Grad-CAM", 0.28, 0.20, 0.16, 0.22, "#fde68a"),
        ("Conditional\nVector Field", 0.54, 0.40, 0.18, 0.24, "#fae8ff"),
        ("ODE\nIntegration", 0.78, 0.40, 0.12, 0.24, "#fee2e2"),
        ("Generated\nECG", 0.94, 0.40, 0.05, 0.24, "#e0f2fe"),
    ]
    for text, x, y, w, h, color in blocks:
        ax.add_patch(Rectangle((x, y), w, h, linewidth=1.2, edgecolor="#111827", facecolor=color))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    arrows = [
        ((0.21, 0.69), (0.28, 0.69)),
        ((0.44, 0.69), (0.54, 0.55)),
        ((0.21, 0.31), (0.54, 0.46)),
        ((0.44, 0.31), (0.54, 0.43)),
        ((0.72, 0.52), (0.78, 0.52)),
        ((0.90, 0.52), (0.94, 0.52)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.3))

    ax.text(
        0.54,
        0.18,
        "Region-aware loss: mean((1 + lambda * mask) * ||v_pred - v_target||^2)",
        fontsize=11,
        ha="left",
        va="center",
    )
    fig.savefig(OUT_DIR / "RCFM_Architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def synthetic_ecg(length: int = 512) -> np.ndarray:
    x = np.linspace(0, 4, length)
    signal = 0.05 * np.sin(2 * np.pi * 1.2 * x)
    for center in [0.7, 1.6, 2.5, 3.35]:
        signal += 1.0 * np.exp(-((x - center) / 0.025) ** 2)
        signal -= 0.25 * np.exp(-((x - center + 0.035) / 0.018) ** 2)
        signal += 0.18 * np.exp(-((x - center - 0.18) / 0.07) ** 2)
    return signal


def draw_gradcam_examples() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(31)
    ecg = synthetic_ecg()
    x = np.arange(ecg.size)
    importance = np.zeros_like(ecg)
    for peak in [90, 205, 320, 430]:
        importance += np.exp(-((x - peak) / 28) ** 2)
    importance = importance / importance.max()
    noisy = ecg + rng.normal(scale=0.04 + 0.20 * importance, size=ecg.shape)

    for name, signal, title in [
        ("Gradv2.png", ecg, "Grad-CAM importance on target ECG"),
        ("GradNoisev2.png", noisy, "Region-weighted perturbation"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 2.2))
        ax.imshow(
            importance.reshape(1, -1),
            extent=[0, ecg.size - 1, signal.min() - 0.15, signal.max() + 0.15],
            cmap="autumn_r",
            aspect="auto",
            alpha=0.35,
        )
        ax.plot(x, signal, color="#111827", linewidth=1.1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.2)
        fig.savefig(OUT_DIR / name, dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    draw_architecture()
    draw_gradcam_examples()
