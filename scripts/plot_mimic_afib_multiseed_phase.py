"""Plot frozen MIMIC-AFib three-seed oracle-aligned waveform results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("CFM", "RCFM-Pan", "RCFM-Pan+OT", "RDDM")
METRICS = (
    ("rmse", "RMSE", "lower is better"),
    ("mae", "MAE", "lower is better"),
    ("waveform_fd", "Waveform FD", "lower is better"),
    ("pearson_window_median", "Median window Pearson", "higher is better"),
)
COLORS = {
    "CFM": "#2878B5",
    "RCFM-Pan": "#2F8F5B",
    "RCFM-Pan+OT": "#C43D4B",
    "RDDM": "#D17A00",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_phase_rows(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    values: dict[str, dict[int, dict[str, float]]] = {model: {} for model in MODELS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["phase_mode"] != "oracle_aligned_fixed_support":
                continue
            model = row["model"]
            seed = int(row["seed"])
            if model not in values or seed not in {31, 32, 33} or seed in values[model]:
                raise ValueError("invalid model/seed row in phase metric artifact")
            values[model][seed] = {
                metric: float(row[metric]) for metric, _, _ in METRICS
            }
    if any(set(seeds) != {31, 32, 33} for seeds in values.values()):
        raise ValueError("phase figure requires exactly seeds 31, 32, and 33 per model")
    return values


def run(args: argparse.Namespace) -> Path:
    source = args.per_seed_metrics.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    values = _read_phase_rows(source)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Times New Roman", "Times"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 5.0), constrained_layout=True)
    x = np.arange(len(MODELS), dtype=np.float64)
    offsets = {31: -0.12, 32: 0.0, 33: 0.12}
    markers = {31: "o", 32: "s", 33: "^"}
    for axis, (metric, title, direction) in zip(axes.ravel(), METRICS):
        for model_index, model in enumerate(MODELS):
            samples = np.asarray(
                [values[model][seed][metric] for seed in (31, 32, 33)],
                dtype=np.float64,
            )
            for seed, sample in zip((31, 32, 33), samples):
                axis.scatter(
                    model_index + offsets[seed],
                    sample,
                    s=24,
                    marker=markers[seed],
                    facecolor="white",
                    edgecolor=COLORS[model],
                    linewidth=1.0,
                    zorder=3,
                    label=f"Seed {seed}" if model_index == 0 else None,
                )
            axis.errorbar(
                model_index,
                float(samples.mean()),
                yerr=float(samples.std(ddof=1)),
                fmt="D",
                markersize=4.5,
                color=COLORS[model],
                ecolor=COLORS[model],
                capsize=3,
                linewidth=1.4,
                zorder=4,
            )
        axis.set_title(f"{title} ({direction})", fontsize=10)
        axis.set_xticks(x, ("CFM", "RCFM\nPan", "RCFM\nPan+OT", "RDDM"))
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8, loc="best")
    figure.suptitle(
        "MIMIC-AFib target-informed phase-corrected morphology: three training seeds",
        fontsize=11,
    )
    pdf_path = output_dir / "mimic_afib_multiseed_phase_corrected.pdf"
    png_path = output_dir / "mimic_afib_multiseed_phase_corrected.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(figure)

    protocol = {
        "status": "completed",
        "source": {"path": str(source), "sha256": _sha256(source)},
        "phase_mode": "oracle_aligned_fixed_support",
        "seeds": [31, 32, 33],
        "display": "individual training seeds plus mean and sample SD",
        "claim_boundary": (
            "Each lag was selected using its held-out ECG target. The figure is an oracle "
            "morphology diagnostic, not deployable alignment or evidence of significance."
        ),
        "outputs": {
            pdf_path.name: _sha256(pdf_path),
            png_path.name: _sha256(png_path),
        },
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per_seed_metrics", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
