"""Build compact phase-agreement figures with fixed PTB-XL/MIMIC-AFib/mmECG slots."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _sha256


DATASETS = (
    ("ptbxl", "PTB-XL"),
    ("mimic_afib", "MIMIC-AFib"),
    ("mmecg", "mmECG"),
)
MODELS = ("cfm", "rcfm", "rcfm_ot", "rddm")
MODEL_LABELS = {"cfm": "CFM", "rcfm": "RCFM", "rcfm_ot": "RCFM-OT", "rddm": "RDDM"}
MODEL_COLORS = {
    "cfm": "#2878b5",
    "rcfm": "#2f8f5b",
    "rcfm_ot": "#c43d4b",
    "rddm": "#d17a00",
}
INTERVALS = ("rr_ms", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms")
INTERVAL_LABELS = ("RR", "PR", "QRS", "QT", "QTc")
PHASE_MODES = ("unshifted", "oracle_aligned")


def _load_dataset(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    waveform = payload.get("waveform_agreement")
    parameter_rows = payload.get("parameter_agreement")
    if not isinstance(waveform, dict) or not isinstance(parameter_rows, list):
        raise ValueError(f"invalid phase-clinical summary schema: {path}")
    if set(waveform) != set(MODELS):
        raise ValueError(f"phase summary must contain exactly {MODELS}")
    for model in MODELS:
        if set(waveform[model]) != set(PHASE_MODES):
            raise ValueError(f"{model} must contain unshifted and oracle_aligned waveform results")
    row_map: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in parameter_rows:
        key = (str(row["model"]), str(row["phase_mode"]), str(row["parameter"]))
        if key in row_map:
            raise ValueError(f"duplicate parameter row: {key}")
        row_map[key] = row
    required = {
        (model, phase_mode, parameter)
        for model in MODELS
        for phase_mode in PHASE_MODES
        for parameter in INTERVALS
    }
    if missing := required - set(row_map):
        raise ValueError(f"phase summary is missing parameter rows: {sorted(missing)}")
    for model in MODELS:
        for parameter in INTERVALS:
            before = row_map[(model, "unshifted", parameter)]
            after = row_map[(model, "oracle_aligned", parameter)]
            if before.get("status") != "ok" or after.get("status") != "ok":
                raise ValueError(f"insufficient {model}/{parameter} agreement")
            if int(before["n"]) != int(after["n"]):
                raise ValueError(f"before/after rows do not use matched {model}/{parameter} records")
    return {"waveform": waveform, "parameters": row_map, "path": path.resolve()}


def _interval_arrays(dataset: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before = np.empty((len(MODELS), len(INTERVALS)), dtype=np.float64)
    after = np.empty_like(before)
    rows = dataset["parameters"]
    for model_index, model in enumerate(MODELS):
        for parameter_index, parameter in enumerate(INTERVALS):
            before[model_index, parameter_index] = float(
                rows[(model, "unshifted", parameter)]["mae"]
            )
            after[model_index, parameter_index] = float(
                rows[(model, "oracle_aligned", parameter)]["mae"]
            )
    change = 100.0 * (after - before) / before
    return before, after, change


def _placeholder(
    axis: plt.Axes,
    title: str | None = None,
    message: str | None = None,
    title_fontsize: float = 10,
    message_fontsize: float = 9,
) -> None:
    axis.set_facecolor("#f4f5f7")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#c7cbd1")
        spine.set_linestyle("--")
    if title:
        axis.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=6)
    if message:
        axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            color="#656d78",
            fontsize=message_fontsize,
            transform=axis.transAxes,
        )


def _dumbbell(
    axis: plt.Axes,
    before: np.ndarray,
    after: np.ndarray,
    xlabel: str,
    xlim: tuple[float, float] | None = None,
    fontsize: float = 8,
    marker_scale: float = 1.0,
) -> None:
    y = np.arange(len(MODELS))
    for index, model in enumerate(MODELS):
        color = MODEL_COLORS[model]
        axis.plot([before[index], after[index]], [y[index], y[index]], color="#a8adb5", linewidth=1.4)
        axis.scatter(before[index], y[index], s=34 * marker_scale, facecolor="white", edgecolor=color, linewidth=1.1, zorder=3)
        axis.scatter(after[index], y[index], s=38 * marker_scale, marker="D", color=color, edgecolor="white", linewidth=0.5, zorder=4)
    axis.set_yticks(y, [MODEL_LABELS[model] for model in MODELS])
    axis.invert_yaxis()
    axis.set_xlabel(xlabel, fontsize=fontsize)
    axis.tick_params(labelsize=fontsize)
    axis.grid(axis="x", alpha=0.22)
    if xlim is not None:
        axis.set_xlim(*xlim)


def _draw_dataset(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    display_name: str,
    dataset: dict[str, object] | None,
    style: dict[str, float],
) -> matplotlib.image.AxesImage | None:
    correlation_axis, ba_axis, heatmap_axis = axes
    if dataset is None:
        _placeholder(
            correlation_axis,
            title=f"{display_name}\nWaveform correlation",
            title_fontsize=style["title"],
        )
        _placeholder(
            ba_axis,
            title="Waveform Bland-Altman",
            message="Reserved\nmatched phase\nanalysis",
            title_fontsize=style["title"],
            message_fontsize=style["placeholder"],
        )
        _placeholder(
            heatmap_axis,
            title="Clinical interval MAE",
            title_fontsize=style["title"],
        )
        return None

    waveform = dataset["waveform"]
    correlation_before = np.asarray(
        [waveform[model]["unshifted"]["per_record_pearson_median"] for model in MODELS]
    )
    correlation_after = np.asarray(
        [waveform[model]["oracle_aligned"]["per_record_pearson_median"] for model in MODELS]
    )
    _dumbbell(
        correlation_axis,
        correlation_before,
        correlation_after,
        "Median per-record Pearson",
        xlim=(-0.1, 1.0),
        fontsize=style["axis"],
        marker_scale=style["marker"],
    )
    correlation_axis.axvline(0.0, color="#6d737c", linewidth=0.7)
    correlation_axis.set_title(
        f"{display_name}\nWaveform correlation",
        fontsize=style["title"],
        fontweight="bold",
        pad=6,
    )

    ba_before = []
    ba_after = []
    for model in MODELS:
        for phase_mode, output in (("unshifted", ba_before), ("oracle_aligned", ba_after)):
            agreement = waveform[model][phase_mode]["pointwise_bland_altman"]
            output.append(float(agreement["upper_limit"]) - float(agreement["lower_limit"]))
    _dumbbell(
        ba_axis,
        np.asarray(ba_before),
        np.asarray(ba_after),
        "95% LoA width (normalized)",
        fontsize=style["axis"],
        marker_scale=style["marker"],
    )
    ba_axis.set_title(
        "Waveform Bland-Altman",
        fontsize=style["title"],
        fontweight="bold",
        pad=6,
    )

    before, after, change = _interval_arrays(dataset)
    image = heatmap_axis.imshow(
        change,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0),
        aspect="auto",
    )
    heatmap_axis.set_xticks(np.arange(len(INTERVALS)), INTERVAL_LABELS)
    heatmap_axis.set_yticks(
        np.arange(len(MODELS)), [MODEL_LABELS[model] for model in MODELS]
    )
    heatmap_axis.tick_params(labelsize=style["axis"])
    for row in range(len(MODELS)):
        for column in range(len(INTERVALS)):
            heatmap_axis.text(
                column,
                row,
                f"{before[row, column]:.1f}\n{after[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=style["heat"],
                color="#111827",
            )
    heatmap_axis.set_title(
        "Clinical interval MAE (ms)\nraw / aligned",
        fontsize=style["title"],
        fontweight="bold",
        pad=6,
    )
    return image


def _plot(
    datasets: list[tuple[str, str, dict[str, object] | None]],
    output_stem: Path,
) -> list[Path]:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Times New Roman", "Times"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    columns = len(datasets)
    style = (
        {"title": 7.0, "axis": 5.8, "heat": 4.4, "placeholder": 6.0, "marker": 0.72}
        if columns == 3
        else {"title": 8.5, "axis": 7.0, "heat": 6.0, "placeholder": 7.0, "marker": 0.9}
    )
    figure = plt.figure(figsize=(7.2 if columns == 3 else 3.5, 5.8))
    grid = figure.add_gridspec(
        3,
        columns,
        height_ratios=(1.0, 1.0, 1.35),
        hspace=0.95 if columns == 3 else 0.78,
        wspace=0.52,
    )
    heatmap_image = None
    for column, (_, display_name, dataset) in enumerate(datasets):
        axes = tuple(figure.add_subplot(grid[row, column]) for row in range(3))
        current_image = _draw_dataset(axes, display_name, dataset, style)
        if current_image is not None:
            heatmap_image = current_image
    legend = [
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#4b5563", color="none", label="Raw"),
        Line2D([0], [0], marker="D", markerfacecolor="#4b5563", markeredgecolor="white", color="none", label="Oracle aligned"),
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=6 if columns == 3 else 7,
        bbox_to_anchor=(0.5, 0.995),
    )
    if heatmap_image is not None:
        color_axis = figure.add_axes([0.33, 0.025, 0.34, 0.018])
        colorbar = figure.colorbar(heatmap_image, cax=color_axis, orientation="horizontal")
        colorbar.set_label(
            "MAE change after alignment (%)  |  blue: lower",
            fontsize=6 if columns == 3 else 7,
        )
        colorbar.ax.tick_params(labelsize=5.5 if columns == 3 else 6.5)
    figure.subplots_adjust(top=0.84, bottom=0.10)
    outputs = []
    for suffix in ("png", "pdf"):
        path = output_stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ptbxl": args.ptbxl_summary,
        "mimic_afib": args.mimic_summary,
        "mmecg": args.mmecg_summary,
    }
    loaded = {
        key: _load_dataset(path) if path is not None else None
        for key, path in paths.items()
    }
    if loaded["mimic_afib"] is None:
        raise ValueError("--mimic_summary is required for the current figure")
    template_datasets = [
        (key, display, loaded[key]) for key, display in DATASETS
    ]
    outputs = _plot(
        template_datasets,
        output_dir / "phase_agreement_three_dataset_template",
    )
    outputs.extend(
        _plot(
            [("mimic_afib", "MIMIC-AFib", loaded["mimic_afib"])],
            output_dir / "mimic_afib_phase_agreement_compact",
        )
    )
    if loaded["mmecg"] is not None:
        outputs.extend(
            _plot(
                [("mmecg", "mmECG", loaded["mmecg"])],
                output_dir / "mmecg_phase_agreement_compact",
            )
        )
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "dataset_order": [key for key, _ in DATASETS],
        "dataset_status": {
            key: "populated" if loaded[key] is not None else "reserved_pending"
            for key, _ in DATASETS
        },
        "inputs": {
            key: None
            if path is None
            else {"path": str(path.resolve()), "sha256": _sha256(path)}
            for key, path in paths.items()
        },
        "figure_encoding": {
            "top": "median per-record waveform Pearson, raw circle to oracle-aligned diamond",
            "middle": "pointwise waveform Bland-Altman 95% limits-of-agreement width",
            "bottom": "interval MAE in ms, raw/aligned text, color is relative MAE change",
            "heatmap_color_range_percent": [-3.0, 3.0],
        },
        "claim_boundary": (
            "Missing dataset columns are layout reservations, not results. Oracle-aligned values "
            "are target-informed diagnostics and cannot replace raw primary metrics."
        ),
        "execution": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "script_sha256": _sha256(Path(__file__)),
        },
        "outputs": [path.name for path in outputs],
        "artifact_sha256": {path.name: _sha256(path) for path in outputs},
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic_summary", type=Path, required=True)
    parser.add_argument("--ptbxl_summary", type=Path, default=None)
    parser.add_argument("--mmecg_summary", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    result = run(build_argparser().parse_args())
    print(f"phase-agreement figures saved to {result}")
