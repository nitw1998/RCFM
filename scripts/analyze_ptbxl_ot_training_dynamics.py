"""Analyze PTB-XL validation dynamics and exact-OT coupling diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEEDS = (31, 32, 33)
VARIANTS = ("cfm", "cfm_ot", "diag", "diag_ot")
OT_VARIANTS = ("cfm_ot", "diag_ot")
VALIDATION_EPOCHS = tuple(range(25, 501, 25))
THRESHOLDS = (0.38, 0.36, 0.35, 0.34)

RUNS = {
    ("cfm", 31): "runs/training/ecg2ecg/PTBXL/ptbxl_cfm_minmax_neg1_1_seed31",
    ("cfm_ot", 31): (
        "runs/training/cfm_ot_five_dataset_v1/ecg2ecg/PTBXL/"
        "ptbxl-cfm-ot-s31-20260814T155707Z"
    ),
    ("diag", 31): (
        "runs/training/ptbxl_diagmask_factorial_v1/ecg2ecg/PTBXL/"
        "ptbxl_rcfm_diagmask_no_ot_s31_v1"
    ),
    ("diag_ot", 31): (
        "runs/training/ptbxl_diagmask_factorial_v1/ecg2ecg/PTBXL/"
        "ptbxl_rcfm_diagmask_exact_ot_s31_v1"
    ),
}
for _seed in (32, 33):
    RUNS[("cfm", _seed)] = (
        "runs/training/ptbxl_factorial_multiseed_v1/ecg2ecg/PTBXL/"
        f"ptbxl_cfm_no_ot_s{_seed}_v1"
    )
    RUNS[("cfm_ot", _seed)] = (
        "runs/training/ptbxl_factorial_multiseed_v1/ecg2ecg/PTBXL/"
        f"ptbxl_cfm_exact_ot_s{_seed}_v1"
    )
    RUNS[("diag", _seed)] = (
        "runs/training/ptbxl_factorial_multiseed_v1/ecg2ecg/PTBXL/"
        f"ptbxl_diagmask_no_ot_s{_seed}_v1"
    )
    RUNS[("diag_ot", _seed)] = (
        "runs/training/ptbxl_factorial_multiseed_v1/ecg2ecg/PTBXL/"
        f"ptbxl_diagmask_exact_ot_s{_seed}_v1"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_long_metric(path: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    epochs: list[int] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] == metric:
                epochs.append(int(row["epoch"]))
                values.append(float(row["value"]))
    if not values or len(set(epochs)) != len(epochs):
        raise ValueError(f"missing or duplicate {metric} rows in {path}")
    order = np.argsort(epochs)
    return np.asarray(epochs, dtype=int)[order], np.asarray(values, dtype=float)[order]


def _first_epoch_at_or_below(
    epochs: np.ndarray, values: np.ndarray, threshold: float
) -> int | None:
    selected = epochs[values <= threshold]
    return int(selected[0]) if selected.size else None


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_config(
    path: Path, variant: str, seed: int, reference_contract: dict[str, object] | None
) -> tuple[dict[str, object], dict[str, object]]:
    config = json.loads((path / "resolved_config.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
    expected = {
        "seed": seed,
        "epochs": 500,
        "batch_size": 128,
        "validation_interval_epochs": 25,
        "validation_fixed_noise_seed": 2025,
        "signal_length": 512,
        "output_channels": 11,
        "normalization_id": "record_minmax_neg1_1_v1",
        "dataset_version": "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1",
        "use_minibatch_ot": variant in OT_VARIANTS,
        # The frozen no-OT configs retain the dormant solver choice; the enable flag
        # is the experiment-defining switch.
        "ot_method": "exact",
        "region_weight": 0.01 if variant.startswith("diag") else 0.0,
    }
    bad = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if bad:
        raise ValueError(f"configuration mismatch for {variant} seed {seed}: {bad}")
    if metadata.get("status") != "completed":
        raise ValueError(f"run is not completed: {path}")
    contract = {
        key: config[key]
        for key in (
            "dataset_version",
            "split_hash",
            "normalization_id",
            "condition_lead",
            "target_leads",
            "batch_size",
            "epochs",
            "validation_fixed_noise_seed",
        )
    }
    if reference_contract is not None and contract != reference_contract:
        raise ValueError(f"shared training contract mismatch for {variant} seed {seed}")
    return metadata, contract


def _plot(
    validation: dict[str, np.ndarray],
    ot_reduction: dict[str, np.ndarray],
    output_pdf: Path,
    output_png: Path,
) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman"],
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
    colors = {"no_ot": "#0072B2", "ot": "#D55E00"}
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25), constrained_layout=True)
    panels = (
        ("cfm", "cfm_ot", "(a) CFM validation"),
        ("diag", "diag_ot", "(b) DiagMask validation"),
    )
    epochs = np.asarray(VALIDATION_EPOCHS)
    for axis, (no_ot, ot, title) in zip(axes[:2], panels):
        for key, label, color_key, marker in (
            (no_ot, "No OT", "no_ot", "o"),
            (ot, "Exact OT", "ot", "s"),
        ):
            values = validation[key]
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1)
            axis.plot(
                epochs,
                mean,
                color=colors[color_key],
                linewidth=1.25,
                marker=marker,
                markersize=2.5,
                markevery=2,
                label=label,
            )
            axis.fill_between(epochs, mean - sd, mean + sd, color=colors[color_key], alpha=0.15)
        axis.set_title(title, loc="left")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Fold-9 validation RMSE")
        axis.set_xlim(25, 500)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)

    axis = axes[2]
    if not np.array_equal(ot_reduction["cfm_ot"], ot_reduction["diag_ot"]):
        raise ValueError("expected shared-seed OT coupling diagnostics to be identical")
    ot_styles = (("cfm_ot", "Both exact-OT cells", "#009E73", "o"),)
    ot_epochs = np.asarray(VALIDATION_EPOCHS)
    for key, label, color, marker in ot_styles:
        # Match the 25-epoch reporting cadence of validation and suppress only
        # within-bin logging noise; the unbinned epoch values remain in CSV.
        values = 100.0 * ot_reduction[key].reshape(len(SEEDS), len(VALIDATION_EPOCHS), 25).mean(axis=2)
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        axis.plot(
            ot_epochs,
            mean,
            color=color,
            linewidth=1.15,
            marker=marker,
            markersize=2.3,
            markevery=50,
            label=label,
        )
        axis.fill_between(ot_epochs, mean - sd, mean + sd, color=color, alpha=0.15)
    axis.set_title("(c) Exact-OT coupling diagnostic", loc="left")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Endpoint cost reduction (%)")
    axis.set_xlim(1, 500)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    workspace = args.workspace_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    validation: dict[str, list[np.ndarray]] = {variant: [] for variant in VARIANTS}
    ot_reduction: dict[str, list[np.ndarray]] = {variant: [] for variant in OT_VARIANTS}
    convergence_rows: list[dict[str, object]] = []
    input_files: list[dict[str, object]] = []
    reference_contract: dict[str, object] | None = None

    for variant in VARIANTS:
        for seed in SEEDS:
            run_dir = workspace / RUNS[(variant, seed)]
            metadata, contract = _validate_config(run_dir, variant, seed, reference_contract)
            reference_contract = contract
            validation_path = run_dir / "validation_metrics.csv"
            epochs, rmse = _read_long_metric(validation_path, "val/rmse")
            if tuple(epochs) != VALIDATION_EPOCHS or not np.all(np.isfinite(rmse)):
                raise ValueError(f"validation epoch grid/finiteness mismatch: {run_dir}")
            validation[variant].append(rmse)
            best_index = int(np.argmin(rmse))
            row: dict[str, object] = {
                "variant": variant,
                "seed": seed,
                "best_validation_rmse": float(rmse[best_index]),
                "best_epoch": int(epochs[best_index]),
                "early_epoch_25_250_mean_rmse": float(rmse[epochs <= 250].mean()),
                "late_epoch_275_500_mean_rmse": float(rmse[epochs >= 275].mean()),
                "training_duration_hours_observed": float(metadata["training_duration_seconds"]) / 3600.0,
            }
            for threshold in THRESHOLDS:
                row[f"first_epoch_rmse_le_{threshold:.2f}"] = _first_epoch_at_or_below(
                    epochs, rmse, threshold
                )
            convergence_rows.append(row)
            for name in ("resolved_config.yaml", "run_metadata.json", "validation_metrics.csv"):
                path = run_dir / name
                input_files.append({"path": str(path), "sha256": _sha256(path)})

            diagnostics_path = run_dir / "ot_diagnostics.csv"
            input_files.append({"path": str(diagnostics_path), "sha256": _sha256(diagnostics_path)})
            if variant in OT_VARIANTS:
                diag_epochs, reduction = _read_long_metric(
                    diagnostics_path, "ot/cost_reduction_ratio"
                )
                if tuple(diag_epochs) != tuple(range(1, 501)):
                    raise ValueError(f"OT diagnostic epoch grid mismatch: {run_dir}")
                ot_reduction[variant].append(reduction)
                for metric, expected in (
                    ("ot/unique_target_fraction", 1.0),
                    ("ot/fallback_count", 0.0),
                    ("ot/nonfinite_plan_count", 0.0),
                ):
                    metric_epochs, values = _read_long_metric(diagnostics_path, metric)
                    if tuple(metric_epochs) != tuple(diag_epochs) or not np.allclose(values, expected):
                        raise ValueError(f"invalid {metric} diagnostics: {run_dir}")

    validation_arrays = {key: np.stack(value) for key, value in validation.items()}
    ot_arrays = {key: np.stack(value) for key, value in ot_reduction.items()}
    _write_csv(output / "per_seed_convergence.csv", convergence_rows)

    curve_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        for index, epoch in enumerate(VALIDATION_EPOCHS):
            values = validation_arrays[variant][:, index]
            curve_rows.append(
                {
                    "variant": variant,
                    "epoch": epoch,
                    "seed_count": len(SEEDS),
                    "validation_rmse_mean": float(values.mean()),
                    "validation_rmse_sample_sd": float(values.std(ddof=1)),
                }
            )
    _write_csv(output / "validation_curve_summary.csv", curve_rows)

    ot_rows: list[dict[str, object]] = []
    for variant in OT_VARIANTS:
        values = ot_arrays[variant]
        for index, epoch in enumerate(range(1, 501)):
            epoch_values = values[:, index]
            ot_rows.append(
                {
                    "variant": variant,
                    "epoch": epoch,
                    "seed_count": len(SEEDS),
                    "cost_reduction_ratio_mean": float(epoch_values.mean()),
                    "cost_reduction_ratio_sample_sd": float(epoch_values.std(ddof=1)),
                    "unique_target_fraction": 1.0,
                    "fallback_count": 0,
                    "nonfinite_plan_count": 0,
                }
            )
    _write_csv(output / "ot_diagnostics_summary.csv", ot_rows)

    contrast_rows: list[dict[str, object]] = []
    for no_ot, ot, family in (("cfm", "cfm_ot", "CFM"), ("diag", "diag_ot", "DiagMask")):
        no_rows = {int(row["seed"]): row for row in convergence_rows if row["variant"] == no_ot}
        ot_rows_by_seed = {int(row["seed"]): row for row in convergence_rows if row["variant"] == ot}
        for seed in SEEDS:
            base = no_rows[seed]
            comparison = ot_rows_by_seed[seed]
            contrast = {
                    "family": family,
                    "seed": seed,
                    "comparison": "exact_ot_minus_no_ot",
                    "best_validation_rmse_difference": float(comparison["best_validation_rmse"])
                    - float(base["best_validation_rmse"]),
                    "best_epoch_difference": int(comparison["best_epoch"]) - int(base["best_epoch"]),
                    "early_mean_rmse_difference": float(comparison["early_epoch_25_250_mean_rmse"])
                    - float(base["early_epoch_25_250_mean_rmse"]),
                    "late_mean_rmse_difference": float(comparison["late_epoch_275_500_mean_rmse"])
                    - float(base["late_epoch_275_500_mean_rmse"]),
                    "observed_duration_hours_difference": float(
                        comparison["training_duration_hours_observed"]
                    )
                    - float(base["training_duration_hours_observed"]),
                }
            for threshold in THRESHOLDS:
                key = f"first_epoch_rmse_le_{threshold:.2f}"
                if base[key] is None or comparison[key] is None:
                    contrast[f"first_epoch_difference_rmse_le_{threshold:.2f}"] = None
                else:
                    contrast[f"first_epoch_difference_rmse_le_{threshold:.2f}"] = int(
                        comparison[key]
                    ) - int(base[key])
            contrast_rows.append(contrast)
    _write_csv(output / "paired_convergence_contrasts.csv", contrast_rows)

    aggregate: dict[str, object] = {
        "difference_definition": "exact_ot_minus_no_ot; negative RMSE is better; negative epoch is earlier",
        "families": {},
        "ot_mechanism": {},
    }
    for family in ("CFM", "DiagMask"):
        rows = [row for row in contrast_rows if row["family"] == family]
        numeric_keys = (
            "best_validation_rmse_difference",
            "best_epoch_difference",
            "early_mean_rmse_difference",
            "late_mean_rmse_difference",
            "observed_duration_hours_difference",
            *(f"first_epoch_difference_rmse_le_{threshold:.2f}" for threshold in THRESHOLDS),
        )
        aggregate["families"][family] = {
            key: float(np.mean([float(row[key]) for row in rows if row[key] is not None]))
            if any(row[key] is not None for row in rows)
            else None
            for key in numeric_keys
        }
    for variant in OT_VARIANTS:
        per_seed_means = ot_arrays[variant].mean(axis=1)
        aggregate["ot_mechanism"][variant] = {
            "cost_reduction_ratio_mean": float(per_seed_means.mean()),
            "cost_reduction_ratio_between_seed_sample_sd": float(per_seed_means.std(ddof=1)),
            "per_seed_means": [float(value) for value in per_seed_means],
            "unique_target_fraction": 1.0,
            "fallback_count": 0,
            "nonfinite_plan_count": 0,
        }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")

    figure_pdf = output / "ptbxl_ot_training_dynamics.pdf"
    figure_png = output / "ptbxl_ot_training_dynamics.png"
    _plot(validation_arrays, ot_arrays, figure_pdf, figure_png)
    if args.paper_figure is not None:
        args.paper_figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(figure_pdf, args.paper_figure)

    protocol = {
        "schema_version": 1,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "PTB-XL three-seed validation dynamics and exact-OT diagnostics",
        "training_seeds": list(SEEDS),
        "validation_epochs": list(VALIDATION_EPOCHS),
        "predeclared_rmse_thresholds": list(THRESHOLDS),
        "shared_contract": reference_contract,
        "difference_definition": "exact_ot_minus_no_ot",
        "wall_time_caveat": (
            "Observed wall time is descriptive only because GPU model, contention, and system load "
            "were not matched; it is not an algorithmic speed comparison."
        ),
        "training_loss_caveat": (
            "Training velocity losses are not used for the primary convergence comparison because "
            "OT changes the endpoint coupling and therefore the sampled regression targets."
        ),
        "inputs": input_files,
        "outputs": {
            path.name: {"sha256": _sha256(path)}
            for path in (
                output / "per_seed_convergence.csv",
                output / "paired_convergence_contrasts.csv",
                output / "validation_curve_summary.csv",
                output / "ot_diagnostics_summary.csv",
                summary_path,
                figure_pdf,
                figure_png,
            )
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/evaluation/ptbxl_ot_training_dynamics_v1"),
    )
    parser.add_argument("--paper-figure", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
