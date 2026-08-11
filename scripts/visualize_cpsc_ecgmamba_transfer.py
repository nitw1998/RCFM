"""Visualize ECGMambaFormer NORM Grad-CAM and semantic masks on fixed CPSC records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import torch
from scipy.io import loadmat
from scipy.special import expit


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocess_cpsc2018 import discover_record_paths, load_reference
from src.rcfm.interpretability.ecgmambaformer_compat import (
    diagnostic_class_names,
    load_ecgmambaformer_fca_mgda,
    record_global_zscore,
)
from src.rcfm.interpretability.gradcam import gradcam_1d, normalize_soft_mask


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SEMANTIC_NAMES = ["background", "P", "QRS", "T"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _load_fixed_records(
    selection_csv: Path,
    cpsc_root: Path,
    source_rate_hz: int,
    window_seconds: int,
) -> list[dict[str, object]]:
    with selection_csv.open(newline="", encoding="utf-8") as handle:
        selected_rows = list(csv.DictReader(handle))
    if not selected_rows or {row["group"] for row in selected_rows} != {"AF", "non-AF"}:
        raise ValueError("selection CSV must contain AF and non-AF groups")
    reference = {str(row["record_id"]): row for row in load_reference(cpsc_root / "REFERENCE.csv")}
    paths = discover_record_paths(cpsc_root)
    records: list[dict[str, object]] = []
    required_samples = source_rate_hz * window_seconds
    for selected in selected_rows:
        record_id = selected["record_id"]
        if record_id not in reference or record_id not in paths:
            raise ValueError(f"selected record is unavailable: {record_id}")
        labels = tuple(int(value) for value in reference[record_id]["labels"])
        group = selected["group"]
        if (group == "AF" and 2 not in labels) or (group == "non-AF" and labels != (1,)):
            raise ValueError(f"selected group disagrees with CPSC labels for {record_id}")
        payload = loadmat(
            paths[record_id],
            squeeze_me=True,
            struct_as_record=False,
            variable_names=["ECG"],
        )
        if "ECG" not in payload or not hasattr(payload["ECG"], "data"):
            raise ValueError(f"missing ECG.data for {record_id}")
        signal = np.asarray(payload["ECG"].data, dtype=np.float64)
        if signal.ndim != 2 or signal.shape[0] != 12 or signal.shape[1] < required_samples:
            raise ValueError(f"invalid 12-lead 10-second record: {record_id}")
        waveform = signal[:, :required_samples].T.astype(np.float32)
        if not np.all(np.isfinite(waveform)) or np.any(np.std(waveform, axis=0) < 1e-6):
            raise ValueError(f"selected record fails finite/all-lead variance QC: {record_id}")
        normalized, mean, scale = record_global_zscore(waveform)
        records.append(
            {
                "group": group,
                "sample_index": int(selected["sample_index"]),
                "record_id": record_id,
                "labels": labels,
                "waveform": waveform,
                "normalized": normalized,
                "normalization_mean": mean,
                "normalization_scale": scale,
            }
        )
    return records


def _infer_records(
    model: torch.nn.Module,
    records: list[dict[str, object]],
    target_index: int,
    classes: list[str],
    device: torch.device,
) -> None:
    for record in records:
        values = np.asarray(record["normalized"]).T.copy()
        inputs = torch.from_numpy(values).unsqueeze(0).to(device=device, dtype=torch.float32)
        raw_cam, logits = gradcam_1d(model, model.encoder, inputs, target_index)
        mask, degenerate = normalize_soft_mask(raw_cam)
        with torch.no_grad():
            semantic = model.semantic_probabilities(inputs)[0].detach().cpu().numpy()
        if semantic.shape != (4, values.shape[-1]) or not np.all(np.isfinite(semantic)):
            raise ValueError("semantic decoder returned an invalid array")
        probabilities = expit(logits).astype(np.float32)
        semantic_labels = np.argmax(semantic, axis=0).astype(np.uint8)
        occupancy = np.bincount(semantic_labels, minlength=4) / len(semantic_labels)
        top_index = int(np.argmax(probabilities))
        record.update(
            {
                "norm_cam": mask,
                "raw_norm_cam": raw_cam,
                "cam_degenerate": degenerate,
                "diagnostic_logits": logits,
                "diagnostic_probabilities": probabilities,
                "norm_probability": float(probabilities[target_index]),
                "top_class_index": top_index,
                "top_class": classes[top_index],
                "semantic_probabilities": semantic.astype(np.float32),
                "semantic_labels": semantic_labels,
                "semantic_occupancy": occupancy.astype(np.float32),
            }
        )


def _panel_limits(waveform: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(waveform, [0.01, 0.99])
    margin = max(float(upper - lower) * 0.18, 0.05)
    return float(lower - margin), float(upper + margin)


def _plot_norm_gradcam(records: list[dict[str, object]], output_dir: Path, lead_index: int) -> None:
    groups = ["AF", "non-AF"]
    grouped = {group: [record for record in records if record["group"] == group] for group in groups}
    rows = max(map(len, grouped.values()))
    figure, axes = plt.subplots(rows, 2, figsize=(14, 2.7 * rows), squeeze=False)
    image = None
    for column, group in enumerate(groups):
        for row_index, record in enumerate(grouped[group]):
            axis = axes[row_index, column]
            waveform = np.asarray(record["waveform"])[:, lead_index]
            time = np.arange(len(waveform), dtype=np.float32) / 500.0
            y_min, y_max = _panel_limits(waveform)
            image = axis.imshow(
                np.asarray(record["norm_cam"])[np.newaxis, :],
                extent=(0.0, 10.0, y_min, y_max),
                aspect="auto",
                origin="lower",
                cmap="YlOrRd",
                vmin=0.0,
                vmax=1.0,
                alpha=0.62,
                interpolation="bilinear",
            )
            axis.plot(time, waveform, color="#17202a", linewidth=0.75)
            axis.set_xlim(0.0, 10.0)
            axis.set_ylim(y_min, y_max)
            title_group = "AF" if group == "AF" else "Non-AF (normal)"
            axis.set_title(
                f"{title_group} {row_index + 1} | NORM p={record['norm_probability']:.3f} | "
                f"top={record['top_class']}"
            )
            axis.set_ylabel(f"Lead {LEAD_NAMES[lead_index]} (source units)")
            if row_index == rows - 1:
                axis.set_xlabel("Time (s)")
    figure.suptitle(
        "ECGMambaFormer fixed-NORM Grad-CAM: PTB-XL diagnostic -> CPSC2018",
        fontsize=14,
        y=0.995,
    )
    figure.subplots_adjust(left=0.07, right=0.875, top=0.92, bottom=0.08, hspace=0.48, wspace=0.20)
    if image is not None:
        colorbar_axis = figure.add_axes((0.90, 0.22, 0.018, 0.52))
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Per-record normalized NORM-evidence Grad-CAM")
    figure.text(
        0.5,
        0.008,
        "AFIB is absent from this checkpoint's 44-class head; this is NORM evidence, not AFIB Grad-CAM.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(output_dir / "gradcam_norm_af_vs_non_af.png", dpi=240)
    figure.savefig(output_dir / "gradcam_norm_af_vs_non_af.pdf")
    plt.close(figure)


def _plot_semantic(records: list[dict[str, object]], output_dir: Path, lead_index: int) -> None:
    groups = ["AF", "non-AF"]
    grouped = {group: [record for record in records if record["group"] == group] for group in groups}
    rows = max(map(len, grouped.values()))
    cmap = ListedColormap(["#ffffff", "#4c78a8", "#d1495b", "#59a14f"])
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)
    figure, axes = plt.subplots(rows, 2, figsize=(14, 2.7 * rows), squeeze=False)
    image = None
    for column, group in enumerate(groups):
        for row_index, record in enumerate(grouped[group]):
            axis = axes[row_index, column]
            waveform = np.asarray(record["waveform"])[:, lead_index]
            time = np.arange(len(waveform), dtype=np.float32) / 500.0
            y_min, y_max = _panel_limits(waveform)
            image = axis.imshow(
                np.asarray(record["semantic_labels"])[np.newaxis, :],
                extent=(0.0, 10.0, y_min, y_max),
                aspect="auto",
                origin="lower",
                cmap=cmap,
                norm=norm,
                alpha=0.48,
                interpolation="nearest",
            )
            axis.plot(time, waveform, color="#17202a", linewidth=0.75)
            axis.set_xlim(0.0, 10.0)
            axis.set_ylim(y_min, y_max)
            occupancy = np.asarray(record["semantic_occupancy"])
            title_group = "AF" if group == "AF" else "Non-AF (normal)"
            axis.set_title(
                f"{title_group} {row_index + 1} | predicted P/QRS/T occupancy "
                f"{occupancy[1]:.1%}/{occupancy[2]:.1%}/{occupancy[3]:.1%}"
            )
            axis.set_ylabel(f"Lead {LEAD_NAMES[lead_index]} (source units)")
            if row_index == rows - 1:
                axis.set_xlabel("Time (s)")
    figure.suptitle(
        "ECGMambaFormer direct PTB-XL+ semantic transfer -> CPSC2018",
        fontsize=14,
        y=0.995,
    )
    figure.subplots_adjust(left=0.07, right=0.875, top=0.92, bottom=0.08, hspace=0.48, wspace=0.20)
    if image is not None:
        colorbar_axis = figure.add_axes((0.90, 0.12, 0.018, 0.74))
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_ticks(range(4), labels=SEMANTIC_NAMES)
        colorbar.ax.set_title("Class", pad=8)
    figure.text(
        0.5,
        0.008,
        "Direct semantic-decoder output; this panel is not Grad-CAM and has no CPSC fiducial ground truth.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(output_dir / "semantic_p_qrs_t_af_vs_non_af.png", dpi=240)
    figure.savefig(output_dir / "semantic_p_qrs_t_af_vs_non_af.pdf")
    plt.close(figure)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ecgmambaformer_root", type=Path, required=True)
    parser.add_argument("--scp_statements", type=Path, required=True)
    parser.add_argument("--selection_csv", type=Path, required=True)
    parser.add_argument("--cpsc_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source_rate_hz", type=int, default=500)
    parser.add_argument("--window_seconds", type=int, default=10)
    parser.add_argument("--plot_lead", choices=LEAD_NAMES, default="II")
    return parser


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.source_rate_hz != 500 or args.window_seconds != 10:
        raise ValueError("the frozen checkpoint requires 10 seconds at 500 Hz")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)

    classes = diagnostic_class_names(args.scp_statements.resolve())
    if len(classes) != 44 or "AFIB" in classes or "NORM" not in classes:
        raise ValueError("expected the 44-class diagnostic head without AFIB and with NORM")
    target_index = classes.index("NORM")
    model = load_ecgmambaformer_fca_mgda(
        args.ecgmambaformer_root,
        args.checkpoint,
        num_diagnostic_classes=len(classes),
        map_location="cpu",
    )
    model.eval().to(device)
    records = _load_fixed_records(
        args.selection_csv.resolve(),
        args.cpsc_root.resolve(),
        args.source_rate_hz,
        args.window_seconds,
    )
    _infer_records(model, records, target_index, classes, device)
    lead_index = LEAD_NAMES.index(args.plot_lead)
    _plot_norm_gradcam(records, output_dir, lead_index)
    _plot_semantic(records, output_dir, lead_index)

    with (output_dir / "selected_records.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "group", "sample_index", "record_id", "cpsc_labels", "norm_probability",
            "top_class", "cam_degenerate", "background_occupancy", "p_occupancy",
            "qrs_occupancy", "t_occupancy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            occupancy = np.asarray(record["semantic_occupancy"])
            writer.writerow(
                {
                    "group": record["group"],
                    "sample_index": record["sample_index"],
                    "record_id": record["record_id"],
                    "cpsc_labels": ";".join(map(str, record["labels"])),
                    "norm_probability": f"{record['norm_probability']:.9g}",
                    "top_class": record["top_class"],
                    "cam_degenerate": record["cam_degenerate"],
                    "background_occupancy": f"{occupancy[0]:.9g}",
                    "p_occupancy": f"{occupancy[1]:.9g}",
                    "qrs_occupancy": f"{occupancy[2]:.9g}",
                    "t_occupancy": f"{occupancy[3]:.9g}",
                }
            )

    np.savez_compressed(
        output_dir / "ecgmamba_transfer_results.npz",
        record_ids=np.asarray([record["record_id"] for record in records], dtype="U16"),
        groups=np.asarray([record["group"] for record in records], dtype="U8"),
        cpsc_labels=np.asarray([",".join(map(str, record["labels"])) for record in records], dtype="U16"),
        waveforms=np.stack([record["waveform"] for record in records]).astype(np.float32),
        normalized_inputs=np.stack([record["normalized"] for record in records]).astype(np.float32),
        normalization_means=np.asarray([record["normalization_mean"] for record in records]),
        normalization_scales=np.asarray([record["normalization_scale"] for record in records]),
        norm_gradcams=np.stack([record["norm_cam"] for record in records]).astype(np.float32),
        raw_norm_gradcams=np.stack([record["raw_norm_cam"] for record in records]).astype(np.float32),
        diagnostic_logits=np.stack([record["diagnostic_logits"] for record in records]).astype(np.float32),
        diagnostic_probabilities=np.stack([record["diagnostic_probabilities"] for record in records]).astype(np.float32),
        diagnostic_class_names=np.asarray(classes, dtype="U32"),
        semantic_probabilities=np.stack([record["semantic_probabilities"] for record in records]).astype(np.float32),
        semantic_labels=np.stack([record["semantic_labels"] for record in records]).astype(np.uint8),
        semantic_class_names=np.asarray(SEMANTIC_NAMES, dtype="U16"),
    )

    source_root = args.ecgmambaformer_root.resolve()
    artifact_names = [
        "gradcam_norm_af_vs_non_af.png",
        "gradcam_norm_af_vs_non_af.pdf",
        "semantic_p_qrs_t_af_vs_non_af.png",
        "semantic_p_qrs_t_af_vs_non_af.pdf",
        "ecgmamba_transfer_results.npz",
        "selected_records.csv",
    ]
    protocol = {
        "status": "exploratory_unpublished_model",
        "reviewer_scope": "Reviewer 2 Comment 4 cross-domain mask robustness",
        "claim_boundary": (
            "The checkpoint has a 44-class PTB-XL diagnostic head that excludes AFIB. "
            "The Grad-CAM therefore targets NORM for both groups; the P/QRS/T view is direct "
            "semantic output and not Grad-CAM. Neither view is quantitative mask validation."
        ),
        "checkpoint_selection": {
            "path_label": "ecgmamba_FCA_MGDA_s42/best.pt",
            "basis": "author recollection plus final-method directory name",
            "best_performance_verified_from_matching_log": False,
            "matching_log_issue": "available same-name logs terminate with traceback or failed launch",
        },
        "model": {
            "architecture": "ECGMambaImproved + FCA-MGDA multi-task model",
            "training_dataset": "PTB-XL plus local semantic annotations",
            "diagnostic_task": "44 diagnostic SCP statements",
            "diagnostic_classes": classes,
            "AFIB_available": False,
            "gradcam_target": "NORM",
            "gradcam_target_index": target_index,
            "semantic_classes": SEMANTIC_NAMES,
            "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
            "strict_checkpoint_load": True,
            "source_hashes": {
                "ecgmamba_improved.py": _sha256(source_root / "LibMTL/model/ecgmamba_improved.py"),
                "FCA_MGDA.py": _sha256(source_root / "LibMTL/weighting/FCA_MGDA.py"),
                "prepare_ptbxl.py": _sha256(source_root / "scripts/prepare_ptbxl.py"),
                "scp_statements.csv": _sha256(args.scp_statements.resolve()),
            },
        },
        "input": {
            "dataset": "CPSC2018",
            "selection_csv_sha256": _sha256(args.selection_csv.resolve()),
            "selection_rule": "exactly reuse the prior XResNet visualization records and order",
            "sampling_rate_hz": args.source_rate_hz,
            "window_seconds": args.window_seconds,
            "window_rule": "first 10 seconds",
            "lead_order": LEAD_NAMES,
            "normalization": "per-record global z-score across all time samples and 12 leads",
        },
        "gradcam": {
            "score": "pre-sigmoid NORM logit for both AF and non-AF groups",
            "target_layer": "complete encoder output (1024 channels x 5000 samples)",
            "channel_weights": "temporal mean of activation gradients",
            "positive_evidence": "ReLU",
            "normalization": "independent per-record min-max to [0,1]",
            "threshold": None,
        },
        "semantic": {
            "source": "checkpoint semantic decoder softmax",
            "display": "per-sample argmax over background/P/QRS/T",
            "ground_truth_available_on_CPSC": False,
        },
        "execution": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "requested_device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "gradient_determinism": (
                "CuBLAS uses a deterministic workspace configuration, but CUDA adaptive "
                "max-pool backward may lack a deterministic implementation in PyTorch 2.0; "
                "record selection and preprocessing are deterministic."
            ),
            "rcfm_git_commit": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
            "rcfm_git_dirty": bool(_git_value(REPO_ROOT, "status", "--porcelain")),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "artifact_sha256": {name: _sha256(output_dir / name) for name in artifact_names},
        "outputs": [*artifact_names, "protocol.json"],
    }
    with (output_dir / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output_dir": str(output_dir), "records": len(records)}, sort_keys=True))
    return output_dir


if __name__ == "__main__":
    run(build_argparser().parse_args())
