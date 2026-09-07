"""Summarize three RDDM training seeds for PTB-XL or CPSC2018."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _json, _sha256
from scripts.evaluate_ptbxl_fourway import _metric_summary


SEEDS = (31, 32, 33)
METRICS = ("rmse", "mae", "waveform_fd_macro_lead", "per_record_pearson_median")


def _parse_predictions(values: list[str]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("prediction specs must use SEED=DIR")
        raw_seed, raw_path = value.split("=", 1)
        try:
            seed = int(raw_seed)
        except ValueError as error:
            raise ValueError(f"invalid training seed: {raw_seed}") from error
        if seed not in SEEDS or seed in parsed:
            raise ValueError(f"invalid or duplicate RDDM training seed: {seed}")
        parsed[seed] = Path(raw_path)
    if set(parsed) != set(SEEDS):
        raise ValueError("RDDM results require exactly training seeds 31, 32, and 33")
    return parsed


def _load_prediction(
    directory: Path, dataset: str, seed: int, reference_hash: str, shape: tuple[int, ...]
) -> tuple[np.ndarray, Path]:
    protocol_path = directory.resolve() / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    inference = protocol.get("inference", {})
    if (
        protocol.get("status") != "completed"
        or protocol.get("model_name") != "rddm"
        or protocol.get("dataset") != dataset
        or protocol.get("training_seed") != seed
        or protocol.get("selection")
        != "predeclared_epoch_500_endpoint_without_validation_selection"
        or protocol.get("paired_reference", {}).get("sha256") != reference_hash
        or inference.get("sampling_seed") != 2025
        or inference.get("deterministic_seed") != 31
        or inference.get("phase_correction_applied") is not False
    ):
        raise ValueError(f"RDDM seed {seed} prediction protocol mismatch")
    prediction_path = directory.resolve() / "predictions.npy"
    if _sha256(prediction_path) != protocol.get("prediction", {}).get("sha256"):
        raise ValueError(f"RDDM seed {seed} prediction hash changed")
    prediction = np.asarray(np.load(prediction_path, mmap_mode="r"))
    if prediction.shape != shape or not np.all(np.isfinite(prediction)):
        raise ValueError(f"RDDM seed {seed} prediction shape/finiteness changed")
    return prediction, protocol_path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    prediction_dirs = _parse_predictions(args.prediction)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    reference_path = args.reference_artifact.resolve()
    reference_hash = _sha256(reference_path)
    with np.load(reference_path, allow_pickle=False) as arrays:
        targets = np.asarray(arrays["targets"], dtype=np.float32)
    if targets.ndim != 3 or targets.shape[1:] != (11, 512):
        raise ValueError("paired reference has the wrong shape")
    seed_rows: list[dict[str, object]] = []
    protocol_sources: dict[str, object] = {}
    for seed in SEEDS:
        prediction, protocol_path = _load_prediction(
            prediction_dirs[seed], args.dataset, seed, reference_hash, targets.shape
        )
        summary, _ = _metric_summary(targets, prediction)
        seed_rows.append({"model": "rddm", "seed": seed, **{key: summary[key] for key in METRICS}})
        protocol_sources[f"rddm_s{seed}"] = {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
        }
    model_row: dict[str, object] = {"model": "rddm", "training_seeds": 3}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in seed_rows], dtype=np.float64)
        model_row[f"{metric}_mean"] = float(values.mean())
        model_row[f"{metric}_sd"] = float(values.std(ddof=1))
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "seed_metrics.csv", seed_rows)
    _write_csv(output / "model_summary.csv", [model_row])
    _json(
        output / "protocol.json",
        {
            "schema_version": 1,
            "status": "completed",
            "dataset": args.dataset,
            "model": "RDDM-ECG (adapted)",
            "training_seeds": list(SEEDS),
            "result_definition": "mean and sample SD across three independently trained seeds",
            "inferential_p_values": None,
            "p_value_note": "No p-value is defined for a one-model seed summary; paired model comparisons require separately predeclared contrasts.",
            "phase_correction_applied": False,
            "paired_reference": {"path": str(reference_path), "sha256": reference_hash},
            "prediction_protocols": protocol_sources,
            "artifacts": {
                name: _sha256(output / name)
                for name in ("seed_metrics.csv", "model_summary.csv")
            },
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ptbxl", "cpsc2018"), required=True)
    parser.add_argument("--reference_artifact", type=Path, required=True)
    parser.add_argument("--prediction", action="append", default=[], help="SEED=DIR; repeat for 31, 32, and 33")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
