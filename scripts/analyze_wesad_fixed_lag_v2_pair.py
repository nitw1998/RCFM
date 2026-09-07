#!/usr/bin/env python3
"""Patient-level descriptive comparison for WESAD fixed-lag-v2 RCFM pair."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODELS = ("rcfm", "rcfm_ot")


def subject_waveform_metrics(reference: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    real = np.asarray(reference, dtype=np.float64)[:, 0]
    fake = np.asarray(generated, dtype=np.float64)[:, 0]
    error = fake - real
    real_centered = real - np.mean(real, axis=1, keepdims=True)
    fake_centered = fake - np.mean(fake, axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(real_centered**2, axis=1) * np.sum(fake_centered**2, axis=1))
    correlation = np.divide(np.sum(real_centered * fake_centered, axis=1), denominator,
                            out=np.full(len(real), np.nan), where=denominator > 0)
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error))),
            "median_record_pearson": float(np.nanmedian(correlation))}


def run(args: argparse.Namespace) -> Path:
    with np.load(args.raw_predictions, allow_pickle=False) as artifact:
        required = {"targets", "subject_ids", "rcfm_predictions", "rcfm_ot_predictions"}
        if missing := sorted(required - set(artifact.files)):
            raise ValueError("raw pair artifact is missing: " + ", ".join(missing))
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        subjects = np.asarray(artifact["subject_ids"]).astype(str)
        predictions = {model: np.asarray(artifact[f"{model}_predictions"], dtype=np.float32) for model in MODELS}
    rows = []
    for subject in sorted(set(subjects)):
        take = subjects == subject
        for model in MODELS:
            rows.append({"subject_id": subject, "model": model, "windows": int(np.sum(take)),
                         **subject_waveform_metrics(targets[take], predictions[model][take])})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_subject_waveform_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    indexed = {(row["subject_id"], row["model"]): row for row in rows}
    differences = {}
    for metric in ("rmse", "mae", "median_record_pearson"):
        values = [indexed[(subject, "rcfm_ot")][metric] - indexed[(subject, "rcfm")][metric]
                  for subject in sorted(set(subjects))]
        differences[metric] = {
            "definition": "rcfm_ot_minus_rcfm", "per_subject": values,
            "mean": float(np.mean(values)), "all_same_sign": bool(all(x > 0 for x in values) or all(x < 0 for x in values)),
            "inference_status": "descriptive_only_n3_no_p_value",
        }
    summary_path = args.output_dir / "paired_subject_summary.json"
    summary_path.write_text(json.dumps({"subjects": sorted(set(subjects)), "rows": rows,
                                        "paired_differences": differences},
                                       indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
