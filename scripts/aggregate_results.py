"""Aggregate multi-seed run records and paired subject-level comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.metrics.statistics import (
    aggregate_seed_metrics,
    compare_models,
    holm_adjust,
    validate_run_record,
)


def _load(paths: list[Path]) -> list[dict[str, object]]:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for record in records:
        validate_run_record(record)
    return records


def _common_metrics(records: list[dict[str, object]]) -> list[str]:
    metric_sets = [
        set(subject["metrics"])
        for record in records
        for subject in record["subject_metrics"]
    ]
    return sorted(set.intersection(*metric_sets)) if metric_sets else []


def aggregate(
    records: list[dict[str, object]],
    reference_model: str,
    comparison_model: str,
    bootstrap_iterations: int,
) -> dict[str, object]:
    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_model[str(record["provenance"]["model"])].append(record)
    summaries = {
        model: aggregate_seed_metrics(model_records)
        for model, model_records in sorted(by_model.items())
    }
    comparisons = {}
    if reference_model in by_model and comparison_model in by_model:
        metrics = _common_metrics(by_model[reference_model] + by_model[comparison_model])
        for metric in metrics:
            comparisons[metric] = compare_models(
                by_model[reference_model],
                by_model[comparison_model],
                metric,
                bootstrap_iterations=bootstrap_iterations,
            )
        adjusted = holm_adjust(
            {metric: result["p_value"] for metric, result in comparisons.items()}
        )
        for metric, value in adjusted.items():
            comparisons[metric]["holm_adjusted_p_value"] = value
    return {
        "schema_version": 1,
        "reference_model": reference_model,
        "comparison_model": comparison_model,
        "run_count": len(records),
        "seed_summaries": summaries,
        "paired_comparisons": comparisons,
    }


def _write_summary_csv(path: Path, payload: dict[str, object]) -> None:
    fields = ["model", "metric", "n_seeds", "mean", "std", "status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, metrics in payload["seed_summaries"].items():
            for metric, result in metrics.items():
                writer.writerow(
                    {
                        "model": model,
                        "metric": metric,
                        **{name: result[name] for name in fields[2:]},
                    }
                )


def _write_comparison_csv(path: Path, payload: dict[str, object]) -> None:
    fields = [
        "metric",
        "n_subjects",
        "n_seeds",
        "mean_difference",
        "ci_lower",
        "ci_upper",
        "test",
        "p_value",
        "holm_adjusted_p_value",
        "effect_name",
        "effect_value",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric, result in payload["paired_comparisons"].items():
            writer.writerow(
                {
                    "metric": metric,
                    "n_subjects": result["n_subjects"],
                    "n_seeds": result["n_seeds"],
                    "mean_difference": result["mean_difference"],
                    "ci_lower": result["confidence_interval"]["lower"],
                    "ci_upper": result["confidence_interval"]["upper"],
                    "test": result["test"],
                    "p_value": result["p_value"],
                    "holm_adjusted_p_value": result["holm_adjusted_p_value"],
                    "effect_name": result["effect_size"]["name"],
                    "effect_value": result["effect_size"]["value"],
                }
            )


def _write_latex(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "% Generated from validated run records; do not edit numerical values manually.",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Model & Metric & Mean & Std \\",
        r"\midrule",
    ]
    if not payload["seed_summaries"]:
        lines.append(r"TODO & TODO & -- & -- \\")
    else:
        for model, metrics in payload["seed_summaries"].items():
            for metric, result in metrics.items():
                std = "--" if result["std"] is None else f"{result['std']:.6g}"
                lines.append(
                    f"{model} & {metric} & {result['mean']:.6g} & {std} \\\\"
                )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="*", default=[])
    parser.add_argument("--reference_model", default="RDDM")
    parser.add_argument("--comparison_model", default="RCFM")
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--comparison_csv", type=Path, required=True)
    parser.add_argument("--latex", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate(
        _load(args.runs),
        args.reference_model,
        args.comparison_model,
        args.bootstrap_iterations,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_summary_csv(args.summary_csv, payload)
    _write_comparison_csv(args.comparison_csv, payload)
    _write_latex(args.latex, payload)
    print(f"aggregated {payload['run_count']} run records")


if __name__ == "__main__":
    main()
