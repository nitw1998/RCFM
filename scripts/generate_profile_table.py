"""Generate review-ready CSV/LaTeX profiling tables from profiler JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _latencies(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {profile["model"]: profile["latency_ms"]["median"] for profile in payload["profiles"]}


def rows(complexity_path: Path, latency_path: Path | None, status: str) -> list[dict[str, object]]:
    payload = json.loads(complexity_path.read_text(encoding="utf-8"))
    latency = _latencies(latency_path)
    input_metadata = payload["input"]
    return [
        {
            "status": status,
            "model": profile["model"],
            "batch_size": input_metadata["batch_size"],
            "length": input_metadata["length"],
            "precision": input_metadata["precision"],
            "device": input_metadata["device"],
            "steps": profile["sampling_steps"],
            "nfe": profile["neural_function_evaluations"],
            "total_parameters": profile["parameters"]["total"],
            "trainable_parameters": profile["parameters"]["trainable"],
            "macs_per_nfe": profile["macs_per_neural_function_evaluation"],
            "flops_per_nfe": profile["flops_per_neural_function_evaluation"],
            "total_macs": profile["total_macs_per_generated_sample_including_condition"],
            "total_flops": profile["total_flops_per_generated_sample_including_condition"],
            "median_latency_ms": latency.get(profile["model"], ""),
        }
        for profile in payload["profiles"]
    ]


def write_csv(path: Path, table_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)


def write_latex(path: Path, table_rows: list[dict[str, object]]) -> None:
    lines = [
        "% Generated from machine-readable profiler JSON; do not edit values manually.",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Status & Model & Params & NFE & MACs/NFE & Total MACs & Latency (ms) \\\\",
        "\\midrule",
    ]
    for row in table_rows:
        latency = row["median_latency_ms"] if row["median_latency_ms"] != "" else "--"
        lines.append(
            f"{row['status']} & {row['model']} & {row['total_parameters']} & {row['nfe']} & "
            f"{float(row['macs_per_nfe']):.0f} & {float(row['total_macs']):.0f} & {latency} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complexity", type=Path, required=True)
    parser.add_argument("--latency", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--latex", type=Path, required=True)
    parser.add_argument("--status", default="PENDING_FULL_PROFILE")
    args = parser.parse_args()
    table_rows = rows(args.complexity, args.latency, args.status)
    write_csv(args.csv, table_rows)
    write_latex(args.latex, table_rows)
    print(f"generated {len(table_rows)} profiling rows")


if __name__ == "__main__":
    main()
