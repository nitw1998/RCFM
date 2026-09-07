#!/usr/bin/env python3
"""Plot window-level Bland--Altman panels from a completed delineation audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_window_bland_altman_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TIMING = ("heart_rate_bpm", "rr_ms", "pr_ms", "qrs_ms", "qt_ms")
AMPLITUDE = ("p_amplitude", "r_amplitude", "t_amplitude")
LABELS = {
    "heart_rate_bpm": "Heart rate (bpm)", "rr_ms": "RR interval (ms)",
    "pr_ms": "PR interval (ms)", "qrs_ms": "QRS duration (ms)",
    "qt_ms": "QT interval (ms)", "p_amplitude": "P amplitude (normalized)",
    "r_amplitude": "R amplitude (normalized)", "t_amplitude": "T amplitude (normalized)",
}
ALGORITHM_LABELS = {
    "ecgdeli_port_fixed": "ECGdeli unofficial port (fixed)",
    "wfdb_ecgpuwave_1.3.4": "WFDB ecgpuwave 1.3.4",
}
SEED_COLORS = {31: "#2878b5", 32: "#2f8f5b", 33: "#c43d4b"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pair_window_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Pivot reference/generated rows without treating seeds as repeated windows."""
    lookup: dict[tuple[str, int, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        if row.get("success") != "True":
            continue
        key = (row["algorithm"], int(row["seed"]), int(row["window_index"]))
        source = row["source"]
        if source in lookup.setdefault(key, {}):
            raise ValueError(f"duplicate {source} row for {key}")
        lookup[key][source] = row

    output: list[dict[str, object]] = []
    for (algorithm, seed, window), sources in sorted(lookup.items()):
        if set(sources) != {"reference", "generated"}:
            continue
        reference, generated = sources["reference"], sources["generated"]
        result: dict[str, object] = {
            "algorithm": algorithm, "seed": seed, "window_index": window,
            "group_id": reference["group_id"], "record_id": reference["record_id"],
            "afib": reference["afib"],
        }
        for mode in ("raw", "qc"):
            for parameter in (*TIMING, *AMPLITUDE):
                for source, row in (("reference", reference), ("generated", generated)):
                    value = row.get(f"{mode}_{parameter}", "")
                    result[f"{source}_{mode}_{parameter}"] = float(value) if value not in ("", None) else None
        output.append(result)
    return output


def agreement_rows(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    algorithms = sorted({str(row["algorithm"]) for row in pairs})
    seeds = sorted({int(row["seed"]) for row in pairs})
    for algorithm in algorithms:
        for seed in seeds:
            take = [row for row in pairs if row["algorithm"] == algorithm and row["seed"] == seed]
            for mode in ("raw", "qc"):
                for parameter in (*TIMING, *AMPLITUDE):
                    values = [
                        (float(row[f"reference_{mode}_{parameter}"]),
                         float(row[f"generated_{mode}_{parameter}"]))
                        for row in take
                        if row[f"reference_{mode}_{parameter}"] is not None
                        and row[f"generated_{mode}_{parameter}"] is not None
                    ]
                    difference = np.asarray([generated - reference for reference, generated in values])
                    bias = float(np.mean(difference)) if len(difference) else None
                    sd = float(np.std(difference, ddof=1)) if len(difference) > 1 else None
                    output.append({
                        "algorithm": algorithm, "seed": seed, "mode": mode,
                        "parameter": parameter, "paired_windows": len(values), "bias": bias, "sd": sd,
                        "lower_loa": bias - 1.96 * sd if sd is not None else None,
                        "upper_loa": bias + 1.96 * sd if sd is not None else None,
                    })
    return output


def _plot(
    output: Path, pairs: list[dict[str, object]], algorithm: str, mode: str,
    parameters: tuple[str, ...], stem: str,
) -> None:
    seeds = sorted({int(row["seed"]) for row in pairs})
    figure, axes = plt.subplots(
        len(seeds), len(parameters),
        figsize=(15.0, 8.4) if len(parameters) == 5 else (10.0, 8.4),
        constrained_layout=True, squeeze=False,
    )
    for row_index, seed in enumerate(seeds):
        color = SEED_COLORS.get(seed, "#2878b5")
        for column, parameter in enumerate(parameters):
            axis = axes[row_index, column]
            take = [row for row in pairs if row["algorithm"] == algorithm and row["seed"] == seed
                    and row[f"reference_{mode}_{parameter}"] is not None
                    and row[f"generated_{mode}_{parameter}"] is not None]
            reference = np.asarray([row[f"reference_{mode}_{parameter}"] for row in take], dtype=float)
            generated = np.asarray([row[f"generated_{mode}_{parameter}"] for row in take], dtype=float)
            means = (reference + generated) / 2
            differences = generated - reference
            if len(differences):
                bias = float(np.mean(differences))
                sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
                axis.scatter(means, differences, s=3, alpha=0.16, color=color, edgecolors="none", rasterized=True)
                axis.axhline(bias, color="#d62728", linewidth=0.8)
                axis.axhline(bias + 1.96 * sd, color="#555", linewidth=0.6, linestyle="--")
                axis.axhline(bias - 1.96 * sd, color="#555", linewidth=0.6, linestyle="--")
            axis.set_title(f"{LABELS[parameter]}; n={len(take)}", fontsize=7)
            axis.set_xlabel("Pair mean", fontsize=6)
            axis.set_ylabel(f"seed {seed}: generated - reference", fontsize=6)
            axis.tick_params(labelsize=5)
            axis.grid(alpha=0.15, linewidth=0.4)
    figure.suptitle(f"{ALGORITHM_LABELS[algorithm]}: {mode.upper()} window-level Bland-Altman", fontsize=9)
    figure.savefig(output / f"{stem}.png", dpi=600)
    figure.savefig(output / f"{stem}.pdf")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    source = args.per_window_csv.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pairs = pair_window_rows(rows)
    if not pairs:
        raise ValueError("no complete reference/generated window pairs")
    _write_csv(output / "per_window_bland_altman_pairs.csv", pairs)
    agreement = agreement_rows(pairs)
    _write_csv(output / "window_bland_altman_summary.csv", agreement)
    for algorithm, prefix in (("ecgdeli_port_fixed", "ecgdeli_port_fixed"),
                              ("wfdb_ecgpuwave_1.3.4", "wfdb_ecgpuwave")):
        for mode in ("raw", "qc"):
            _plot(output, pairs, algorithm, mode, TIMING, f"{prefix}_timing_window_bland_altman_{mode}")
        _plot(output, pairs, algorithm, "qc", AMPLITUDE,
              f"{prefix}_amplitude_window_bland_altman_qc")
    summary = {
        "schema_version": 1, "status": "complete", "dataset": args.dataset,
        "aggregation": "window", "seeds": sorted({int(row["seed"]) for row in pairs}),
        "window_pairs": len(pairs), "source_rows": len(rows), "agreement": agreement,
        "interpretation_boundary": (
            "Windows are correlated and identities overlap the random-window training split; "
            "window-level points are descriptive morphology diagnostics, not independent clinical replicates."
        ),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    protocol = {"command": shlex.join([sys.executable, *sys.argv]),
                "input": {"path": str(source), "sha256": _sha256(source)}}
    protocol["outputs"] = {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mimic_afib", "mmecg"), required=True)
    parser.add_argument("--per_window_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
