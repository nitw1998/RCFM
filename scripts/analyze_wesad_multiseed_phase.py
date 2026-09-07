"""Aggregate WESAD three-seed raw and oracle-aligned waveform results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_cpsc2018_multiseed import COMPARISONS, SEEDS, _holm_adjust, _paired_test
from scripts.evaluate_cpsc_zscore_paired import _array_sha256, _waveform_metrics

MODELS = ("CFM", "RCFM-Pan", "RCFM-Pan+OT", "RDDM")
MODEL_KEYS = dict(zip(MODELS, ("cfm", "rcfm", "rcfm_ot", "rddm")))
PHASES = ("raw_full_window", "oracle_aligned_fixed_support")
METRICS = ("rmse", "mae", "waveform_fd", "pearson_window_median")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_values(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    summary, _ = _waveform_metrics(targets, predictions)
    return {"rmse": float(summary["rmse"]), "mae": float(summary["mae"]),
            "waveform_fd": float(summary["waveform_fd"]),
            "pearson_window_median": float(summary["per_record_pearson"]["median"])}


def _load(raw_path: Path, phase_path: Path) -> tuple[dict[str, dict[str, np.ndarray]], tuple[str, ...]]:
    with np.load(raw_path, allow_pickle=False) as raw, np.load(phase_path, allow_pickle=False) as phase:
        raw_required = {"targets", "subject_ids", "labels"} | {f"{key}_predictions" for key in MODEL_KEYS.values()}
        phase_required = {"targets", "subject_ids", "labels"} | {f"{key}_oracle_aligned_predictions" for key in MODEL_KEYS.values()}
        if missing := sorted(raw_required - set(raw.files)):
            raise ValueError("WESAD raw artifact is missing: " + ", ".join(missing))
        if missing := sorted(phase_required - set(phase.files)):
            raise ValueError("WESAD phase artifact is missing: " + ", ".join(missing))
        raw_targets = np.asarray(raw["targets"], dtype=np.float32)
        phase_targets = np.asarray(phase["targets"], dtype=np.float32)
        identities = (str(_array_sha256(raw["subject_ids"])), str(_array_sha256(raw["labels"])))
        if identities != (str(_array_sha256(phase["subject_ids"])), str(_array_sha256(phase["labels"]))):
            raise ValueError("WESAD raw/phase identities differ")
        values = {
            model: {
                "raw_full_window": np.asarray(raw[f"{key}_predictions"], dtype=np.float32),
                "oracle_aligned_fixed_support": np.asarray(phase[f"{key}_oracle_aligned_predictions"], dtype=np.float32),
            } for model, key in MODEL_KEYS.items()
        }
    if raw_targets.shape != (4213, 1, 512) or phase_targets.shape != (4213, 1, 480):
        raise ValueError("WESAD target arrays violate the frozen shape contract")
    for model in MODELS:
        if values[model]["raw_full_window"].shape != raw_targets.shape or values[model]["oracle_aligned_fixed_support"].shape != phase_targets.shape:
            raise ValueError("WESAD prediction arrays violate the frozen shape contract")
    values["targets"] = {"raw_full_window": raw_targets, "oracle_aligned_fixed_support": phase_targets}
    return values, identities


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {seed: (getattr(args, f"seed{seed}_raw").resolve(), getattr(args, f"seed{seed}_phase").resolve()) for seed in SEEDS}
    values: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
    frozen_hashes = None
    for seed, (raw_path, phase_path) in artifacts.items():
        arrays, identities = _load(raw_path, phase_path)
        hashes = tuple(_array_sha256(arrays["targets"][phase]) for phase in PHASES) + identities
        if frozen_hashes is None:
            frozen_hashes = hashes
        elif hashes != frozen_hashes:
            raise ValueError("WESAD seeds do not share identical target rows and identities")
        values[seed] = {model: {phase: _metric_values(arrays["targets"][phase], arrays[model][phase]) for phase in PHASES} for model in MODELS}

    per_seed_rows = [{"model": model, "seed": seed, "phase_mode": phase, **values[seed][model][phase]}
                     for model in MODELS for seed in SEEDS for phase in PHASES]
    with (output / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_rows[0]); writer.writeheader(); writer.writerows(per_seed_rows)
    summary_rows = []
    for model in MODELS:
        for phase in PHASES:
            for metric in METRICS:
                samples = np.asarray([values[seed][model][phase][metric] for seed in SEEDS])
                summary_rows.append({"model": model, "phase_mode": phase, "metric": metric,
                                     "n_training_seeds": 3, "mean": float(samples.mean()),
                                     "sample_sd_ddof1": float(samples.std(ddof=1))})
    with (output / "model_phase_mean_sd.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0]); writer.writeheader(); writer.writerows(summary_rows)

    tests, raw_p = {}, {}
    for comparison, first, second in COMPARISONS:
        tests[comparison] = {}
        for metric in METRICS:
            a = np.asarray([values[seed][first]["oracle_aligned_fixed_support"][metric] for seed in SEEDS])
            b = np.asarray([values[seed][second]["oracle_aligned_fixed_support"][metric] for seed in SEEDS])
            test = _paired_test(a, b); test.update({"first_model": first, "second_model": second,
                                                    "lower_is_better": metric != "pearson_window_median"})
            tests[comparison][metric] = test; raw_p[f"{comparison}/{metric}"] = float(test["raw_paired_t_p_value"])
    adjusted = _holm_adjust(raw_p); test_rows = []
    for comparison, metric_tests in tests.items():
        for metric, test in metric_tests.items():
            test["holm_adjusted_p_value_16_tests"] = adjusted[f"{comparison}/{metric}"]
            test_rows.append({"comparison": comparison, "metric": metric, **test})
    with (output / "paired_seed_phase_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key in test_rows[0] if key != "seed_differences_first_minus_second"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key in fields} for row in test_rows)

    summary = {"status": "completed", "dataset": "WESAD", "seeds": list(SEEDS),
               "mean_and_sample_sd": summary_rows, "paired_seed_tests_oracle_aligned": tests,
               "multiplicity_family": "Holm adjustment across 4 predeclared comparisons x 4 oracle-aligned waveform metrics",
               "claim_boundary": "Three training seeds but only three held-out subjects. Exact sign-flip minimum p=0.25. Oracle phase correction uses each held-out ECG target."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inputs = {f"seed{seed}_{kind}": _sha256(path) for seed, pair in artifacts.items() for kind, path in zip(("raw", "phase"), pair)}
    protocol = {"status": "completed", "input_sha256": inputs, "frozen_array_sha256": frozen_hashes,
                "script_sha256": _sha256(Path(__file__).resolve()),
                "outputs": ["per_seed_metrics.csv", "model_phase_mean_sd.csv", "paired_seed_phase_tests.csv", "summary.json", "protocol.json"]}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in SEEDS:
        parser.add_argument(f"--seed{seed}_raw", type=Path, required=True)
        parser.add_argument(f"--seed{seed}_phase", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
