"""Create paired patient-cluster comparisons for the PTB-XL path ablation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_ptbxl_fourway import _cluster_bootstrap_difference
from scripts.evaluate_cpsc_zscore_paired import _sha256
from scripts.evaluate_ptbxl_path_ablation import MODEL_ORDER


def _read_per_record(path: Path) -> dict[str, dict[str, np.ndarray]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError("per-record metric table is empty")
    return {
        model: {
            metric: np.asarray(
                [float(row[f"{model}_{metric}"]) for row in rows], dtype=np.float64
            )
            for metric in ("rmse", "mae")
        }
        for model in MODEL_ORDER
    }


def run(args: argparse.Namespace) -> Path:
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_protocol = json.loads((input_dir / "protocol.json").read_text(encoding="utf-8"))
    if (
        source_protocol.get("status") != "completed"
        or source_protocol.get("protocol", {}).get("phase_correction_applied") is not False
        or set(source_protocol.get("selection", {})) != set(MODEL_ORDER)
    ):
        raise ValueError("analysis requires the completed raw epoch-500 path-ablation artifact")
    with np.load(input_dir / "paired_reference.npz", allow_pickle=False) as artifact:
        patient_ids = np.asarray(artifact["patient_ids"])
    per_record = _read_per_record(input_dir / "per_record_metrics.csv")
    if any(len(values["rmse"]) != len(patient_ids) for values in per_record.values()):
        raise ValueError("per-record metrics and patient IDs do not align")

    comparisons = {}
    for pair_index, (first, second) in enumerate(itertools.combinations(MODEL_ORDER, 2)):
        comparisons[f"{first}_vs_{second}"] = {
            metric: _cluster_bootstrap_difference(
                per_record[first][metric],
                per_record[second][metric],
                patient_ids,
                args.bootstrap_seed + pair_index * 10 + metric_index,
                args.bootstrap_replicates,
            )
            for metric_index, metric in enumerate(("rmse", "mae"))
        }
    result_path = output_dir / "patient_cluster_comparisons.json"
    result_path.write_text(
        json.dumps(
            {"schema_version": 1, "model_order": list(MODEL_ORDER), "comparisons": comparisons},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "records": int(len(patient_ids)),
        "patients": int(len(np.unique(patient_ids))),
        "model_order": list(MODEL_ORDER),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "inference_scope": "fixed_seed_fixed_checkpoint_test_patient_resampling_only",
        "source_protocol_sha256": _sha256(input_dir / "protocol.json"),
        "source_per_record_metrics_sha256": _sha256(input_dir / "per_record_metrics.csv"),
        "script_sha256": _sha256(Path(__file__)),
        "outputs": {result_path.name: _sha256(result_path)},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_seed", type=int, default=2031)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    return parser


if __name__ == "__main__":
    output = run(build_argparser().parse_args())
    print(f"PTB-XL path/coupling paired analysis saved to {output}")
