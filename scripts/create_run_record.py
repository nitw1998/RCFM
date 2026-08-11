"""Create a validated per-run result record from subject-metric CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.metrics.statistics import make_run_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject_metrics_csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_hash", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--git_commit", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--implementation_status",
        choices=["official", "reproduced", "adapted", "proposed"],
        required=True,
    )
    parser.add_argument("--run_command", required=True)
    args = parser.parse_args()

    with args.subject_metrics_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "subject_id" not in reader.fieldnames:
            raise ValueError("subject metric CSV requires a subject_id column")
        metric_names = [name for name in reader.fieldnames if name != "subject_id"]
        if not metric_names:
            raise ValueError("subject metric CSV requires at least one metric column")
        subject_metrics = [
            {
                "subject_id": row["subject_id"],
                "metrics": {name: float(row[name]) for name in metric_names},
            }
            for row in reader
        ]
    provenance = {
        "model": args.model,
        "config": args.config,
        "seed": args.seed,
        "split_hash": args.split_hash,
        "checkpoint": args.checkpoint,
        "git_commit": args.git_commit,
        "dataset": args.dataset,
        "task": args.task,
        "implementation_status": args.implementation_status,
        "command": args.run_command,
    }
    record = make_run_record(provenance, subject_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"created run record with {len(subject_metrics)} subjects")


if __name__ == "__main__":
    main()
