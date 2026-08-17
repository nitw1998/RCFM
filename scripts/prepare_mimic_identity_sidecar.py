"""Recover MIMIC-AFib window identity from public WFDB headers and waveforms."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb
import wfdb.processing


SUBJECT_PATTERN = re.compile(r"<Original Subject ID>:\s*([^\s;]+)")
RECORD_PATTERN = re.compile(r"<Original Recording ID>:\s*([^\s;]+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _header_metadata(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    subject = SUBJECT_PATTERN.search(text)
    record = RECORD_PATTERN.search(text)
    if subject is None or record is None:
        raise ValueError(f"header lacks original subject/record metadata: {path}")
    return subject.group(1), record.group(1)


def _signal_index(names: list[str], prefix: str) -> int:
    matches = [index for index, name in enumerate(names) if name.upper().startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {prefix} channel, found {matches}")
    return matches[0]


def _record_windows(path: Path, output_rate: int = 128) -> tuple[np.ndarray, np.ndarray]:
    signal, metadata = wfdb.rdsamp(str(path.with_suffix("")))
    names = list(metadata["sig_name"])
    source_rate = int(round(float(metadata["fs"])))
    ppg = wfdb.processing.resample_sig(
        signal[:, _signal_index(names, "PPG")], source_rate, output_rate
    )[0]
    ecg = wfdb.processing.resample_sig(
        signal[:, _signal_index(names, "ECG")], source_rate, output_rate
    )[0]
    ppg = np.nan_to_num(ppg)
    ecg = np.nan_to_num(ecg)
    window_samples = 4 * output_rate
    count = min(len(ppg), len(ecg)) // window_samples
    return (
        ecg[: count * window_samples].reshape(count, window_samples),
        ppg[: count * window_samples].reshape(count, window_samples),
    )


def _match_source_windows(
    upstream_ecg: np.ndarray,
    upstream_ppg: np.ndarray,
    source_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Map every source window to one upstream row, allowing only zero-PPG fallback."""

    if upstream_ecg.shape != upstream_ppg.shape or upstream_ecg.ndim != 2:
        raise ValueError("upstream ECG/PPG arrays must have matching 2D shapes")
    pair_lookup: dict[str, list[int]] = defaultdict(list)
    ecg_lookup: dict[str, list[int]] = defaultdict(list)
    for index in range(len(upstream_ecg)):
        pair_lookup[_row_hash(upstream_ecg[index], upstream_ppg[index])].append(index)
        ecg_lookup[_row_hash(upstream_ecg[index])].append(index)
    used: set[int] = set()
    matched: list[dict[str, object]] = []
    exact_count = 0
    zero_ppg_fallback_count = 0
    for source in source_rows:
        ecg = np.asarray(source["ecg"])
        ppg = np.asarray(source["ppg"])
        pair_candidates = [
            index for index in pair_lookup[_row_hash(ecg, ppg)] if index not in used
        ]
        if len(pair_candidates) == 1:
            upstream_index = pair_candidates[0]
            match_method = "exact_ecg_ppg"
            exact_count += 1
        else:
            ecg_candidates = [
                index for index in ecg_lookup[_row_hash(ecg)] if index not in used
            ]
            if len(ecg_candidates) != 1:
                raise ValueError(
                    "source window does not map uniquely by ECG+PPG or ECG-only: "
                    f"record={source['source_record_name']} window={source['window_index']} "
                    f"pair_candidates={len(pair_candidates)} ecg_candidates={len(ecg_candidates)}"
                )
            upstream_index = ecg_candidates[0]
            if not np.all(upstream_ppg[upstream_index] == 0):
                raise ValueError("ECG-only fallback is permitted only for an all-zero upstream PPG row")
            match_method = "ecg_only_upstream_ppg_all_zero"
            zero_ppg_fallback_count += 1
        used.add(upstream_index)
        row = {key: value for key, value in source.items() if key not in {"ecg", "ppg"}}
        row.update({"upstream_index": upstream_index, "match_method": match_method})
        matched.append(row)
    if len(used) != len(upstream_ecg) or len(matched) != len(upstream_ecg):
        raise ValueError(
            f"identity mapping is incomplete: matched={len(matched)} used={len(used)} "
            f"upstream={len(upstream_ecg)}"
        )
    return matched, {
        "exact_ecg_ppg_windows": exact_count,
        "ecg_only_all_zero_ppg_windows": zero_ppg_fallback_count,
    }


def _save_split(output_dir: Path, split: str, rows: list[dict[str, object]]) -> list[str]:
    ordered = sorted(rows, key=lambda row: int(row["split_row_before_qc"]))
    arrays = {
        f"subject_ids_{split}.npy": np.asarray([row["subject_id"] for row in ordered]),
        f"record_ids_{split}.npy": np.asarray([row["original_record_id"] for row in ordered]),
        f"source_record_names_{split}.npy": np.asarray([row["source_record_name"] for row in ordered]),
        f"afib_labels_{split}.npy": np.asarray([row["afib"] for row in ordered], dtype=np.bool_),
        f"window_indices_{split}.npy": np.asarray([row["window_index"] for row in ordered], dtype=np.int16),
        f"start_samples_128hz_{split}.npy": np.asarray(
            [int(row["window_index"]) * 512 for row in ordered], dtype=np.int32
        ),
        f"source_rows_before_qc_{split}.npy": np.asarray(
            [row["split_row_before_qc"] for row in ordered], dtype=np.int32
        ),
    }
    for name, values in arrays.items():
        np.save(output_dir / name, values, allow_pickle=False)
    return list(arrays)


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream_root = args.upstream_array_root.resolve()
    qc_root = args.qc_root.resolve()
    train_ecg = np.load(upstream_root / "ecg_train_4sec.npy", allow_pickle=False)
    train_ppg = np.load(upstream_root / "ppg_train_4sec.npy", allow_pickle=False)
    test_ecg = np.load(upstream_root / "ecg_test_4sec.npy", allow_pickle=False)
    test_ppg = np.load(upstream_root / "ppg_test_4sec.npy", allow_pickle=False)
    upstream_ecg = np.concatenate((train_ecg, test_ecg))
    upstream_ppg = np.concatenate((train_ppg, test_ppg))

    source_rows: list[dict[str, object]] = []
    header_paths = []
    for root, afib in ((args.af_root.resolve(), True), (args.non_af_root.resolve(), False)):
        for header in sorted(root.glob("*.hea")):
            subject_id, original_record_id = _header_metadata(header)
            ecg_windows, ppg_windows = _record_windows(header)
            if len(ecg_windows) != 300:
                raise ValueError(f"expected 300 complete windows for {header.name}, got {len(ecg_windows)}")
            header_paths.append(header)
            for window_index, (ecg, ppg) in enumerate(zip(ecg_windows, ppg_windows)):
                source_rows.append(
                    {
                        "subject_id": subject_id,
                        "original_record_id": original_record_id,
                        "source_record_name": header.stem,
                        "afib": afib,
                        "window_index": window_index,
                        "ecg": ecg,
                        "ppg": ppg,
                    }
                )
    if len(header_paths) != 35 or len(source_rows) != 10500:
        raise ValueError("the frozen MIMIC-PERform source must contain 35 records and 10,500 windows")
    subjects = [str(row["subject_id"]) for row in source_rows[::300]]
    if len(set(subjects)) != 35:
        raise ValueError("source records do not have one unique subject each")

    matched, match_counts = _match_source_windows(upstream_ecg, upstream_ppg, source_rows)
    train_count = len(train_ecg)
    pre_qc = {"train": [], "test": []}
    for row in matched:
        upstream_index = int(row["upstream_index"])
        split = "train" if upstream_index < train_count else "test"
        row["split_row_before_qc"] = upstream_index if split == "train" else upstream_index - train_count
        pre_qc[split].append(row)
    record_splits: dict[str, set[str]] = defaultdict(set)
    for split, rows in pre_qc.items():
        for row in rows:
            record_splits[str(row["source_record_name"])].add(split)
    crossing = [name for name, splits in record_splits.items() if len(splits) != 1]
    if crossing:
        raise ValueError(f"source records cross train/test: {len(crossing)}")

    kept = {
        split: np.load(qc_root / f"kept_indices_{split}.npy", allow_pickle=False).astype(int)
        for split in ("train", "test")
    }
    retained: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "test"):
        by_row = {int(row["split_row_before_qc"]): row for row in pre_qc[split]}
        if set(by_row) != set(range(len(pre_qc[split]))):
            raise ValueError(f"{split} pre-QC rows are not contiguous")
        retained[split] = [by_row[int(index)] for index in kept[split]]

    outputs = []
    for split in ("train", "test"):
        outputs.extend(_save_split(output_dir, split, retained[split]))
        outputs.extend(_save_split(output_dir, f"{split}_before_qc", pre_qc[split]))
    output_sha = {name: _sha256(output_dir / name) for name in sorted(outputs)}
    train_subjects = {str(row["subject_id"]) for row in retained["train"]}
    test_subjects = {str(row["subject_id"]) for row in retained["test"]}
    if train_subjects & test_subjects:
        raise ValueError("recovered retained train/test subjects overlap")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MIMIC-AFib",
        "source": "MIMIC PERform AF and non-AF WFDB cohorts",
        "mapping": (
            "source WFDB resample 125-to-128 Hz and nonoverlapping 512-sample windows "
            "matched to upstream arrays by exact ECG+PPG bytes, with ECG-only fallback "
            "allowed solely when upstream PPG is all zero"
        ),
        "match_counts": match_counts,
        "pre_qc": {
            "train_windows": len(pre_qc["train"]),
            "test_windows": len(pre_qc["test"]),
            "train_subjects": len({str(row["subject_id"]) for row in pre_qc["train"]}),
            "test_subjects": len({str(row["subject_id"]) for row in pre_qc["test"]}),
        },
        "retained": {
            "train_windows": len(retained["train"]),
            "test_windows": len(retained["test"]),
            "train_subjects": len(train_subjects),
            "test_subjects": len(test_subjects),
            "train_afib_windows": int(sum(bool(row["afib"]) for row in retained["train"])),
            "test_afib_windows": int(sum(bool(row["afib"]) for row in retained["test"])),
        },
        "subject_disjoint_verified": True,
        "identifiers_are_deidentified_public_source_metadata": True,
        "identifiers_not_for_git": True,
        "source_header_inventory_sha256": hashlib.sha256(
            "\n".join(f"{path.name}:{_sha256(path)}" for path in header_paths).encode("ascii")
        ).hexdigest(),
        "upstream_array_sha256": {
            path.name: _sha256(path)
            for path in sorted(upstream_root.glob("*.npy"))
            if path.name in {
                "ecg_train_4sec.npy", "ppg_train_4sec.npy",
                "ecg_test_4sec.npy", "ppg_test_4sec.npy",
            }
        },
        "qc_manifest_sha256": _sha256(qc_root / "dataset_manifest.json"),
        "output_sha256": output_sha,
    }
    (output_dir / "identity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--af_root", type=Path, required=True)
    parser.add_argument("--non_af_root", type=Path, required=True)
    parser.add_argument("--upstream_array_root", type=Path, required=True)
    parser.add_argument("--qc_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    result = run(build_argparser().parse_args())
    print(f"MIMIC identity sidecar saved to {result}")
