#!/usr/bin/env python3
"""Reconstruct QC-retained MIMIC-AFib Lead-II windows in physical mV."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import wfdb


SPLITS = ("train", "test")
SOURCE_RATE = 125
MODEL_RATE = 100
WINDOW_SECONDS = 4
NAN_QC_MARGIN_SAMPLES = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ecg_channel_and_lead(header) -> tuple[int, str]:
    matches = []
    for index, name in enumerate(header.sig_name):
        match = re.search(r"\blead\s+(II|III|I|V|MCL1)\b", name, flags=re.IGNORECASE)
        if name.lower().startswith("ecg,") and match:
            matches.append((index, match.group(1).upper()))
    if len(matches) != 1:
        raise ValueError(f"expected one labelled ECG channel, got {header.sig_name}")
    return matches[0]


def load_retained_identities(identity_dir: Path) -> list[tuple[str, int, int]]:
    rows = []
    for split in SPLITS:
        names = np.load(identity_dir / f"source_record_names_{split}.npy").astype(str)
        starts = np.load(identity_dir / f"start_samples_128hz_{split}.npy").astype(np.int64)
        labels = np.load(identity_dir / f"afib_labels_{split}.npy").astype(np.uint8)
        if not (len(names) == len(starts) == len(labels)):
            raise ValueError(f"unaligned identity arrays for {split}")
        rows.extend(zip(names.tolist(), starts.tolist(), labels.tolist()))
    if len({(name, start) for name, start, _ in rows}) != len(rows):
        raise ValueError("retained record/start identities are not unique")
    return sorted(rows)


def record_inventory(raw_root: Path) -> dict[str, Path]:
    paths = list(raw_root.glob("mimic_perform_*_wfdb/mimic_perform_*_wfdb/*.hea"))
    inventory = {path.stem: path.with_suffix("") for path in paths}
    if len(inventory) != len(paths):
        raise ValueError("duplicate WFDB record basenames")
    return inventory


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    repository = Path(__file__).resolve().parents[1]
    if output == repository or repository in output.parents:
        raise ValueError("patient-derived arrays must be written outside the Git repository")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    ptb_manifest_path = args.ptb_manifest.resolve()
    ptb_manifest = json.loads(ptb_manifest_path.read_text(encoding="utf-8"))
    if (
        ptb_manifest.get("lead") != "II"
        or ptb_manifest.get("sampling_rate_hz") != MODEL_RATE
        or ptb_manifest.get("samples") != MODEL_RATE * WINDOW_SECONDS
        or ptb_manifest.get("normalization", {}).get("type") != "global_train_zscore"
    ):
        raise ValueError("PTB-XL manifest is not the required Lead-II 100-Hz four-second protocol")
    mean = float(ptb_manifest["normalization"]["mean"])
    std = float(ptb_manifest["normalization"]["std"])

    identities = load_retained_identities(args.identity_dir.resolve())
    inventory = record_inventory(args.raw_root.resolve())
    selected_records: dict[str, tuple[Path, int]] = {}
    excluded = {}
    for name in sorted({row[0] for row in identities}):
        if name not in inventory:
            raise FileNotFoundError(f"retained record is absent from WFDB inventory: {name}")
        header = wfdb.rdheader(str(inventory[name]))
        channel, lead = ecg_channel_and_lead(header)
        if int(header.fs) != SOURCE_RATE or header.units[channel] != "mV":
            raise ValueError(f"{name}: expected 125-Hz mV ECG, got fs={header.fs}, units={header.units}")
        if lead == "II":
            selected_records[name] = (inventory[name], channel)
        else:
            excluded[name] = lead

    selected_rows = [row for row in identities if row[0] in selected_records]
    raw_windows, model_windows, labels, names, starts_125, starts_100 = [], [], [], [], [], []
    source_hashes = {}
    excluded_nonfinite_windows: dict[str, int] = {}
    by_record: dict[str, list[tuple[int, int]]] = {}
    for name, start_128, label in selected_rows:
        by_record.setdefault(name, []).append((start_128, label))
    for name in sorted(by_record):
        record_path, channel = selected_records[name]
        signal, _ = wfdb.rdsamp(str(record_path), channels=[channel])
        signal = signal[:, 0].astype(np.float64)
        finite = np.isfinite(signal)
        if not np.any(finite):
            raise ValueError(f"{name}: ECG is entirely nonfinite")
        signal_for_resampling = signal.copy()
        if not np.all(finite):
            positions = np.arange(len(signal))
            signal_for_resampling[~finite] = np.interp(
                positions[~finite], positions[finite], signal[finite]
            )
        resampled = resample_poly(signal_for_resampling, MODEL_RATE, SOURCE_RATE).astype(np.float32)
        excluded_nonfinite_windows[name] = 0
        for start_128, label in sorted(by_record[name]):
            if (start_128 * SOURCE_RATE) % 128 or (start_128 * MODEL_RATE) % 128:
                raise ValueError(f"{name}: window start does not map exactly between rates")
            start125 = start_128 * SOURCE_RATE // 128
            start100 = start_128 * MODEL_RATE // 128
            qc_start = max(0, start125 - NAN_QC_MARGIN_SAMPLES)
            qc_stop = min(len(signal), start125 + SOURCE_RATE * WINDOW_SECONDS + NAN_QC_MARGIN_SAMPLES)
            if not np.all(finite[qc_start:qc_stop]):
                excluded_nonfinite_windows[name] += 1
                continue
            raw = signal[start125 : start125 + SOURCE_RATE * WINDOW_SECONDS]
            model = resampled[start100 : start100 + MODEL_RATE * WINDOW_SECONDS]
            if raw.shape != (500,) or model.shape != (400,):
                raise ValueError(f"{name}: incomplete four-second window at {start_128}")
            raw_windows.append(raw.astype(np.float32))
            model_windows.append(((model - mean) / std).astype(np.float32))
            labels.append(label); names.append(name)
            starts_125.append(start125); starts_100.append(start100)
        for suffix in (".hea", ".dat"):
            path = record_path.with_suffix(suffix)
            source_hashes[f"{name}{suffix}"] = sha256(path)

    outputs = {
        "ecg_mV_125hz.npy": np.stack(raw_windows),
        "ecg_ptb_normalized_100hz.npy": np.stack(model_windows)[:, None, :],
        "afib_labels.npy": np.asarray(labels, dtype=np.uint8),
        "source_record_names.npy": np.asarray(names),
        "start_samples_125hz.npy": np.asarray(starts_125, dtype=np.int64),
        "start_samples_100hz.npy": np.asarray(starts_100, dtype=np.int64),
    }
    output_hashes = {}
    for filename, values in outputs.items():
        path = output / filename
        np.save(path, values, allow_pickle=False)
        output_hashes[filename] = sha256(path)

    unique_names = np.unique(outputs["source_record_names.npy"])
    unique_labels = [int(outputs["afib_labels.npy"][outputs["source_record_names.npy"] == name][0]) for name in unique_names]
    manifest = {
        "schema_version": 1, "status": "completed", "dataset": "MIMIC PERform AF",
        "selection": "records retained by existing paired PPG-QC identity artifact and labelled Lead II in WFDB header",
        "windows": len(labels), "records": len(unique_names),
        "af_records": int(sum(unique_labels)), "non_af_records": int(len(unique_labels) - sum(unique_labels)),
        "windows_per_record": sorted(set(np.unique(outputs["source_record_names.npy"], return_counts=True)[1].tolist())),
        "excluded_non_lead_ii": excluded,
        "nonfinite_qc": {
            "policy": "linearly interpolate only to resample the complete record, then exclude every window with a nonfinite source sample within a 25-sample (200-ms) margin",
            "margin_samples_125hz": NAN_QC_MARGIN_SAMPLES,
            "excluded_windows_by_record": {
                name: count for name, count in sorted(excluded_nonfinite_windows.items()) if count
            },
            "excluded_windows": int(sum(excluded_nonfinite_windows.values())),
        },
        "physical_input": {"rate_hz": SOURCE_RATE, "unit": "mV", "samples": 500},
        "model_input": {"rate_hz": MODEL_RATE, "lead": "II", "samples": 400, "shape": ["window", 1, 400]},
        "resampling": "scipy.signal.resample_poly over each complete WFDB record, up=4, down=5",
        "normalization": {"source": "PTB-XL training folds only", "mean": mean, "std": std},
        "ptb_manifest_sha256": sha256(ptb_manifest_path),
        "identity_manifest_sha256": sha256(args.identity_dir.resolve() / "identity_manifest.json"),
        "source_file_sha256": dict(sorted(source_hashes.items())),
        "output_sha256": dict(sorted(output_hashes.items())),
        "claim_boundary": "No MIMIC labels, scores, or waveforms were used to fit or calibrate the classifier.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("windows", "records", "af_records", "non_af_records", "excluded_non_lead_ii")}, indent=2))
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_root", type=Path, required=True)
    parser.add_argument("--identity_dir", type=Path, required=True)
    parser.add_argument("--ptb_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
