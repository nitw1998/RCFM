"""Build sparse PTB-XL+ delineation sidecars aligned to frozen PTB-XL arrays."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shlex
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import wfdb

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.rcfm.interpretability.delineation_dataset import (
    EVENT_NAMES,
    LIMB_LEADS,
    PTBXL_LEAD_ORDER,
    parse_fiducial_events,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_save(path: Path, array: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _annotation_relative_path(ecg_id: int, lead: str) -> str:
    bucket = (int(ecg_id) // 1000) * 1000
    return f"fiducial_points/ecgdeli/{bucket:05d}/{int(ecg_id):05d}_points_lead_{lead}.atr"


def _declared_checksums(checksum_file: Path) -> dict[str, str]:
    declared: dict[str, str] = {}
    with checksum_file.open(encoding="utf-8") as handle:
        for line in handle:
            digest, relative = line.rstrip("\n").split(" ", maxsplit=1)
            declared[relative] = digest
    return declared


def _rhythm_flags(metadata_path: Path) -> dict[int, bool]:
    metadata = pd.read_csv(metadata_path, usecols=["ecg_id", "scp_codes"])
    flags: dict[int, bool] = {}
    for row in metadata.itertuples(index=False):
        codes = ast.literal_eval(str(row.scp_codes))
        flags[int(row.ecg_id)] = bool({"AFIB", "AFLT"} & set(codes))
    return flags


def _split_assignment_hash(
    split_ids: Mapping[str, np.ndarray], eligible: Mapping[str, np.ndarray], leads: tuple[str, ...]
) -> str:
    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        for row_index, ecg_id in enumerate(split_ids[split]):
            for lead_index, lead in enumerate(leads):
                digest.update(
                    f"{split}\t{int(ecg_id)}\t{lead}\t{int(eligible[split][row_index, lead_index])}\n".encode()
                )
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Path:
    waveform_root = args.waveform_root.resolve()
    ptbxl_root = args.ptbxl_root.resolve()
    plus_root = args.ptbxl_plus_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == REPOSITORY_ROOT or REPOSITORY_ROOT in output_dir.parents:
        raise ValueError("delineation artifacts must be written outside the Git repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    waveform_manifest_path = waveform_root / "dataset_manifest.json"
    waveform_manifest = json.loads(waveform_manifest_path.read_text(encoding="utf-8"))
    if waveform_manifest.get("split_method") != "official_ptbxl_strat_fold_1_8_train_9_val_10_test":
        raise ValueError("waveform artifact does not use official PTB-XL folds")
    if int(waveform_manifest["stored_waveforms"]["sampling_rate_hz"]) != args.output_rate:
        raise ValueError("waveform artifact sampling rate does not match requested output rate")
    if int(waveform_manifest["stored_waveforms"]["samples_per_record"]) != args.output_samples:
        raise ValueError("waveform artifact record length does not match requested output length")

    leads = tuple(args.leads)
    invalid_leads = sorted(set(leads) - set(LIMB_LEADS))
    if invalid_leads or len(set(leads)) != len(leads):
        raise ValueError(f"selected leads must be unique limb leads; invalid={invalid_leads}")
    selected_indices = tuple(PTBXL_LEAD_ORDER.index(lead.upper()) for lead in leads)
    checksum_path = plus_root / "SHA256SUMS.txt"
    declared = _declared_checksums(checksum_path)
    af_or_flutter = _rhythm_flags(ptbxl_root / "ptbxl_database.csv")
    split_ids = {
        split: np.load(waveform_root / f"record_ids_{split}.npy", allow_pickle=False)
        for split in ("train", "val", "test")
    }
    output_hashes: dict[str, str] = {}
    eligible_by_split: dict[str, np.ndarray] = {}
    split_summaries: dict[str, object] = {}
    unavailable: list[dict[str, object]] = []
    unmatched_totals = {name: 0 for name in EVENT_NAMES}

    for split, record_ids in split_ids.items():
        shape_prefix = (len(record_ids), len(leads))
        positions = np.full(shape_prefix + (len(EVENT_NAMES), args.max_events), -1, dtype=np.int16)
        counts = np.zeros(shape_prefix + (len(EVENT_NAMES),), dtype=np.uint8)
        wave_valid = np.zeros(shape_prefix + (3,), dtype=np.uint8)
        eligible = np.zeros(shape_prefix, dtype=np.uint8)

        tasks: list[tuple[int, int, int, str, Path, str | None]] = []
        for row_index, ecg_id_value in enumerate(record_ids):
            ecg_id = int(ecg_id_value)
            for lead_index, lead in enumerate(leads):
                relative = _annotation_relative_path(ecg_id, lead)
                declared_digest = declared.get(relative)
                path = plus_root / relative
                if declared_digest is None:
                    unavailable.append(
                        {
                            "split": split,
                            "ecg_id": ecg_id,
                            "lead": lead,
                            "reason": "not_in_ptbxl_plus_release",
                        }
                    )
                elif not path.is_file():
                    unavailable.append(
                        {
                            "split": split,
                            "ecg_id": ecg_id,
                            "lead": lead,
                            "reason": "local_download_missing",
                        }
                    )
                else:
                    tasks.append((row_index, lead_index, ecg_id, lead, path, declared_digest))

        local_missing = [item for item in unavailable if item["split"] == split and item["reason"] == "local_download_missing"]
        if local_missing and not args.allow_incomplete_download:
            raise FileNotFoundError(
                f"PTB-XL+ download is incomplete for {split}; missing={local_missing[:10]}"
            )

        def parse_task(task):
            row_index, lead_index, ecg_id, lead, path, declared_digest = task
            if args.verify_checksums and _sha256(path) != declared_digest:
                raise ValueError(f"checksum mismatch for declared PTB-XL+ annotation {path.name}")
            annotation = wfdb.rdann(str(path.with_suffix("")), "atr")
            if int(annotation.fs or -1) != args.source_rate:
                raise ValueError(f"{path.name} has annotation sampling rate {annotation.fs}")
            parsed = parse_fiducial_events(
                annotation.sample,
                annotation.aux_note,
                source_rate_hz=args.source_rate,
                output_rate_hz=args.output_rate,
                output_samples=args.output_samples,
                max_events=args.max_events,
                disable_p_wave=af_or_flutter.get(ecg_id, False),
            )
            return row_index, lead_index, ecg_id, lead, parsed

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for completed, result in enumerate(executor.map(parse_task, tasks), start=1):
                row_index, lead_index, _ecg_id, _lead, parsed = result
                parsed_positions, parsed_counts, parsed_valid, unmatched = parsed
                positions[row_index, lead_index] = parsed_positions
                counts[row_index, lead_index] = parsed_counts
                wave_valid[row_index, lead_index] = parsed_valid
                eligible[row_index, lead_index] = parsed_valid[1]
                for name, value in unmatched.items():
                    unmatched_totals[name] += int(value)
                if completed % 5000 == 0:
                    print(f"{split}: parsed {completed}/{len(tasks)} annotations", flush=True)

        arrays = {
            f"record_ids_{split}.npy": np.asarray(record_ids, dtype=np.int32),
            f"fiducial_positions_{split}.npy": positions,
            f"fiducial_counts_{split}.npy": counts,
            f"wave_valid_{split}.npy": wave_valid,
            f"eligible_leads_{split}.npy": eligible,
        }
        for filename, array in arrays.items():
            path = output_dir / filename
            _atomic_save(path, array)
            output_hashes[filename] = _sha256(path)
        eligible_by_split[split] = eligible
        split_summaries[split] = {
            "records": int(len(record_ids)),
            "declared_local_annotations_parsed": int(len(tasks)),
            "eligible_record_leads": int(eligible.sum()),
            "eligible_windows": int(eligible.sum()) * len(args.crop_starts),
            "p_wave_valid_record_leads": int(wave_valid[:, :, 0].sum()),
            "qrs_valid_record_leads": int(wave_valid[:, :, 1].sum()),
            "t_wave_valid_record_leads": int(wave_valid[:, :, 2].sum()),
        }
        print(
            f"{split}: eligible lead-records={int(eligible.sum())}/{len(record_ids) * len(leads)}",
            flush=True,
        )

    assignment_hash = _split_assignment_hash(split_ids, eligible_by_split, leads)
    unavailable_counts: dict[str, int] = {}
    for item in unavailable:
        reason = str(item["reason"])
        unavailable_counts[reason] = unavailable_counts.get(reason, 0) + 1
    manifest = {
        "schema_version": 1,
        "dataset": "PTB-XL+ ECGdeli delineation",
        "dataset_version": "ptbxl-plus-1.0.1-ecgdeli-limb-delineation-128hz-v1",
        "annotation_provenance": "PTB-XL+ ECGdeli algorithm-generated fiducials; not manual ground truth",
        "waveform_dataset_version": waveform_manifest["dataset_version"],
        "waveform_split_hash": waveform_manifest["split_hash"],
        "waveform_manifest_sha256": _sha256(waveform_manifest_path),
        "ptbxl_metadata_sha256": _sha256(ptbxl_root / "ptbxl_database.csv"),
        "ptbxl_plus_checksums_sha256": _sha256(checksum_path),
        "split_method": waveform_manifest["split_method"],
        "patient_disjoint_verified": bool(waveform_manifest["patient_disjoint_verified"]),
        "selected_leads": list(leads),
        "selected_lead_indices": list(selected_indices),
        "source_sampling_rate_hz": args.source_rate,
        "sampling_rate_hz": args.output_rate,
        "samples_per_record": args.output_samples,
        "window_samples": args.window_samples,
        "crop_starts": list(args.crop_starts),
        "coordinate_mapping": "round_half_up(source_sample*128/500), clipped to [0,1279]",
        "event_names": list(EVENT_NAMES),
        "max_events_per_type": args.max_events,
        "eligibility": {
            "record_lead_requires_at_least_two_complete_qrs_onset_r_peak_qrs_offset_triples": True,
            "p_wave_supervision_disabled_for_afib_or_aflt_records": True,
            "missing_or_malformed_wave_types_are_ignored_not_forced_to_background": True,
        },
        "download": {
            "allow_incomplete_download": bool(args.allow_incomplete_download),
            "checksums_verified_for_parsed_selected_lead_files": bool(args.verify_checksums),
            "declared_all_fiducial_files": sum(
                relative.startswith("fiducial_points/ecgdeli/") for relative in declared
            ),
            "unavailable_selected_record_leads": len(unavailable),
            "unavailable_counts": unavailable_counts,
            "unavailable": unavailable,
        },
        "splits": split_summaries,
        "split_and_eligibility_hash": assignment_hash,
        "unmatched_annotation_note_counts": unmatched_totals,
        "output_sha256": output_hashes,
        "command": shlex.join(sys.argv),
    }
    _atomic_json(output_dir / "dataset_manifest.json", manifest)
    print(f"PTB-XL+ delineation sidecar complete: {output_dir}", flush=True)
    print(f"split_and_eligibility_hash={assignment_hash}", flush=True)
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waveform_root", type=Path, required=True)
    parser.add_argument("--ptbxl_root", type=Path, required=True)
    parser.add_argument("--ptbxl_plus_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--leads", nargs="+", default=list(LIMB_LEADS))
    parser.add_argument("--source_rate", type=int, default=500)
    parser.add_argument("--output_rate", type=int, default=128)
    parser.add_argument("--output_samples", type=int, default=1280)
    parser.add_argument("--window_samples", type=int, default=512)
    parser.add_argument("--crop_starts", nargs="+", type=int, default=[0, 384, 768])
    parser.add_argument("--max_events", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--allow_incomplete_download", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--verify_checksums", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
