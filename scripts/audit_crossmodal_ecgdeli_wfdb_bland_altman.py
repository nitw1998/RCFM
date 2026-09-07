#!/usr/bin/env python3
"""Run ECGdeli-port and WFDB ecgpuwave Bland--Altman sensitivity audits.

The two delineators receive the same NeuroKit2 R peaks.  Short phase-corrected
windows are reflect-padded to eight seconds for delineation, while measurements
are retained only for fiducials whose R peak lies in the original 480 samples.
ECGdeli results are explicitly an unofficial-port sensitivity analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_crossmodal_delineation_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import scipy
import wfdb

try:
    from scripts.audit_wesad_ecgdeli_port_delineation import decode_fpt_samples
    from scripts.audit_wesad_ecgpuwave_delineation import parse_ecgpuwave_annotations
except ModuleNotFoundError:  # Direct ``python scripts/...py`` entry point.
    from audit_wesad_ecgdeli_port_delineation import decode_fpt_samples
    from audit_wesad_ecgpuwave_delineation import parse_ecgpuwave_annotations


PARAMETERS = ("heart_rate_bpm", "rr_ms", "pr_ms", "qrs_ms", "qt_ms")
AMPLITUDES = ("p_amplitude", "r_amplitude", "t_amplitude")
LABELS = {
    "heart_rate_bpm": "Heart rate (bpm)", "rr_ms": "RR interval (ms)",
    "pr_ms": "PR interval (ms)", "qrs_ms": "QRS duration (ms)",
    "qt_ms": "QT interval (ms)", "p_amplitude": "P amplitude (normalized)",
    "r_amplitude": "R amplitude (normalized)", "t_amplitude": "T amplitude (normalized)",
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


def _reference_fpt(ecgdeli: object, r_peaks: np.ndarray) -> object:
    raw = np.zeros((len(r_peaks), 13), dtype=np.float64)
    raw[:, 5] = r_peaks + 1  # MATLAB FPT input is one-based.
    return ecgdeli.FPT.from_matlab(raw)


def _common_preprocess(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    target_length = max(len(signal), int(np.ceil(8.0 * fs)))
    total = target_length - len(signal)
    left = total // 2
    padded = np.pad(np.asarray(signal, dtype=np.float64), (left, total - left), mode="reflect")
    cleaned = nk.ecg_clean(padded, sampling_rate=fs, method="neurokit")
    _, peak_info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
    r_peaks = np.asarray(peak_info.get("ECG_R_Peaks", []), dtype=np.int64)
    if len(r_peaks) < 3:
        raise ValueError("fewer_than_three_common_r_peaks")
    return padded, cleaned, r_peaks, left


def _ecgdeli_rows(signal: np.ndarray, fs: float, r_peaks: np.ndarray, left: int) -> list[dict[str, int]]:
    import ecgdeli

    ecgdeli.set_fidelity("fixed")
    padded = np.pad(signal, (left, max(0, int(np.ceil(8 * fs)) - len(signal) - left)), mode="reflect")
    filtered = ecgdeli.filter_ecg(padded, fs, notch=50)
    annotation = ecgdeli.annotate(filtered, fs, reference=_reference_fpt(ecgdeli, r_peaks))
    return decode_fpt_samples(annotation.leads[0].samples, pad=left, signal_length=len(signal))


def _ecgpuwave_rows(
    cleaned: np.ndarray, fs: float, r_peaks: np.ndarray, left: int, signal_length: int,
    executable: str,
) -> list[dict[str, int]]:
    with tempfile.TemporaryDirectory(prefix="rcfm_ecgpuwave_crossmodal_") as temporary:
        root = Path(temporary)
        wfdb.wrsamp("record", fs=fs, units=["normalized"], sig_name=["ECG"],
                    p_signal=cleaned[:, None], write_dir=str(root))
        wfdb.wrann("record", "rpk", sample=r_peaks, symbol=["N"] * len(r_peaks),
                   write_dir=str(root))
        environment = dict(os.environ)
        environment["WFDB"] = "."
        completed = subprocess.run(
            [executable, "-r", "record", "-i", "rpk", "-a", "epu"], cwd=root,
            env=environment, check=False, capture_output=True, text=True,
        )
        if completed.returncode:
            raise RuntimeError(f"ecgpuwave_exit_{completed.returncode}:{completed.stderr.strip()}")
        annotation = wfdb.rdann(str(root / "record"), "epu")
        return parse_ecgpuwave_annotations(
            annotation.sample, annotation.symbol, annotation.num,
            pad_samples=left, signal_length=signal_length,
        )


def _mean_or_none(values: list[float]) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.mean(finite)) if len(finite) else None


def summarize_fiducials(
    fiducials: list[dict[str, int]], signal: np.ndarray, fs: float, *, afib: bool,
) -> dict[str, object]:
    """Return raw and point-order/physiology-QC window measurements."""
    raw: dict[str, list[float]] = {name: [] for name in (*PARAMETERS, *AMPLITUDES)}
    qc: dict[str, list[float]] = {name: [] for name in (*PARAMETERS, *AMPLITUDES)}
    r_peaks = sorted({int(row["r_peak"]) for row in fiducials if int(row["r_peak"]) >= 0})
    rr = np.diff(r_peaks) * 1000.0 / fs
    raw["rr_ms"].extend(rr.tolist())
    raw["heart_rate_bpm"].extend((60000.0 / rr[rr > 0]).tolist())
    plausible_rr = rr[(rr >= 300.0) & (rr <= 2000.0)]
    qc["rr_ms"].extend(plausible_rr.tolist())
    qc["heart_rate_bpm"].extend((60000.0 / plausible_rr).tolist())

    valid_p = valid_qrs = valid_t = 0
    for beat in fiducials:
        p_on, p_peak, p_off = (int(beat[key]) for key in ("p_onset", "p_peak", "p_offset"))
        q_on, r_peak, q_off = (int(beat[key]) for key in ("qrs_onset", "r_peak", "qrs_offset"))
        t_peak, t_off = (int(beat[key]) for key in ("t_peak", "t_offset"))
        p_order = 0 <= p_on < p_peak < p_off < q_on < r_peak
        qrs_order = 0 <= q_on < r_peak < q_off
        t_order = qrs_order and q_off < t_peak < t_off < len(signal)
        valid_p += int(p_order)
        valid_qrs += int(qrs_order)
        valid_t += int(t_order)

        if p_on >= 0 and q_on > p_on and not afib:
            value = (q_on - p_on) * 1000.0 / fs
            raw["pr_ms"].append(value)
            if p_order and 80.0 <= value <= 220.0:
                qc["pr_ms"].append(value)
        if q_on >= 0 and q_off > q_on:
            value = (q_off - q_on) * 1000.0 / fs
            raw["qrs_ms"].append(value)
            if qrs_order and 60.0 <= value <= 180.0:
                qc["qrs_ms"].append(value)
        if q_on >= 0 and t_off > q_on:
            value = (t_off - q_on) * 1000.0 / fs
            raw["qt_ms"].append(value)
            if t_order and 200.0 <= value <= 600.0:
                qc["qt_ms"].append(value)
        if p_order and not afib:
            baseline = float(np.median(signal[p_off:q_on])) if p_off < q_on else float(signal[p_on])
            raw["p_amplitude"].append(float(signal[p_peak] - baseline))
            qc["p_amplitude"].append(float(signal[p_peak] - baseline))
        if qrs_order:
            value = float(signal[r_peak] - np.median(signal[q_on:q_off + 1]))
            raw["r_amplitude"].append(value)
            qc["r_amplitude"].append(value)
        if t_order:
            baseline = float(np.median(signal[q_off:t_off + 1]))
            raw["t_amplitude"].append(float(signal[t_peak] - baseline))
            qc["t_amplitude"].append(float(signal[t_peak] - baseline))

    result: dict[str, object] = {
        "beats": len(fiducials), "p_order_valid_fraction": valid_p / len(fiducials) if fiducials else None,
        "qrs_order_valid_fraction": valid_qrs / len(fiducials) if fiducials else None,
        "t_order_valid_fraction": valid_t / len(fiducials) if fiducials else None,
    }
    for mode, values in (("raw", raw), ("qc", qc)):
        for parameter, items in values.items():
            result[f"{mode}_{parameter}"] = _mean_or_none(items)
            result[f"{mode}_{parameter}_beats"] = len(items)
    return result


def _worker(task: tuple[object, ...]) -> dict[str, object]:
    dataset, seed, index, source, signal, fs, group, record, afib, executable = task
    base: dict[str, object] = {
        "dataset": dataset, "seed": seed, "window_index": index, "source": source,
        "group_id": group, "record_id": record, "afib": bool(afib),
    }
    try:
        padded, cleaned, r_peaks, left = _common_preprocess(np.asarray(signal), float(fs))
    except Exception as error:
        return {**base, "algorithm": "common_r_peaks", "success": False,
                "failure_reason": f"{type(error).__name__}:{error}"}
    outputs = []
    for algorithm in ("ecgdeli_port_fixed", "wfdb_ecgpuwave_1.3.4"):
        try:
            if algorithm == "ecgdeli_port_fixed":
                rows = _ecgdeli_rows(np.asarray(signal), float(fs), r_peaks, left)
            else:
                rows = _ecgpuwave_rows(cleaned, float(fs), r_peaks, left, len(signal), str(executable))
            outputs.append({**base, "algorithm": algorithm, "success": bool(rows),
                            "failure_reason": None if rows else "no_center_beats",
                            **summarize_fiducials(rows, np.asarray(signal), float(fs), afib=bool(afib))})
        except Exception as error:
            outputs.append({**base, "algorithm": algorithm, "success": False,
                            "failure_reason": f"{type(error).__name__}:{error}"})
    return {"outputs": outputs}


def _group_rows(window_rows: list[dict[str, object]], seeds: list[int]) -> list[dict[str, object]]:
    result = []
    algorithms = sorted({str(row["algorithm"]) for row in window_rows if row.get("success")})
    groups = sorted({str(row["group_id"]) for row in window_rows})
    for algorithm in algorithms:
        for seed in seeds:
            for group in groups:
                for mode in ("raw", "qc"):
                    row: dict[str, object] = {"algorithm": algorithm, "seed": seed,
                                              "group_id": group, "mode": mode}
                    for source in ("reference", "generated"):
                        take = [item for item in window_rows if item.get("success") and
                                item["algorithm"] == algorithm and item["seed"] == seed and
                                item["group_id"] == group and item["source"] == source]
                        row[f"{source}_windows"] = len(take)
                        for parameter in (*PARAMETERS, *AMPLITUDES):
                            values = [float(item[f"{mode}_{parameter}"]) for item in take
                                      if item.get(f"{mode}_{parameter}") is not None]
                            row[f"{source}_{parameter}"] = _mean_or_none(values)
                    result.append(row)
    return result


def _agreement(group_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    keys = sorted({(str(row["algorithm"]), int(row["seed"]), str(row["mode"])) for row in group_rows})
    for algorithm, seed, mode in keys:
        take = [row for row in group_rows if row["algorithm"] == algorithm and
                row["seed"] == seed and row["mode"] == mode]
        for parameter in (*PARAMETERS, *AMPLITUDES):
            pairs = [(float(row[f"reference_{parameter}"]), float(row[f"generated_{parameter}"]))
                     for row in take if row.get(f"reference_{parameter}") is not None and
                     row.get(f"generated_{parameter}") is not None]
            difference = np.asarray([right - left for left, right in pairs], dtype=float)
            bias = float(np.mean(difference)) if len(difference) else None
            sd = float(np.std(difference, ddof=1)) if len(difference) > 1 else None
            output.append({"algorithm": algorithm, "seed": seed, "mode": mode,
                           "parameter": parameter, "paired_groups": len(pairs), "bias": bias,
                           "sd": sd, "lower_loa": bias - 1.96 * sd if sd is not None else None,
                           "upper_loa": bias + 1.96 * sd if sd is not None else None})
    return output


def _plot(output: Path, group_rows: list[dict[str, object]], algorithm: str,
          mode: str, parameters: tuple[str, ...], stem: str) -> None:
    columns = 5 if len(parameters) == 5 else 3
    figure, axes = plt.subplots(1, columns, figsize=(15.0, 3.2) if columns == 5 else (10.0, 3.2),
                                constrained_layout=True)
    for axis, parameter in zip(np.atleast_1d(axes), parameters):
        for seed in sorted({int(row["seed"]) for row in group_rows}):
            take = [row for row in group_rows if row["algorithm"] == algorithm and
                    row["seed"] == seed and row["mode"] == mode and
                    row.get(f"reference_{parameter}") is not None and
                    row.get(f"generated_{parameter}") is not None]
            real = np.asarray([row[f"reference_{parameter}"] for row in take], dtype=float)
            generated = np.asarray([row[f"generated_{parameter}"] for row in take], dtype=float)
            if not len(real):
                continue
            means, differences = (real + generated) / 2, generated - real
            bias = float(np.mean(differences))
            sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
            color = SEED_COLORS.get(seed, None)
            axis.scatter(means, differences, s=8, alpha=0.55, color=color, label=f"seed {seed}")
            axis.axhline(bias, color=color, linewidth=0.7)
            axis.axhline(bias + 1.96 * sd, color=color, linewidth=0.45, linestyle="--")
            axis.axhline(bias - 1.96 * sd, color=color, linewidth=0.45, linestyle="--")
        axis.set_title(LABELS[parameter], fontsize=7)
        axis.set_xlabel("Pair mean", fontsize=6)
        axis.set_ylabel("Generated - reference", fontsize=6)
        axis.tick_params(labelsize=5)
        axis.grid(alpha=0.15, linewidth=0.4)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=5)
    title = "ECGdeli unofficial port (fixed)" if algorithm.startswith("ecgdeli") else "WFDB ecgpuwave 1.3.4"
    figure.suptitle(f"{title}: {mode.upper()} group-level Bland-Altman", fontsize=8)
    figure.savefig(output / f"{stem}.png", dpi=600)
    figure.savefig(output / f"{stem}.pdf")
    plt.close(figure)


def _parse_seed_path(value: str) -> tuple[int, Path]:
    seed, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("expected SEED=PATH")
    return int(seed), Path(path)


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    ecgpuwave = args.ecgpuwave.resolve()
    if not ecgpuwave.is_file() or not os.access(ecgpuwave, os.X_OK):
        raise FileNotFoundError(f"ecgpuwave is not executable: {ecgpuwave}")
    seeds = [seed for seed, _ in args.seed_npz]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seeds")

    arrays = {}
    identities = None
    for seed, path in args.seed_npz:
        path = path.resolve()
        with np.load(path, allow_pickle=False) as artifact:
            required = {"targets", "oracle_aligned_predictions", "subject_ids", "record_ids"}
            if not required <= set(artifact.files):
                raise ValueError(f"{path} missing keys {sorted(required - set(artifact.files))}")
            targets = np.asarray(artifact["targets"], dtype=np.float32)
            generated = np.asarray(artifact["oracle_aligned_predictions"], dtype=np.float32)
            subjects = np.asarray(artifact["subject_ids"]).astype(str)
            records = np.asarray(artifact["record_ids"]).astype(str)
            afib = np.asarray(artifact["afib_labels"], dtype=bool) if "afib_labels" in artifact else np.zeros(len(targets), bool)
        if targets.shape != generated.shape or targets.ndim != 3 or targets.shape[1:] != (1, 480):
            raise ValueError(f"invalid phase array shape in {path}: {targets.shape}/{generated.shape}")
        identity = (subjects.tolist(), records.tolist())
        if identities is not None and identity != identities:
            raise ValueError("seed artifacts do not share ordered identities")
        identities = identity
        arrays[seed] = (path, targets, generated, subjects, records, afib)
    first_seed = seeds[0]
    _, targets, _, subjects, records, afib = arrays[first_seed]
    group_ids = records if args.dataset == "mimic_afib" else subjects

    tasks: list[tuple[object, ...]] = []
    for index in range(len(targets)):
        tasks.append((args.dataset, first_seed, index, "reference", targets[index, 0],
                      args.sampling_rate, group_ids[index], records[index], afib[index], str(ecgpuwave)))
    for seed in seeds:
        generated = arrays[seed][2]
        for index in range(len(generated)):
            tasks.append((args.dataset, seed, index, "generated", generated[index, 0],
                          args.sampling_rate, group_ids[index], records[index], afib[index], str(ecgpuwave)))

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for number, result in enumerate(executor.map(_worker, tasks, chunksize=8), start=1):
            outputs = result.get("outputs")
            if outputs is None:
                failures.append(result)
            else:
                for row in outputs:
                    (rows if row.get("success") else failures).append(row)
            if number % 1000 == 0:
                print(f"completed {number}/{len(tasks)} signal tasks", flush=True)

    reference = [row for row in rows if row["source"] == "reference"]
    for seed in seeds[1:]:
        rows.extend([{**row, "seed": seed} for row in reference])
    rows.sort(key=lambda row: (str(row["algorithm"]), int(row["seed"]), int(row["window_index"]), str(row["source"])))
    _write_csv(output / "per_window_delineation_parameters.csv", rows)
    _write_csv(output / "delineation_failures.csv", failures)
    grouped = _group_rows(rows, seeds)
    _write_csv(output / "per_group_parameter_means.csv", grouped)
    agreement = _agreement(grouped)
    _write_csv(output / "bland_altman_summary.csv", agreement)
    for algorithm, prefix in (("ecgdeli_port_fixed", "ecgdeli_port_fixed"),
                              ("wfdb_ecgpuwave_1.3.4", "wfdb_ecgpuwave")):
        for mode in ("raw", "qc"):
            _plot(output, grouped, algorithm, mode, PARAMETERS, f"{prefix}_timing_bland_altman_{mode}")
        _plot(output, grouped, algorithm, "qc", AMPLITUDES, f"{prefix}_amplitude_bland_altman_qc")

    summary = {
        "schema_version": 1, "status": "complete", "dataset": args.dataset,
        "sampling_rate_hz": args.sampling_rate, "seeds": seeds,
        "windows": len(targets), "aggregation": "source_record" if args.dataset == "mimic_afib" else "subject",
        "groups": len(set(group_ids)), "phase_mode": "target_informed_oracle_aligned_central_480",
        "common_r_policy": "NeuroKit2 neurokit-method R peaks supplied to both delineators",
        "short_window_policy": "reflect-pad to at least 8 s; retain only original-window R-centered beats",
        "p_wave_policy": "MIMIC-AFib AF-labelled windows excluded from P amplitude and PR",
        "ecgdeli": {"identity": "unofficial Python port, fixed fidelity", "port_commit": args.port_commit,
                     "upstream_matlab_commit": args.upstream_commit},
        "wfdb": {"identity": "WFDB ecgpuwave traditional independent baseline", "version": "1.3.4",
                 "executable": str(ecgpuwave), "sha256": _sha256(ecgpuwave)},
        "successful_window_rows": len(rows), "failed_rows": len(failures),
        "agreement": agreement,
        "interpretation_boundary": (
            "Algorithmic agreement sensitivity analysis, not expert fiducial ground truth or clinical equivalence. "
            "ECGdeli output is not an execution of the official MATLAB toolbox. Oracle alignment is not deployable."
        ),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "neurokit2": nk.__version__, "wfdb_python": wfdb.__version__},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "inputs": {str(seed): {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
                   for seed, path in args.seed_npz},
    }
    protocol["outputs"] = {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mimic_afib", "mmecg"), required=True)
    parser.add_argument("--sampling_rate", type=float, required=True)
    parser.add_argument("--seed_npz", action="append", type=_parse_seed_path, required=True,
                        help="Repeat as SEED=phase_predictions.npz")
    parser.add_argument("--ecgpuwave", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--port_commit", default="313d6dea54d88c8a616edce820c2ab9bc2b1deec")
    parser.add_argument("--upstream_commit", default="c3738612771264e4d6c4686898ef7b2d6a700ad3")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
