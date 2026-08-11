"""Clinical ECG measurements with explicit units and applicability guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import neurokit2 as nk
import numpy as np


_PHYSICAL_AMPLITUDE_UNITS = {"mV", "uV"}


def _indices(values: Iterable[float | int]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    array[~np.isfinite(array)] = -1
    return array.astype(np.int64)


@dataclass(frozen=True)
class ECGFiducials:
    r_peaks: np.ndarray
    p_onsets: np.ndarray
    p_peaks: np.ndarray
    p_offsets: np.ndarray
    qrs_onsets: np.ndarray
    qrs_offsets: np.ndarray
    t_peaks: np.ndarray
    t_offsets: np.ndarray

    def __post_init__(self) -> None:
        lengths = {len(np.asarray(getattr(self, name))) for name in self.__dataclass_fields__}
        if len(lengths) != 1:
            raise ValueError("all fiducial arrays must have the same beat count")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Iterable[float | int]]) -> "ECGFiducials":
        return cls(**{name: _indices(values[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class DelineationResult:
    success: bool
    fiducials: Optional[ECGFiducials]
    algorithm: str
    library: str
    library_version: str
    failure_reason: Optional[str] = None


def _aligned_wave_indices(waves: Mapping[str, Iterable[float]], key: str, count: int) -> np.ndarray:
    values = _indices(waves.get(key, []))
    if len(values) == count:
        return values
    padded = np.full(count, -1, dtype=np.int64)
    padded[: min(count, len(values))] = values[:count]
    return padded


def delineate_ecg(
    signal: Iterable[float],
    sampling_rate: float,
    method: str = "dwt",
    clean_method: str = "neurokit",
) -> DelineationResult:
    """Delineate one lead independently; failures are returned, never hidden."""

    algorithm = f"neurokit2.ecg_delineate:{method}"
    version = getattr(nk, "__version__", "unknown")
    try:
        array = np.asarray(signal, dtype=np.float64).reshape(-1)
        if array.size < max(8, int(round(0.5 * sampling_rate))):
            raise ValueError("signal_too_short")
        if not np.all(np.isfinite(array)):
            raise ValueError("signal_contains_non_finite_values")
        cleaned = nk.ecg_clean(array, sampling_rate=sampling_rate, method=clean_method)
        _, peak_info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method="neurokit")
        r_peaks = _indices(peak_info.get("ECG_R_Peaks", []))
        if len(r_peaks) < 2:
            raise ValueError("fewer_than_two_r_peaks")
        _, waves = nk.ecg_delineate(
            cleaned,
            rpeaks=r_peaks,
            sampling_rate=sampling_rate,
            method=method,
            show=False,
        )
        count = len(r_peaks)
        fiducials = ECGFiducials(
            r_peaks=r_peaks,
            p_onsets=_aligned_wave_indices(waves, "ECG_P_Onsets", count),
            p_peaks=_aligned_wave_indices(waves, "ECG_P_Peaks", count),
            p_offsets=_aligned_wave_indices(waves, "ECG_P_Offsets", count),
            qrs_onsets=_aligned_wave_indices(waves, "ECG_R_Onsets", count),
            qrs_offsets=_aligned_wave_indices(waves, "ECG_R_Offsets", count),
            t_peaks=_aligned_wave_indices(waves, "ECG_T_Peaks", count),
            t_offsets=_aligned_wave_indices(waves, "ECG_T_Offsets", count),
        )
        return DelineationResult(True, fiducials, algorithm, "neurokit2", version)
    except Exception as error:
        return DelineationResult(
            False,
            None,
            algorithm,
            "neurokit2",
            version,
            f"{type(error).__name__}:{error}",
        )


def _valid_index(index: int, signal_length: int) -> bool:
    return 0 <= int(index) < signal_length


def _duration_ms(start: int, end: int, sampling_rate: float) -> Optional[float]:
    if start < 0 or end <= start:
        return None
    return (end - start) * 1000.0 / sampling_rate


def _qtc_seconds(qt_seconds: float, rr_seconds: float, formula: str) -> float:
    if qt_seconds <= 0 or rr_seconds <= 0:
        raise ValueError("QT and RR must be positive")
    formula = formula.lower()
    if formula == "bazett":
        return qt_seconds / np.sqrt(rr_seconds)
    if formula == "fridericia":
        return qt_seconds / np.cbrt(rr_seconds)
    if formula == "framingham":
        return qt_seconds + 0.154 * (1.0 - rr_seconds)
    if formula == "hodges":
        heart_rate = 60.0 / rr_seconds
        return qt_seconds + 0.00175 * (heart_rate - 60.0)
    raise ValueError(f"Unsupported QTc formula: {formula}")


def _baseline(
    signal: np.ndarray,
    p_offset: int,
    qrs_onset: int,
    sampling_rate: float,
) -> Optional[float]:
    if qrs_onset <= 0:
        return None
    if p_offset >= 0 and p_offset < qrs_onset:
        start = p_offset
    else:
        start = max(0, qrs_onset - max(1, int(round(0.04 * sampling_rate))))
    if start >= qrs_onset:
        return None
    return float(np.median(signal[start:qrs_onset]))


def compute_hrv(
    rr_intervals_seconds: Iterable[float],
    sequence_duration_seconds: float,
    continuous: bool,
    minimum_duration_seconds: float = 30.0,
) -> dict[str, object]:
    rr = np.asarray(list(rr_intervals_seconds), dtype=np.float64)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if not continuous:
        return {"status": "blocked_non_continuous", "sdnn_ms": None, "rmssd_ms": None, "n_rr": int(rr.size)}
    if sequence_duration_seconds < minimum_duration_seconds:
        return {
            "status": "blocked_insufficient_duration",
            "minimum_duration_seconds": float(minimum_duration_seconds),
            "sequence_duration_seconds": float(sequence_duration_seconds),
            "sdnn_ms": None,
            "rmssd_ms": None,
            "n_rr": int(rr.size),
        }
    if rr.size < 2:
        return {"status": "blocked_insufficient_rr", "sdnn_ms": None, "rmssd_ms": None, "n_rr": int(rr.size)}
    return {
        "status": "ok",
        "sdnn_ms": float(np.std(rr, ddof=1) * 1000.0),
        "rmssd_ms": float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0),
        "n_rr": int(rr.size),
    }


def measure_ecg_parameters(
    signal: Iterable[float],
    sampling_rate: float,
    fiducials: ECGFiducials,
    amplitude_unit: str = "normalized",
    inverse_transformed: bool = False,
    qtc_formula: str = "fridericia",
    st_offset_ms: float = 60.0,
    continuous: bool = False,
    minimum_hrv_duration_seconds: float = 30.0,
    p_wave_applicable: bool = True,
    allow_normalized_amplitudes: bool = False,
) -> dict[str, object]:
    """Measure intervals and amplitudes from one independently delineated lead."""

    array = np.asarray(signal, dtype=np.float64).reshape(-1)
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if not np.all(np.isfinite(array)):
        raise ValueError("signal must contain only finite values")

    fields: dict[str, list[float]] = {
        "rr_ms": [],
        "pr_ms": [],
        "qrs_ms": [],
        "qt_ms": [],
        "qtc_ms": [],
        "p_amplitude": [],
        "r_amplitude": [],
        "t_amplitude": [],
        "st_deviation": [],
    }
    r_peaks = np.asarray(fiducials.r_peaks, dtype=np.int64)
    valid_r = r_peaks[(r_peaks >= 0) & (r_peaks < len(array))]
    rr_seconds = np.diff(valid_r) / sampling_rate
    fields["rr_ms"] = (rr_seconds * 1000.0).tolist()
    physical_amplitudes = inverse_transformed and amplitude_unit in _PHYSICAL_AMPLITUDE_UNITS
    normalized_amplitudes = (
        allow_normalized_amplitudes
        and not inverse_transformed
        and amplitude_unit == "normalized"
    )
    amplitudes_enabled = physical_amplitudes or normalized_amplitudes
    st_offset_samples = int(round(st_offset_ms * sampling_rate / 1000.0))

    for beat in range(len(r_peaks)):
        p_onset = int(fiducials.p_onsets[beat])
        p_peak = int(fiducials.p_peaks[beat])
        p_offset = int(fiducials.p_offsets[beat])
        qrs_onset = int(fiducials.qrs_onsets[beat])
        r_peak = int(r_peaks[beat])
        qrs_offset = int(fiducials.qrs_offsets[beat])
        t_peak = int(fiducials.t_peaks[beat])
        t_offset = int(fiducials.t_offsets[beat])

        pr = _duration_ms(p_onset, qrs_onset, sampling_rate)
        qrs = _duration_ms(qrs_onset, qrs_offset, sampling_rate)
        qt = _duration_ms(qrs_onset, t_offset, sampling_rate)
        if p_wave_applicable and pr is not None:
            fields["pr_ms"].append(pr)
        if qrs is not None:
            fields["qrs_ms"].append(qrs)
        if qt is not None:
            fields["qt_ms"].append(qt)
            if beat > 0 and _valid_index(r_peaks[beat - 1], len(array)) and _valid_index(r_peak, len(array)):
                rr_for_beat = (r_peak - int(r_peaks[beat - 1])) / sampling_rate
                fields["qtc_ms"].append(_qtc_seconds(qt / 1000.0, rr_for_beat, qtc_formula) * 1000.0)

        if not amplitudes_enabled:
            continue
        baseline = _baseline(
            array,
            p_offset if p_wave_applicable else -1,
            qrs_onset,
            sampling_rate,
        )
        if baseline is None:
            continue
        if p_wave_applicable and _valid_index(p_peak, len(array)):
            fields["p_amplitude"].append(float(array[p_peak] - baseline))
        if _valid_index(r_peak, len(array)):
            fields["r_amplitude"].append(float(array[r_peak] - baseline))
        if _valid_index(t_peak, len(array)):
            fields["t_amplitude"].append(float(array[t_peak] - baseline))
        st_index = qrs_offset + st_offset_samples
        if _valid_index(qrs_offset, len(array)) and _valid_index(st_index, len(array)):
            fields["st_deviation"].append(float(array[st_index] - baseline))

    hrv = compute_hrv(
        rr_seconds,
        sequence_duration_seconds=len(array) / sampling_rate,
        continuous=continuous,
        minimum_duration_seconds=minimum_hrv_duration_seconds,
    )
    if physical_amplitudes:
        amplitude_status = "ok_physical_unit"
    elif normalized_amplitudes:
        amplitude_status = "exploratory_normalized_units_not_physical"
    else:
        amplitude_status = "blocked_requires_inverse_physical_units"
    if not p_wave_applicable:
        p_wave_status = "not_applicable_by_record_policy"
    elif fields["pr_ms"]:
        p_wave_status = "ok"
    else:
        p_wave_status = "not_applicable_or_not_delineated"
    return {
        "sampling_rate_hz": float(sampling_rate),
        "amplitude_unit": amplitude_unit,
        "amplitude_status": amplitude_status,
        "amplitude_claim_allowed": physical_amplitudes,
        "amplitude_definition": "signed sample value minus pre-QRS isoelectric median",
        "st_reference": "pre-QRS isoelectric median",
        "st_offset_ms": float(st_offset_ms),
        "qtc_formula": qtc_formula.lower(),
        "p_wave_status": p_wave_status,
        "parameters": fields,
        "hrv": hrv,
    }
