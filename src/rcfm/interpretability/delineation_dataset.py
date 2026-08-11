"""PTB-XL+ fiducial parsing and stride-1 delineation training data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


LIMB_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF")
PTBXL_LEAD_ORDER = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
WAVE_NAMES = ("p", "qrs", "t")
EVENT_NAMES = (
    "p_onset",
    "p_peak",
    "p_offset",
    "qrs_onset",
    "r_peak",
    "qrs_offset",
    "t_onset",
    "t_peak",
    "t_offset",
)
EVENT_AUX_NOTES = {
    "p_onset": "p-wave onset",
    "p_peak": "p-wave peak",
    "p_offset": "p-wave offset",
    "qrs_onset": "QRS onset",
    "r_peak": "R peak",
    "qrs_offset": "QRS offset",
    "t_onset": "t-wave onset",
    "t_peak": "t-wave peak",
    "t_offset": "t-wave offset",
}
WAVE_EVENT_INDICES = ((0, 1, 2), (3, 4, 5), (6, 7, 8))


def annotation_sample_to_output(
    sample: int,
    source_rate_hz: int = 500,
    output_rate_hz: int = 128,
    output_samples: int = 1280,
) -> int:
    """Map a WFDB sample time to the nearest output-grid sample."""

    if sample < 0 or source_rate_hz <= 0 or output_rate_hz <= 0 or output_samples <= 0:
        raise ValueError("sample and sampling-grid arguments must be positive")
    mapped = int(np.floor(float(sample) * output_rate_hz / source_rate_hz + 0.5))
    return min(mapped, output_samples - 1)


def _match_triplets(
    onsets: Sequence[int], peaks: Sequence[int], offsets: Sequence[int]
) -> list[tuple[int, int, int]]:
    """Greedily match non-overlapping physiological onset/peak/offset triples."""

    onset_values = sorted(int(value) for value in onsets)
    peak_values = sorted(int(value) for value in peaks)
    offset_values = sorted(int(value) for value in offsets)
    matched: list[tuple[int, int, int]] = []
    onset_index = 0
    offset_index = 0
    previous_offset = -1
    for peak in peak_values:
        while onset_index < len(onset_values) and onset_values[onset_index] <= previous_offset:
            onset_index += 1
        candidates: list[int] = []
        while onset_index < len(onset_values) and onset_values[onset_index] < peak:
            candidates.append(onset_values[onset_index])
            onset_index += 1
        if not candidates:
            continue
        onset = candidates[-1]
        while offset_index < len(offset_values) and offset_values[offset_index] <= peak:
            offset_index += 1
        if offset_index >= len(offset_values):
            break
        offset = offset_values[offset_index]
        if onset_index < len(onset_values) and onset_values[onset_index] < offset:
            continue
        offset_index += 1
        matched.append((onset, peak, offset))
        previous_offset = offset
    return matched


def parse_fiducial_events(
    samples: Sequence[int],
    aux_notes: Sequence[str],
    *,
    source_rate_hz: int = 500,
    output_rate_hz: int = 128,
    output_samples: int = 1280,
    max_events: int = 64,
    disable_p_wave: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Convert ECGdeli auxiliary notes into aligned complete wave triples."""

    if len(samples) != len(aux_notes):
        raise ValueError("annotation samples and auxiliary notes must have equal length")
    if max_events <= 0:
        raise ValueError("max_events must be positive")
    note_samples: dict[str, list[int]] = {name: [] for name in EVENT_NAMES}
    note_to_event = {note: name for name, note in EVENT_AUX_NOTES.items()}
    for sample, note in zip(samples, aux_notes):
        event = note_to_event.get(str(note).strip())
        if event is not None:
            note_samples[event].append(int(sample))

    positions = np.full((len(EVENT_NAMES), max_events), -1, dtype=np.int16)
    counts = np.zeros(len(EVENT_NAMES), dtype=np.uint8)
    wave_valid = np.zeros(len(WAVE_NAMES), dtype=np.uint8)
    unmatched: dict[str, int] = {}
    for wave_index, event_indices in enumerate(WAVE_EVENT_INDICES):
        names = [EVENT_NAMES[index] for index in event_indices]
        triples = _match_triplets(*(note_samples[name] for name in names))
        if len(triples) > max_events:
            raise ValueError(f"{WAVE_NAMES[wave_index]} has {len(triples)} events; max={max_events}")
        for event_offset, event_index in enumerate(event_indices):
            mapped = [
                annotation_sample_to_output(
                    triple[event_offset], source_rate_hz, output_rate_hz, output_samples
                )
                for triple in triples
            ]
            if mapped:
                positions[event_index, : len(mapped)] = np.asarray(mapped, dtype=np.int16)
                counts[event_index] = len(mapped)
            unmatched[EVENT_NAMES[event_index]] = len(note_samples[EVENT_NAMES[event_index]]) - len(triples)
        wave_valid[wave_index] = int(len(triples) >= 2)
    if disable_p_wave:
        wave_valid[0] = 0
    return positions, counts, wave_valid, unmatched


def build_window_targets(
    positions: np.ndarray,
    counts: np.ndarray,
    wave_valid: np.ndarray,
    *,
    crop_start: int,
    window_samples: int = 512,
    heatmap_sigma_samples: float = 2.0,
    heatmap_edge_ignore_samples: int = 8,
) -> dict[str, np.ndarray]:
    """Build soft-mask supervision from sparse full-record fiducials."""

    event_positions = np.asarray(positions)
    event_counts = np.asarray(counts)
    valid_waves = np.asarray(wave_valid)
    if event_positions.ndim != 2 or event_positions.shape[0] != len(EVENT_NAMES):
        raise ValueError("positions must have shape (9, max_events)")
    if event_counts.shape != (len(EVENT_NAMES),) or valid_waves.shape != (len(WAVE_NAMES),):
        raise ValueError("invalid fiducial count or wave-valid shape")
    if crop_start < 0 or window_samples <= 0 or heatmap_sigma_samples <= 0:
        raise ValueError("invalid crop or heatmap arguments")
    crop_end = crop_start + window_samples
    regions = np.zeros((len(WAVE_NAMES), window_samples), dtype=np.float32)
    region_mask = np.zeros_like(regions)
    heatmaps = np.zeros((len(EVENT_NAMES), window_samples), dtype=np.float32)
    heatmap_mask = np.zeros_like(heatmaps)
    grid = np.arange(window_samples, dtype=np.float32)

    for wave_index, event_indices in enumerate(WAVE_EVENT_INDICES):
        if not valid_waves[wave_index]:
            continue
        onset_index, _peak_index, offset_index = event_indices
        triple_count = min(int(event_counts[index]) for index in event_indices)
        region_mask[wave_index] = 1.0
        for triple_index in range(triple_count):
            onset = int(event_positions[onset_index, triple_index])
            offset = int(event_positions[offset_index, triple_index])
            if onset < 0 or offset < onset:
                continue
            left = max(onset, crop_start)
            right = min(offset + 1, crop_end)
            if left < right:
                local_left = left - crop_start
                local_right = right - crop_start
                regions[wave_index, local_left:local_right] = 1.0
                if onset < crop_start or offset >= crop_end:
                    region_mask[wave_index, local_left:local_right] = 0.0
        for event_index in event_indices:
            heatmap_mask[event_index] = 1.0
            for event_number in range(int(event_counts[event_index])):
                sample = int(event_positions[event_index, event_number]) - crop_start
                if 0 <= sample < window_samples:
                    gaussian = np.exp(
                        -0.5 * ((grid - float(sample)) / heatmap_sigma_samples) ** 2
                    )
                    heatmaps[event_index] = np.maximum(heatmaps[event_index], gaussian)

    edge = min(max(int(heatmap_edge_ignore_samples), 0), window_samples // 2)
    if edge:
        heatmap_mask[:, :edge] = 0.0
        heatmap_mask[:, -edge:] = 0.0
    return {
        "regions": regions,
        "region_mask": region_mask,
        "heatmaps": heatmaps,
        "heatmap_mask": heatmap_mask,
    }


def robust_window_normalize(signal: np.ndarray, clip_z: float = 5.0) -> np.ndarray:
    """Return a robust affine-invariant window scaled to approximately [-1, 1]."""

    values = np.asarray(signal, dtype=np.float32)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or clip_z <= 0:
        raise ValueError("signal must be a finite vector and clip_z must be positive")
    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError("constant or near-constant ECG window")
    return (np.clip((values - median) / scale, -clip_z, clip_z) / clip_z).astype(np.float32)


class PTBXLPlusDelineationDataset(Dataset):
    """Memory-mapped PTB-XL waveform plus sparse PTB-XL+ supervision."""

    def __init__(
        self,
        sidecar_root: str | Path,
        waveform_root: str | Path,
        split: str,
        *,
        window_samples: int = 512,
        crop_starts: Sequence[int] = (0, 384, 768),
        heatmap_sigma_samples: float = 2.0,
        heatmap_edge_ignore_samples: int = 8,
        normalization_clip_z: float = 5.0,
        max_records: int | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.sidecar_root = Path(sidecar_root)
        self.waveform_root = Path(waveform_root)
        self.split = split
        self.window_samples = int(window_samples)
        self.crop_starts = tuple(int(value) for value in crop_starts)
        self.heatmap_sigma_samples = float(heatmap_sigma_samples)
        self.heatmap_edge_ignore_samples = int(heatmap_edge_ignore_samples)
        self.normalization_clip_z = float(normalization_clip_z)
        manifest = json.loads((self.sidecar_root / "dataset_manifest.json").read_text())
        self.manifest: Mapping[str, object] = manifest
        self.lead_names = tuple(str(value) for value in manifest["selected_leads"])
        self.lead_indices = tuple(int(value) for value in manifest["selected_lead_indices"])
        self.waveforms = np.load(
            self.waveform_root / f"X_{split}_resampled.npy", mmap_mode="r", allow_pickle=False
        )
        waveform_ids = np.load(self.waveform_root / f"record_ids_{split}.npy", allow_pickle=False)
        sidecar_ids = np.load(self.sidecar_root / f"record_ids_{split}.npy", allow_pickle=False)
        if not np.array_equal(waveform_ids, sidecar_ids):
            raise ValueError(f"{split} waveform and fiducial record IDs do not align")
        if self.waveforms.shape[0] != len(sidecar_ids) or self.waveforms.shape[1] < self.window_samples:
            raise ValueError("waveform array does not match sidecar records/window")
        self.positions = np.load(
            self.sidecar_root / f"fiducial_positions_{split}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.counts = np.load(
            self.sidecar_root / f"fiducial_counts_{split}.npy", mmap_mode="r", allow_pickle=False
        )
        self.wave_valid = np.load(
            self.sidecar_root / f"wave_valid_{split}.npy", mmap_mode="r", allow_pickle=False
        )
        self.eligible = np.load(
            self.sidecar_root / f"eligible_leads_{split}.npy", mmap_mode="r", allow_pickle=False
        ).astype(bool)
        record_limit = len(sidecar_ids) if max_records is None else min(int(max_records), len(sidecar_ids))
        if record_limit <= 0:
            raise ValueError("max_records must leave at least one record")
        record_leads = np.argwhere(self.eligible[:record_limit])
        samples = [
            (int(record_index), int(lead_index), int(crop_start))
            for record_index, lead_index in record_leads
            for crop_start in self.crop_starts
            if crop_start >= 0 and crop_start + self.window_samples <= self.waveforms.shape[1]
        ]
        if not samples:
            raise ValueError(f"{split} contains no eligible delineation windows")
        self.samples = np.asarray(samples, dtype=np.int32)

    def __len__(self) -> int:
        return int(len(self.samples))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, selected_lead_index, crop_start = self.samples[index].tolist()
        waveform_lead_index = self.lead_indices[selected_lead_index]
        signal = np.asarray(
            self.waveforms[
                record_index,
                crop_start : crop_start + self.window_samples,
                waveform_lead_index,
            ],
            dtype=np.float32,
        )
        normalized = robust_window_normalize(signal, self.normalization_clip_z)
        targets = build_window_targets(
            self.positions[record_index, selected_lead_index],
            self.counts[record_index, selected_lead_index],
            self.wave_valid[record_index, selected_lead_index],
            crop_start=crop_start,
            window_samples=self.window_samples,
            heatmap_sigma_samples=self.heatmap_sigma_samples,
            heatmap_edge_ignore_samples=self.heatmap_edge_ignore_samples,
        )
        return {
            "signal": torch.from_numpy(normalized[None, :]),
            "regions": torch.from_numpy(targets["regions"]),
            "region_mask": torch.from_numpy(targets["region_mask"]),
            "heatmaps": torch.from_numpy(targets["heatmaps"]),
            "heatmap_mask": torch.from_numpy(targets["heatmap_mask"]),
            "record_index": torch.tensor(record_index, dtype=torch.int64),
            "lead_index": torch.tensor(selected_lead_index, dtype=torch.int64),
            "crop_start": torch.tensor(crop_start, dtype=torch.int64),
        }
