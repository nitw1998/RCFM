"""Inference compatibility for the local ECGMambaFormer multi-task checkpoint."""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ECGMambaFormerInference(nn.Module):
    """Inference graph retaining the checkpoint's exact state-dict layout."""

    def __init__(self, encoder, diag_decoder, semantic_decoder, weighting):
        super().__init__()
        self.fusion_weight = weighting.fusion_weight
        self.register_buffer("attn_history", weighting.attn_history)
        self.register_buffer("history_ptr", weighting.history_ptr)
        self.task_backbone_attn = weighting.task_backbone_attn
        self.feat_grad_fusion = weighting.feat_grad_fusion
        self.encoder = encoder
        self.decoders = nn.ModuleDict(
            {"diag": diag_decoder, "semantic": semantic_decoder}
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return diagnostic logits before the checkpoint decoder's sigmoid."""

        features = self.encoder(inputs)
        decoder = self.decoders["diag"]
        values = decoder.c(features)
        values = torch.cat([decoder.avg_a(values), decoder.avg_m(values)], dim=1)
        return decoder.fc[:-1](values.flatten(1).contiguous())

    def semantic_probabilities(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoders["semantic"](self.encoder(inputs))


def diagnostic_class_names(scp_statements_path: Path) -> list[str]:
    """Return the sorted PTB-XL diagnostic-code order used by preprocessing."""

    with scp_statements_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "diagnostic" not in rows[0]:
        raise ValueError("SCP statement table does not contain diagnostic metadata")
    names = sorted(row[""] for row in rows if row["diagnostic"] == "1.0")
    if not names or len(names) != len(set(names)):
        raise ValueError("diagnostic class names must be nonempty and unique")
    return names


def record_global_zscore(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Match ECGMambaFormer's per-record normalization across time and leads."""

    waveform = np.asarray(values, dtype=np.float32)
    if waveform.ndim != 2 or not np.all(np.isfinite(waveform)):
        raise ValueError("values must be a finite (time, leads) array")
    mean = float(np.mean(waveform))
    scale = float(np.std(waveform))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("record has zero or invalid global variance")
    normalized = (waveform - mean) / (scale + 1e-8)
    return normalized.astype(np.float32, copy=False), mean, scale


def load_ecgmambaformer_fca_mgda(
    source_root: Path,
    checkpoint_path: Path,
    num_diagnostic_classes: int,
    map_location: str | torch.device = "cpu",
) -> ECGMambaFormerInference:
    """Build the local unpublished model and strictly restore its checkpoint."""

    source_root = source_root.resolve()
    required = [
        source_root / "LibMTL" / "model" / "ecgmamba_improved.py",
        source_root / "LibMTL" / "weighting" / "FCA_MGDA.py",
    ]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("ECGMambaFormer source files are incomplete")
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

    loaded = sys.modules.get("LibMTL")
    if loaded is not None:
        module_path = getattr(loaded, "__file__", None)
        if module_path is None or source_root not in Path(module_path).resolve().parents:
            raise RuntimeError("an unrelated LibMTL package is already imported")

    sys.path.insert(0, str(source_root))
    try:
        model_module = importlib.import_module("LibMTL.model.ecgmamba_improved")
        weighting_module = importlib.import_module("LibMTL.weighting.FCA_MGDA")
    finally:
        sys.path.remove(str(source_root))

    weighting = weighting_module.FCA_MGDA(
        feature_dim=512,
        num_tasks=2,
        num_heads=4,
        query_dim=64,
        use_gradient_info=True,
    )
    model = ECGMambaFormerInference(
        encoder=model_module.ECGMambaImproved_Encoder(use_mamba=True),
        diag_decoder=model_module.DiagDecoder(n_classes=num_diagnostic_classes),
        semantic_decoder=model_module.SegDecoder(),
        weighting=weighting,
    )
    state = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint must be a nonempty state dictionary")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint restoration reported incompatible keys")
    return model
