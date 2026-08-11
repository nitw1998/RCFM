"""Checkpoint contract for the independent CATransformer reproduction."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Mapping

import torch


CAT_CHECKPOINT_KIND = "independent_catransformer_reproduction"


def validate_cat_checkpoint(payload: Mapping[str, object]) -> None:
    required = {
        "schema_version", "kind", "epoch", "global_step", "model_state",
        "optimizer_state", "scaler_state", "config", "normalization", "output_spec",
        "rng_states", "data_loader_generator_state", "provenance",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("CAT checkpoint is missing: " + ", ".join(missing))
    if payload["schema_version"] != 1 or payload["kind"] != CAT_CHECKPOINT_KIND:
        raise ValueError("invalid CAT checkpoint schema or kind")
    config = payload["config"]
    if not isinstance(config, Mapping) or config.get("reproduction_label") != "CAT-PPG (reproduced)":
        raise ValueError("CAT checkpoint must retain the independent reproduction label")
    if config.get("cycle_source") != "source_ppg_fft_only" or int(config.get("nfe", -1)) != 1:
        raise ValueError("CAT checkpoint violates source-only deterministic inference")
    output = payload["output_spec"]
    if not isinstance(output, Mapping) or int(output.get("channels", 0)) != 1 or int(output.get("length", 0)) != 512:
        raise ValueError("MIMIC CAT checkpoint requires a single 512-sample ECG output")
    if int(payload["epoch"]) <= 0 or int(payload["global_step"]) <= 0:
        raise ValueError("CAT checkpoint epoch and global_step must be positive")
    rng_states = payload["rng_states"]
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(rng_states, Mapping) or required_rng - set(rng_states):
        raise ValueError("CAT checkpoint has incomplete RNG states")
    generator_state = payload["data_loader_generator_state"]
    if not torch.is_tensor(generator_state) or generator_state.dtype != torch.uint8:
        raise ValueError("CAT checkpoint has an invalid DataLoader generator state")
    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("CAT checkpoint provenance must be a mapping")
    required_provenance = {"git_commit", "command", "paper_doi", "implementation_status"}
    if missing_provenance := sorted(required_provenance - set(provenance)):
        raise ValueError("CAT checkpoint provenance is missing: " + ", ".join(missing_provenance))


def save_cat_checkpoint(payload: Mapping[str, object], path: Path) -> None:
    validate_cat_checkpoint(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_cat_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, object]:
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("CAT checkpoint must contain a mapping")
    validate_cat_checkpoint(payload)
    return dict(payload)
