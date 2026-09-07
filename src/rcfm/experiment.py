"""Resolved experiment configuration, local records, and optional W&B logging."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


LOCAL_FILE_HEADERS = {
    "epoch_metrics.csv": ["epoch", "global_step", "metric", "value"],
    "ot_diagnostics.csv": ["epoch", "global_step", "metric", "value"],
    "validation_metrics.csv": ["epoch", "global_step", "metric", "value"],
    "clinical_metrics.csv": ["epoch", "metric", "status", "value", "valid_count", "failure_count"],
}


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


class RunArtifacts:
    """Maintain the non-W&B evidence files required for every run."""

    def __init__(
        self,
        run_dir: Path,
        resolved_config: Mapping[str, Any],
        run_metadata: Mapping[str, Any],
        environment: str,
        git_state: str,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.run_dir / "resolved_config.yaml", dict(resolved_config))
        atomic_json(self.run_dir / "run_metadata.json", dict(run_metadata))
        atomic_json(self.run_dir / "checkpoint_manifest.json", {"schema_version": 1, "checkpoints": {}})
        _atomic_text(self.run_dir / "environment.txt", environment.rstrip() + "\n")
        _atomic_text(self.run_dir / "git_state.txt", git_state.rstrip() + "\n")
        for name, headers in LOCAL_FILE_HEADERS.items():
            self._rewrite_csv(name, headers, [])

    def _rewrite_csv(
        self,
        name: str,
        headers: list[str],
        rows: Iterable[Mapping[str, Any]],
    ) -> None:
        path = self.run_dir / name
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=self.run_dir)
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def append_metrics(
        self,
        name: str,
        epoch: int,
        global_step: int,
        metrics: Mapping[str, float],
    ) -> None:
        if name not in LOCAL_FILE_HEADERS or name == "clinical_metrics.csv":
            raise ValueError(f"unsupported scalar metric file: {name}")
        path = self.run_dir / name
        headers = LOCAL_FILE_HEADERS[name]
        existing: list[dict[str, str]] = []
        with path.open(newline="", encoding="utf-8") as handle:
            existing.extend(csv.DictReader(handle))
        for metric, value in sorted(metrics.items()):
            existing.append(
                {
                    "epoch": str(epoch),
                    "global_step": str(global_step),
                    "metric": metric,
                    "value": repr(float(value)),
                }
            )
        self._rewrite_csv(name, headers, existing)

    def update_checkpoint_manifest(self, label: str, metadata: Mapping[str, Any]) -> None:
        path = self.run_dir / "checkpoint_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["checkpoints"][label] = dict(metadata)
        atomic_json(path, manifest)

    def update_run_metadata(self, updates: Mapping[str, Any]) -> None:
        path = self.run_dir / "run_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update(updates)
        atomic_json(path, metadata)


class WandbLogger:
    """Lazy W&B adapter supporting network-free disabled and offline runs."""

    def __init__(
        self,
        mode: str,
        run_dir: Path,
        config: Mapping[str, Any],
        project: str,
        group: str | None,
        job_type: str,
        run_name: str,
    ) -> None:
        if mode not in {"disabled", "offline", "online"}:
            raise ValueError("wandb mode must be disabled, offline, or online")
        self.mode = mode
        self.run = None
        if mode == "disabled":
            return
        if mode == "offline":
            wandb_local = Path(run_dir) / ".wandb-local"
            wandb_local.mkdir(parents=True, exist_ok=True)
            os.environ["WANDB_DIR"] = str(Path(run_dir))
            os.environ["WANDB_CACHE_DIR"] = str(wandb_local / "cache")
            os.environ["WANDB_CONFIG_DIR"] = str(wandb_local / "config")
            os.environ["WANDB_DATA_DIR"] = str(wandb_local / "data")
            os.environ.setdefault("WANDB_ERROR_REPORTING", "false")
        import wandb

        public_config = {
            key: value
            for key, value in config.items()
            if key not in {"config", "data_root", "output_dir", "resume_checkpoint"}
        }
        self.run = wandb.init(
            project=project,
            group=group,
            job_type=job_type,
            name=run_name,
            config=public_config,
            mode=mode,
            dir=str(run_dir),
            reinit="finish_previous",
        )

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        if self.run is not None:
            self.run.log(dict(metrics), step=step)

    def histogram(self, values: Any) -> Any:
        if self.run is None:
            return None
        import wandb

        return wandb.Histogram(values)

    def finish(self, summary: Mapping[str, Any] | None = None, exit_code: int = 0) -> None:
        if self.run is not None:
            if summary:
                self.run.summary.update(dict(summary))
            self.run.finish(exit_code=exit_code)
