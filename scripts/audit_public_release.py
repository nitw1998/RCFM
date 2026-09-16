#!/usr/bin/env python3
"""Reject files that fall outside the intentionally narrow public release."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PREFIXES = (
    "_private/",
    "private/",
    "research_private/",
    "configs/private/",
    "manuscript_private/",
    "review_private/",
)
PRIVATE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".pt", ".pth", ".ckpt")
ALLOWED_HASH_KEYS = {"split_hash"}


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def audit_configs(errors: list[str]) -> None:
    configs = sorted((ROOT / "configs").rglob("*.yaml"))
    if not configs:
        errors.append("no public RCFM-OT configs found")
        return
    for path in configs:
        relative = path.relative_to(ROOT)
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{relative}: invalid JSON-formatted YAML: {exc}")
            continue
        if config.get("use_minibatch_ot") is not True:
            errors.append(f"{relative}: use_minibatch_ot must be true")
        try:
            region_weight = float(config.get("region_weight", 0.0))
        except (TypeError, ValueError):
            region_weight = 0.0
        if region_weight <= 0.0:
            errors.append(f"{relative}: region_weight must be greater than zero")
        forbidden = sorted(
            key for key in config if "sha256" in key.lower() and key not in ALLOWED_HASH_KEYS
        )
        if forbidden:
            errors.append(f"{relative}: private artifact hashes are forbidden: {forbidden}")


def main() -> int:
    errors: list[str] = []
    for path in tracked_files():
        if path.startswith(PRIVATE_PREFIXES) or path.lower().endswith(PRIVATE_SUFFIXES):
            errors.append(f"tracked private/artifact path: {path}")
    audit_configs(errors)
    if errors:
        print("Public-release audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    config_count = len(list((ROOT / "configs").rglob("*.yaml")))
    print(f"Public-release audit passed: {config_count} RCFM-OT configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
