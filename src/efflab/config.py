"""YAML config loading and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping config. Reject non-mapping roots."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories if missing."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
