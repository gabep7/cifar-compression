"""Checkpoint and results I/O."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path | str, payload: dict) -> None:
    """Save a checkpoint dict to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path | str, map_location=None) -> dict:
    """Load a checkpoint dict from disk."""
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_json(path: Path | str, obj: Any) -> None:
    """Write JSON with parent dirs created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")


def append_result_row(csv_path: Path | str, row: dict[str, Any]) -> None:
    """Append a result row; create CSV with headers on first write."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not write_header:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and list(reader.fieldnames) != fieldnames:
                raise ValueError(
                    f"CSV columns mismatch for {csv_path}: "
                    f"existing={list(reader.fieldnames)} new={fieldnames}"
                )
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
