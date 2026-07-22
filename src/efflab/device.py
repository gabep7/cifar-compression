"""Device resolution."""

from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a torch device from preference: auto|cpu|cuda|mps."""
    pref = (preference or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if pref == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    raise ValueError(f"Unknown device preference: {preference!r}")
