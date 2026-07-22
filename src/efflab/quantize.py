"""Dynamic quantization helpers (CPU inference)."""

from __future__ import annotations

import io
import platform
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _ensure_qengine() -> str | None:
    """Pick a supported quantized engine. Returns engine name or None if unavailable."""
    engines = list(torch.backends.quantized.supported_engines)
    # Prefer fbgemm (x86), then qnnpack (arm/mobile), then anything else.
    preferred = ["fbgemm", "qnnpack", "onednn", "x86"]
    for name in preferred:
        if name in engines:
            torch.backends.quantized.engine = name
            return name
    if engines:
        torch.backends.quantized.engine = engines[0]
        return engines[0]
    return None


def quant_engine_available() -> bool:
    """True if this PyTorch build can run dynamic quantized Linear ops."""
    return _ensure_qengine() is not None


def dynamic_quantize_linear(model: nn.Module) -> nn.Module:
    """Dynamically quantize Linear layers to qint8.

    Raises RuntimeError with a clear message if no quantized engine is available
    (common on some macOS / arm64 PyTorch wheels).
    """
    model = model.cpu().eval()
    engine = _ensure_qengine()
    if engine is None:
        raise RuntimeError(
            "No quantized engine available on this platform "
            f"(os={platform.system()} arch={platform.machine()} "
            f"supported_engines={list(torch.backends.quantized.supported_engines)}). "
            "Dynamic quantization requires fbgemm or qnnpack. "
            "Train/eval still work; skip quant on this machine or use a Linux/x86 build."
        )
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:
        from torch.quantization import quantize_dynamic  # type: ignore

    return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def quantized_size_mb(model: nn.Module, path: Path | str | None = None) -> float:
    """Estimate serialized size of a (possibly quantized) model in MB.

    Prefer writing state_dict to path when provided; else use an in-memory buffer.
    Falls back to parameter nbytes accounting if serialization fails.
    """
    model = model.cpu()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(model.state_dict(), path)
            return path.stat().st_size / (1024 * 1024)
        except Exception:
            pass

    try:
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.tell() / (1024 * 1024)
    except Exception:
        total_bytes = 0
        for p in model.parameters():
            total_bytes += int(p.numel() * p.element_size())
        for b in model.buffers():
            total_bytes += int(b.numel() * b.element_size())
        return total_bytes / (1024 * 1024)


def quantize_notes() -> dict[str, Any]:
    """Metadata for reports: dynamic quant targets Linear only."""
    engine = None
    try:
        engine = torch.backends.quantized.engine
    except Exception:
        engine = None
    return {
        "quant_scheme": "dynamic",
        "dtype": "qint8",
        "modules": "nn.Linear",
        "device": "cpu",
        "engine": engine,
        "caveat": (
            "Dynamic quantization primarily targets Linear layers; "
            "CIFAR ResNet-18 is conv-heavy so gains may be modest. "
            "Requires a supported quantized engine (fbgemm/qnnpack)."
        ),
    }
