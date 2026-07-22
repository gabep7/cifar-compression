"""Shared benchmarking helpers for summary rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from efflab.evaluate import (
    count_parameters,
    evaluate,
    measure_latency,
    model_disk_mb,
    sparsity_ratio,
)
from efflab.io import load_checkpoint
from efflab.models import build_model
from efflab.quantize import dynamic_quantize_linear, quantized_size_mb


REQUIRED_KEYS = ("model_state", "model_name", "method")


def load_model_from_checkpoint(
    path: Path | str,
    map_location=None,
    *,
    apply_quant: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild architecture from checkpoint metadata and load weights."""
    path = Path(path)
    ckpt = load_checkpoint(path, map_location=map_location)
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise KeyError(
            f"Checkpoint {path} missing required keys: {missing}. "
            f"Present: {sorted(ckpt.keys())}"
        )
    model_name = ckpt["model_name"]
    num_classes = int(ckpt.get("num_classes", 10))
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    state = ckpt["model_state"]
    model.load_state_dict(state)
    model.eval()
    if apply_quant or ckpt.get("method") == "quant_dynamic":
        model = dynamic_quantize_linear(model)
    return model, ckpt


def benchmark_checkpoint(
    name: str,
    path: Path | str,
    test_loader,
    device: torch.device,
    *,
    force_quant: bool = False,
    quant_size_path: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint and return a summary row dict."""
    path = Path(path)
    is_quant = force_quant or name.startswith("quant")
    map_loc = "cpu" if is_quant else device
    model, ckpt = load_model_from_checkpoint(
        path, map_location=map_loc, apply_quant=is_quant
    )

    bench_device = torch.device("cpu") if is_quant else device
    model = model.to(bench_device)

    metrics = evaluate(model, test_loader, bench_device)
    counts = count_parameters(model)
    spars = sparsity_ratio(model)

    if is_quant:
        size_mb = quantized_size_mb(
            model,
            path=quant_size_path
            or Path("checkpoints") / f"{name}_serialized.pt",
        )
        method = "dynamic_quant"
        notes = "cpu; dynamic Linear qint8"
    else:
        size_mb = model_disk_mb(path)
        method = str(ckpt.get("method", "unknown"))
        notes = ""

    latency = measure_latency(model, bench_device)

    return {
        "name": name,
        "method": method,
        "accuracy": float(metrics["accuracy"]),
        "params_total": counts["params_total"],
        "params_nonzero": counts["params_nonzero"],
        "sparsity": float(spars),
        "size_mb": float(size_mb),
        "latency_ms_p50": float(latency["latency_ms_p50"]),
        "latency_ms_mean": float(latency["latency_ms_mean"]),
        "device": str(bench_device),
        "notes": notes,
    }
