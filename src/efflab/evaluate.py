"""Evaluation, parameter counts, and latency measurement."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """Return accuracy, mean loss, and sample count."""
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    n = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += float(loss.item()) * targets.size(0)
        pred = outputs.argmax(dim=1)
        correct += int((pred == targets).sum().item())
        n += int(targets.size(0))

    if n == 0:
        return {"accuracy": 0.0, "loss": 0.0, "n": 0}
    return {
        "accuracy": correct / n,
        "loss": total_loss / n,
        "n": n,
    }


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, trainable, and nonzero parameters.

    When torch.prune reparameterization is active, use the effective
    ``module.weight`` (mask applied) instead of raw parameter tensors.
    """
    params_total = 0
    params_trainable = 0
    params_nonzero = 0
    seen: set[int] = set()

    for module in model.modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.requires_grad is not None:
            # Prefer effective weight (handles prune masks).
            if id(weight) not in seen:
                seen.add(id(weight))
                n = int(weight.numel())
                params_total += n
                if weight.requires_grad:
                    params_trainable += n
                params_nonzero += int(torch.count_nonzero(weight.detach()).item())

        for name, p in module.named_parameters(recurse=False):
            if name == "weight" or name.endswith("_orig") or name.endswith("_mask"):
                # weight counted above; skip prune internals
                if name == "weight" or name == "weight_orig" or name == "weight_mask":
                    continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            n = int(p.numel())
            params_total += n
            if p.requires_grad:
                params_trainable += n
            params_nonzero += int(torch.count_nonzero(p.detach()).item())

    if params_total == 0:
        # Fallback for exotic modules
        for p in model.parameters():
            n = int(p.numel())
            params_total += n
            if p.requires_grad:
                params_trainable += n
            params_nonzero += int(torch.count_nonzero(p.detach()).item())

    return {
        "params_total": params_total,
        "params_trainable": params_trainable,
        "params_nonzero": params_nonzero,
    }


def sparsity_ratio(model: nn.Module) -> float:
    """Fraction of zero weights over all parameters."""
    counts = count_parameters(model)
    total = counts["params_total"]
    if total == 0:
        return 0.0
    return 1.0 - (counts["params_nonzero"] / total)


def model_disk_mb(path: Path | str) -> float:
    """File size of a checkpoint or serialized model in megabytes."""
    path = Path(path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    device: torch.device,
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    warmup: int = 50,
    runs: int = 200,
) -> dict[str, float]:
    """Measure forward latency (ms) and throughput (img/s)."""
    model.eval()
    model = model.to(device)
    dummy = torch.randn(*input_shape, device=device)

    for _ in range(warmup):
        _ = model(dummy)
    _synchronize(device)

    times: list[float] = []
    for _ in range(runs):
        _synchronize(device)
        t0 = time.perf_counter()
        _ = model(dummy)
        _synchronize(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times_sorted = sorted(times)
    mean_ms = sum(times) / len(times)
    p50 = times_sorted[len(times_sorted) // 2]
    p95 = times_sorted[int(len(times_sorted) * 0.95)]
    batch = input_shape[0]
    throughput = (batch * 1000.0) / mean_ms if mean_ms > 0 else 0.0
    return {
        "latency_ms_mean": mean_ms,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "throughput_img_s": throughput,
    }
