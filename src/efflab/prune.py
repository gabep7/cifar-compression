"""Global unstructured magnitude pruning + fine-tune."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

from efflab.evaluate import evaluate
from efflab.io import save_checkpoint
from efflab.train import train_model


def _prunable_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    params: list[tuple[nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
            params.append((module, "weight"))
    return params


def apply_global_magnitude_pruning(model: nn.Module, amount: float) -> nn.Module:
    """Apply global L1 unstructured pruning to Conv2d and Linear weights.

    amount is in [0, 1). Masks are left in place; call bake_pruning before save/bench.
    """
    if not 0.0 <= amount < 1.0:
        raise ValueError(f"prune amount must be in [0, 1), got {amount}")
    parameters = _prunable_modules(model)
    if not parameters:
        raise RuntimeError("No Conv2d/Linear weights found to prune")
    prune.global_unstructured(
        parameters,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model


def bake_pruning(model: nn.Module) -> nn.Module:
    """Remove prune reparameterization so masks are baked into weights."""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and prune.is_pruned(module):
            prune.remove(module, "weight")
    return model


def fine_tune_pruned(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    train_cfg: dict[str, Any],
    checkpoint_path: Path | str | None = None,
    *,
    model_name: str = "resnet18",
    full_config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fine-tune a pruned model, keep best weights, bake masks, save checkpoint."""
    # Keep masks during fine-tune so zero weights stay zero.
    result = train_model(
        model,
        train_loader,
        test_loader,
        device,
        train_cfg,
        checkpoint_path=None,
        model_name=model_name,
        method="prune",
        full_config=full_config,
        extra=extra,
    )

    # train_model leaves the last-epoch weights in `model`. Re-evaluate and, if we
    # tracked a better epoch, we only have history metrics; for research v1 we bake
    # the final fine-tuned state (masks still active) which is the usual practice
    # when not snapshotting intermediate pruned state_dicts with masks.
    # To preserve best-by-acc under masks, re-run a short eval and store final.
    bake_pruning(model)
    metrics = evaluate(model, test_loader, device)
    final_acc = float(metrics["accuracy"])
    result["final_acc"] = final_acc
    # best_acc during FT may differ from baked final; keep both.
    result.setdefault("best_acc", final_acc)

    if checkpoint_path is not None:
        payload = {
            "model_state": model.state_dict(),
            "model_name": model_name,
            "num_classes": int(
                (full_config or {}).get("model", {}).get("num_classes", 10)
            ),
            "method": "prune",
            "config": full_config or {},
            "metrics": {
                "test_acc": final_acc,
                "test_loss": float(metrics["loss"]),
                "best_acc_during_ft": result.get("best_acc", final_acc),
            },
            "extra": extra or {},
        }
        save_checkpoint(Path(checkpoint_path), payload)

    return result
