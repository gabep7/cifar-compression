"""Baseline training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from efflab.evaluate import evaluate
from efflab.io import save_checkpoint


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    train_cfg: dict[str, Any],
    checkpoint_path: Path | str | None = None,
    *,
    model_name: str = "resnet18",
    method: str = "baseline",
    full_config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train with SGD + MultiStepLR. Optionally save best checkpoint."""
    model = model.to(device)
    epochs = int(train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", 0.1))
    momentum = float(train_cfg.get("momentum", 0.9))
    weight_decay = float(train_cfg.get("weight_decay", 5e-4))
    milestones = list(train_cfg.get("lr_milestones", [30, 40]))
    gamma = float(train_cfg.get("lr_gamma", 0.1))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    history: list[dict[str, Any]] = []
    best_acc = -1.0
    final_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * targets.size(0)
            pred = outputs.argmax(dim=1)
            correct += int((pred == targets).sum().item())
            n += int(targets.size(0))
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = running_loss / max(n, 1)
        train_acc = correct / max(n, 1)
        eval_metrics = evaluate(model, test_loader, device, criterion)
        final_acc = float(eval_metrics["accuracy"])
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": float(eval_metrics["loss"]),
            "test_acc": final_acc,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        print(
            f"epoch {epoch}/{epochs} "
            f"train_acc={train_acc:.4f} test_acc={final_acc:.4f} "
            f"lr={row['lr']:.4g}"
        )

        if checkpoint_path is not None and final_acc > best_acc:
            best_acc = final_acc
            payload = {
                "model_state": model.state_dict(),
                "model_name": model_name,
                "num_classes": int(
                    (full_config or {}).get("model", {}).get("num_classes", 10)
                ),
                "method": method,
                "config": full_config or {},
                "metrics": {
                    "test_acc": best_acc,
                    "test_loss": float(eval_metrics["loss"]),
                    "epoch": epoch,
                },
                "extra": extra or {},
            }
            save_checkpoint(Path(checkpoint_path), payload)

    if best_acc < 0:
        best_acc = final_acc

    return {
        "history": history,
        "best_acc": best_acc,
        "final_acc": final_acc,
    }
