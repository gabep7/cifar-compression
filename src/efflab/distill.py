"""Response-based knowledge distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from efflab.evaluate import evaluate
from efflab.io import save_checkpoint


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    T: float,
    alpha: float,
) -> torch.Tensor:
    """alpha * CE(student, y) + (1-alpha) * T^2 * KL(log_softmax(s/T), softmax(t/T))."""
    ce = F.cross_entropy(student_logits, targets)
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(log_p, q, reduction="batchmean")
    return alpha * ce + (1.0 - alpha) * (T * T) * kl


def train_student_distill(
    student: nn.Module,
    teacher: nn.Module | None,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    train_cfg: dict[str, Any],
    distill_cfg: dict[str, Any] | None = None,
    checkpoint_path: Path | str | None = None,
    *,
    use_distill: bool = True,
    model_name: str = "mobilenet_v3_small",
    full_config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train student with optional KD. Teacher is frozen when used."""
    student = student.to(device)
    distill_cfg = distill_cfg or {}
    T = float(distill_cfg.get("T", 4.0))
    alpha = float(distill_cfg.get("alpha", 0.3))

    if use_distill:
        if teacher is None:
            raise ValueError("teacher required when use_distill=True")
        teacher = teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        method = "distill"
    else:
        method = "scratch"

    epochs = int(train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", 0.05))
    momentum = float(train_cfg.get("momentum", 0.9))
    weight_decay = float(train_cfg.get("weight_decay", 5e-4))
    milestones = list(train_cfg.get("lr_milestones", [30, 40]))
    gamma = float(train_cfg.get("lr_gamma", 0.1))

    optimizer = SGD(
        student.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    ce_eval = nn.CrossEntropyLoss()

    history: list[dict[str, Any]] = []
    best_acc = -1.0
    final_acc = 0.0
    extra = dict(extra or {})
    extra.update({"T": T, "alpha": alpha, "use_distill": use_distill})

    for epoch in range(1, epochs + 1):
        student.train()
        running_loss = 0.0
        correct = 0
        n = 0
        pbar = tqdm(
            train_loader,
            desc=f"{method} epoch {epoch}/{epochs}",
            leave=False,
        )
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            student_logits = student(inputs)
            if use_distill:
                assert teacher is not None
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
                loss = distillation_loss(
                    student_logits, teacher_logits, targets, T=T, alpha=alpha
                )
            else:
                loss = F.cross_entropy(student_logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * targets.size(0)
            pred = student_logits.argmax(dim=1)
            correct += int((pred == targets).sum().item())
            n += int(targets.size(0))
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = running_loss / max(n, 1)
        train_acc = correct / max(n, 1)
        eval_metrics = evaluate(student, test_loader, device, ce_eval)
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
            f"{method} epoch {epoch}/{epochs} "
            f"train_acc={train_acc:.4f} test_acc={final_acc:.4f}"
        )

        if checkpoint_path is not None and final_acc > best_acc:
            best_acc = final_acc
            payload = {
                "model_state": student.state_dict(),
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
                "extra": extra,
            }
            save_checkpoint(Path(checkpoint_path), payload)

    if best_acc < 0:
        best_acc = final_acc

    return {
        "history": history,
        "best_acc": best_acc,
        "final_acc": final_acc,
        "method": method,
    }
