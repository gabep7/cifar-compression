#!/usr/bin/env python3
"""Load baseline, prune globally, fine-tune, save checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efflab.config import ensure_dirs, load_config
from efflab.data import build_dataloaders
from efflab.device import resolve_device
from efflab.evaluate import evaluate, sparsity_ratio
from efflab.io import load_checkpoint, save_json
from efflab.models import build_model
from efflab.prune import apply_global_magnitude_pruning, fine_tune_pruned
from efflab.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Global magnitude prune + fine-tune")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg.get("device", "auto")))

    prune_cfg = cfg.get("prune", {})
    amount = float(prune_cfg.get("amount", 0.5))
    baseline_path = Path(cfg.get("baseline_checkpoint", "checkpoints/baseline_resnet18_best.pt"))
    if not baseline_path.exists():
        raise SystemExit(
            f"Baseline checkpoint not found: {baseline_path}. Run train_baseline.py first."
        )

    paths = cfg.get("paths", {})
    ckpt_dir = Path(paths.get("checkpoint_dir", "checkpoints"))
    results_dir = Path(paths.get("results_dir", "results/raw"))
    ensure_dirs(ckpt_dir, results_dir)

    train_loader, test_loader = build_dataloaders(cfg)
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "resnet18"))
    num_classes = int(model_cfg.get("num_classes", 10))

    ckpt = load_checkpoint(baseline_path, map_location=device)
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    pre = evaluate(model, test_loader, device)
    print(f"pre_prune_acc={pre['accuracy']:.4f}")

    apply_global_magnitude_pruning(model, amount=amount)
    post_prune = evaluate(model, test_loader, device)
    spars_masked = sparsity_ratio(model)
    print(
        f"post_prune_acc={post_prune['accuracy']:.4f} "
        f"sparsity(masked)={spars_masked:.4f}"
    )

    # Fine-tune schedule from prune block with fallbacks to train.
    ft_epochs = int(prune_cfg.get("finetune_epochs", cfg.get("train", {}).get("epochs", 20)))
    ft_lr = float(prune_cfg.get("finetune_lr", cfg.get("train", {}).get("lr", 0.01)))
    train_cfg = dict(cfg.get("train", {}))
    train_cfg["epochs"] = ft_epochs
    train_cfg["lr"] = ft_lr
    train_cfg.setdefault("lr_milestones", [10, 15])
    train_cfg.setdefault("lr_gamma", 0.1)

    pct = int(round(amount * 100))
    experiment = str(cfg.get("experiment", f"prune_{pct}"))
    out_ckpt = ckpt_dir / f"prune_{pct}_best.pt"

    result = fine_tune_pruned(
        model,
        train_loader,
        test_loader,
        device,
        train_cfg,
        checkpoint_path=out_ckpt,
        model_name=model_name,
        full_config=cfg,
        extra={"amount": amount},
    )

    final_spars = sparsity_ratio(model)
    payload = {
        "experiment": experiment,
        "amount": amount,
        "device": str(device),
        "baseline_checkpoint": str(baseline_path),
        "checkpoint": str(out_ckpt),
        "pre_prune_acc": pre["accuracy"],
        "post_prune_acc": post_prune["accuracy"],
        "post_finetune_acc": result["final_acc"],
        "best_acc": result["best_acc"],
        "sparsity": final_spars,
        "history": result["history"],
    }
    json_path = results_dir / f"{experiment}.json"
    save_json(json_path, payload)
    print(
        f"summary experiment={experiment} amount={amount} "
        f"post_ft_acc={result['final_acc']:.4f} sparsity={final_spars:.4f} "
        f"ckpt={out_ckpt}"
    )


if __name__ == "__main__":
    main()
