#!/usr/bin/env python3
"""Train the CIFAR ResNet-18 baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install when needed
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efflab.config import ensure_dirs, load_config
from efflab.data import build_dataloaders
from efflab.device import resolve_device
from efflab.io import save_json
from efflab.models import build_model
from efflab.seed import set_seed
from efflab.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline ResNet-18 on CIFAR-10")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg.get("device", "auto")))
    print(f"device={device}")

    paths = cfg.get("paths", {})
    ckpt_dir = Path(paths.get("checkpoint_dir", "checkpoints"))
    results_dir = Path(paths.get("results_dir", "results/raw"))
    ensure_dirs(ckpt_dir, results_dir)

    train_loader, test_loader = build_dataloaders(cfg)
    model_cfg = cfg.get("model", {})
    model = build_model(
        name=str(model_cfg.get("name", "resnet18")),
        num_classes=int(model_cfg.get("num_classes", 10)),
        pretrained=bool(model_cfg.get("pretrained", False)),
    )

    experiment = str(cfg.get("experiment", "baseline_resnet18"))
    ckpt_path = ckpt_dir / "baseline_resnet18_best.pt"
    # Smoke runs still write the standard baseline path so prune/distill can chain.
    if experiment.startswith("smoke"):
        ckpt_path = ckpt_dir / "baseline_resnet18_best.pt"

    result = train_model(
        model,
        train_loader,
        test_loader,
        device,
        cfg.get("train", {}),
        checkpoint_path=ckpt_path,
        model_name=str(model_cfg.get("name", "resnet18")),
        method="baseline",
        full_config=cfg,
        extra={},
    )

    out = {
        "experiment": experiment,
        "device": str(device),
        "checkpoint": str(ckpt_path),
        "best_acc": result["best_acc"],
        "final_acc": result["final_acc"],
        "history": result["history"],
    }
    json_path = results_dir / f"{experiment}_train.json"
    save_json(json_path, out)
    print(
        f"summary experiment={experiment} best_acc={result['best_acc']:.4f} "
        f"final_acc={result['final_acc']:.4f} ckpt={ckpt_path}"
    )


if __name__ == "__main__":
    main()
