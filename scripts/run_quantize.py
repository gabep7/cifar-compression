#!/usr/bin/env python3
"""Dynamic quantize baseline Linear layers and evaluate on CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efflab.config import ensure_dirs, load_config
from efflab.data import build_dataloaders
from efflab.device import resolve_device
from efflab.evaluate import count_parameters, evaluate, measure_latency
from efflab.io import load_checkpoint, save_checkpoint, save_json
from efflab.models import build_model
from efflab.quantize import dynamic_quantize_linear, quantize_notes, quantized_size_mb
from efflab.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic quantization (CPU)")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    # Quant eval is CPU-only by design.
    device = resolve_device("cpu")

    baseline_path = Path(
        cfg.get("baseline_checkpoint", "checkpoints/baseline_resnet18_best.pt")
    )
    if not baseline_path.exists():
        raise SystemExit(
            f"Baseline checkpoint not found: {baseline_path}. Run train_baseline.py first."
        )

    paths = cfg.get("paths", {})
    ckpt_dir = Path(paths.get("checkpoint_dir", "checkpoints"))
    results_dir = Path(paths.get("results_dir", "results/raw"))
    ensure_dirs(ckpt_dir, results_dir)

    _, test_loader = build_dataloaders(cfg)
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "resnet18"))
    num_classes = int(model_cfg.get("num_classes", 10))

    ckpt = load_checkpoint(baseline_path, map_location="cpu")
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    fp32_metrics = evaluate(model, test_loader, device)
    qmodel = dynamic_quantize_linear(model)
    q_metrics = evaluate(qmodel, test_loader, device)
    counts = count_parameters(qmodel)
    size_path = ckpt_dir / "quant_dynamic_serialized.pt"
    size_mb = quantized_size_mb(qmodel, path=size_path)
    latency = measure_latency(qmodel, device)

    # Store pointer checkpoint metadata for benchmark (weights still from baseline + quant at load).
    meta_path = ckpt_dir / "quant_dynamic_meta.pt"
    save_checkpoint(
        meta_path,
        {
            "model_state": ckpt["model_state"],
            "model_name": model_name,
            "num_classes": num_classes,
            "method": "quant_dynamic",
            "config": cfg,
            "metrics": {
                "test_acc": float(q_metrics["accuracy"]),
                "fp32_test_acc": float(fp32_metrics["accuracy"]),
            },
            "extra": {
                **quantize_notes(),
                "size_mb": size_mb,
                "baseline_checkpoint": str(baseline_path),
            },
        },
    )

    experiment = str(cfg.get("experiment", "quant_dynamic"))
    payload = {
        "experiment": experiment,
        "device": "cpu",
        "baseline_checkpoint": str(baseline_path),
        "fp32_acc": fp32_metrics["accuracy"],
        "quant_acc": q_metrics["accuracy"],
        "params_total": counts["params_total"],
        "params_nonzero": counts["params_nonzero"],
        "size_mb": size_mb,
        "latency_ms_p50": latency["latency_ms_p50"],
        "latency_ms_mean": latency["latency_ms_mean"],
        "notes": quantize_notes(),
        "meta_checkpoint": str(meta_path),
    }
    json_path = results_dir / f"{experiment}.json"
    save_json(json_path, payload)
    print(
        f"summary experiment={experiment} quant_acc={q_metrics['accuracy']:.4f} "
        f"fp32_acc={fp32_metrics['accuracy']:.4f} size_mb={size_mb:.3f} "
        f"latency_p50={latency['latency_ms_p50']:.3f}ms"
    )


if __name__ == "__main__":
    main()
