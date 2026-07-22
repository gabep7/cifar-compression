#!/usr/bin/env python3
"""Train MobileNet student with distillation or from scratch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efflab.config import ensure_dirs, load_config
from efflab.data import build_dataloaders
from efflab.device import resolve_device
from efflab.distill import train_student_distill
from efflab.io import load_checkpoint, save_json
from efflab.models import build_model
from efflab.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge distillation / student scratch")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--mode",
        choices=("distill", "scratch"),
        default="distill",
        help="distill: KD from teacher; scratch: student only",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg.get("device", "auto")))
    use_distill = args.mode == "distill"

    paths = cfg.get("paths", {})
    ckpt_dir = Path(paths.get("checkpoint_dir", "checkpoints"))
    results_dir = Path(paths.get("results_dir", "results/raw"))
    ensure_dirs(ckpt_dir, results_dir)

    train_loader, test_loader = build_dataloaders(cfg)
    model_cfg = cfg.get("model", {})
    student_name = str(model_cfg.get("name", "mobilenet_v3_small"))
    num_classes = int(model_cfg.get("num_classes", 10))
    student = build_model(
        student_name,
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", False)),
    )

    teacher = None
    teacher_ckpt_path = None
    if use_distill:
        teacher_cfg = cfg.get("teacher", {})
        teacher_ckpt_path = Path(
            teacher_cfg.get("checkpoint", "checkpoints/baseline_resnet18_best.pt")
        )
        if not teacher_ckpt_path.exists():
            raise SystemExit(
                f"Baseline checkpoint not found: {teacher_ckpt_path}. "
                "Run train_baseline.py first."
            )
        t_ckpt = load_checkpoint(teacher_ckpt_path, map_location=device)
        teacher_name = str(teacher_cfg.get("name", t_ckpt.get("model_name", "resnet18")))
        teacher = build_model(
            teacher_name,
            num_classes=int(teacher_cfg.get("num_classes", num_classes)),
            pretrained=False,
        )
        teacher.load_state_dict(t_ckpt["model_state"])

    experiment = str(cfg.get("experiment", "distill_mobilenet"))
    if args.mode == "scratch":
        experiment = experiment.replace("distill", "scratch")
        if "scratch" not in experiment:
            experiment = f"student_scratch_{experiment}"
        out_ckpt = ckpt_dir / "student_scratch_best.pt"
    else:
        out_ckpt = ckpt_dir / "student_distill_best.pt"

    result = train_student_distill(
        student,
        teacher,
        train_loader,
        test_loader,
        device,
        cfg.get("train", {}),
        distill_cfg=cfg.get("distill", {}),
        checkpoint_path=out_ckpt,
        use_distill=use_distill,
        model_name=student_name,
        full_config=cfg,
        extra={"mode": args.mode, "teacher_checkpoint": str(teacher_ckpt_path)},
    )

    payload = {
        "experiment": experiment,
        "mode": args.mode,
        "device": str(device),
        "checkpoint": str(out_ckpt),
        "best_acc": result["best_acc"],
        "final_acc": result["final_acc"],
        "history": result["history"],
    }
    json_path = results_dir / f"{experiment}.json"
    save_json(json_path, payload)
    print(
        f"summary experiment={experiment} mode={args.mode} "
        f"best_acc={result['best_acc']:.4f} ckpt={out_ckpt}"
    )


if __name__ == "__main__":
    main()
