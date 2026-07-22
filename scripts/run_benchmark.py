#!/usr/bin/env python3
"""Benchmark named checkpoints into results/summary.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efflab.benchmark import benchmark_checkpoint
from efflab.config import ensure_dirs, load_config
from efflab.data import build_dataloaders
from efflab.device import resolve_device
from efflab.io import append_result_row, save_json
from efflab.seed import set_seed


def parse_checkpoints(pairs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"Expected name=path, got {item!r}")
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark checkpoints to summary.csv")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="name=path pairs, e.g. baseline=checkpoints/baseline_resnet18_best.pt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Config used for data loading defaults",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="results/summary.csv",
        help="Output CSV path (rewritten each run)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(args.seed if args.seed is not None else cfg.get("seed", 42)))
    device = resolve_device(str(args.device or cfg.get("device", "auto")))
    print(f"device={device}")

    summary_path = Path(args.summary)
    ensure_dirs(summary_path.parent, Path("results/raw"), Path("results/figures"))
    if summary_path.exists():
        summary_path.unlink()

    _, test_loader = build_dataloaders(cfg)
    named = parse_checkpoints(args.checkpoints)
    rows = []

    for name, path in named.items():
        if not path.exists():
            print(f"skip {name}: missing {path}")
            continue
        force_quant = name.startswith("quant") or "quant" in name
        print(f"benchmarking {name} <- {path} quant={force_quant}")
        row = benchmark_checkpoint(
            name,
            path,
            test_loader,
            device,
            force_quant=force_quant,
        )
        append_result_row(summary_path, row)
        rows.append(row)
        print(
            f"  acc={row['accuracy']:.4f} sparsity={row['sparsity']:.4f} "
            f"size_mb={row['size_mb']:.3f} lat_p50={row['latency_ms_p50']:.3f}ms "
            f"device={row['device']}"
        )

    save_json(Path("results/raw/benchmark_summary.json"), {"rows": rows})
    print(f"wrote {summary_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
