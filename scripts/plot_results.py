#!/usr/bin/env python3
"""Plot accuracy vs size / latency / sparsity from summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot efficiency lab results")
    parser.add_argument("--summary", type=str, default="results/summary.csv")
    parser.add_argument("--out-dir", type=str, default="results/figures")
    args = parser.parse_args()

    summary = Path(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary.exists():
        raise SystemExit(f"Summary not found: {summary}. Run run_benchmark.py first.")

    df = pd.read_csv(summary)
    if df.empty:
        raise SystemExit(f"Summary is empty: {summary}")

    # acc vs size
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["size_mb"], df["accuracy"], s=60)
    for _, row in df.iterrows():
        ax.annotate(str(row["name"]), (row["size_mb"], row["accuracy"]), fontsize=8)
    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "acc_vs_size.png", dpi=150)
    plt.close(fig)

    # acc vs latency
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["latency_ms_p50"], df["accuracy"], s=60)
    for _, row in df.iterrows():
        ax.annotate(
            str(row["name"]), (row["latency_ms_p50"], row["accuracy"]), fontsize=8
        )
    ax.set_xlabel("Latency p50 (ms)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs latency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "acc_vs_latency.png", dpi=150)
    plt.close(fig)

    # sparsity vs acc (pruning family only)
    prune_df = df[df["method"].astype(str).str.contains("prune", case=False, na=False)]
    if prune_df.empty:
        prune_df = df[df["sparsity"] > 0.05]
    fig, ax = plt.subplots(figsize=(7, 5))
    if not prune_df.empty:
        ax.scatter(prune_df["sparsity"], prune_df["accuracy"], s=60)
        for _, row in prune_df.iterrows():
            ax.annotate(
                str(row["name"]), (row["sparsity"], row["accuracy"]), fontsize=8
            )
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Sparsity vs accuracy (pruning family)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "sparsity_vs_acc.png", dpi=150)
    plt.close(fig)

    print(f"wrote figures under {out_dir}")


if __name__ == "__main__":
    main()
