# CIFAR-10 model efficiency experiments

Research-first portfolio project comparing **pruning**, **dynamic quantization**, and **knowledge distillation** on CIFAR-10. Measures accuracy, sparsity, disk size, and latency so tradeoffs are explicit.

Primary writeup: [`report.md`](report.md).

## Setup

```bash
cd ~/Projects/Personal/cvproj
uv sync
```

Requires Python 3.11+. Uses `torch` / `torchvision`. CUDA or MPS is used automatically when available; smoke runs work on CPU.

## Reproduce full experiments

```bash
# 1) Baseline teacher (ResNet-18, ~50 epochs)
uv run python scripts/train_baseline.py --config configs/baseline.yaml

# 2) Global magnitude prune + fine-tune
uv run python scripts/run_prune.py --config configs/prune_50.yaml
uv run python scripts/run_prune.py --config configs/prune_80.yaml
uv run python scripts/run_prune.py --config configs/prune_90.yaml

# 3) Student: KD vs from-scratch control
uv run python scripts/run_distill.py --config configs/distill_mobilenet.yaml --mode distill
uv run python scripts/run_distill.py --config configs/distill_mobilenet.yaml --mode scratch

# 4) Dynamic quant (CPU)
uv run python scripts/run_quantize.py --config configs/quant_dynamic.yaml

# 5) Unified table + plots
uv run python scripts/run_benchmark.py --checkpoints \
  baseline=checkpoints/baseline_resnet18_best.pt \
  prune50=checkpoints/prune_50_best.pt \
  prune80=checkpoints/prune_80_best.pt \
  prune90=checkpoints/prune_90_best.pt \
  student_distill=checkpoints/student_distill_best.pt \
  student_scratch=checkpoints/student_scratch_best.pt \
  quant_dynamic=checkpoints/baseline_resnet18_best.pt
uv run python scripts/plot_results.py
```

Outputs:

- `results/summary.csv`
- `results/figures/acc_vs_size.png`
- `results/figures/acc_vs_latency.png`
- `results/figures/sparsity_vs_acc.png`
- raw JSON under `results/raw/`

## Smoke path (wiring only, 1 epoch)

```bash
uv sync
uv run python scripts/train_baseline.py --config configs/smoke.yaml
uv run python scripts/run_prune.py --config configs/smoke_prune.yaml
uv run python scripts/run_distill.py --config configs/smoke_distill.yaml --mode distill
uv run python scripts/run_distill.py --config configs/smoke_distill.yaml --mode scratch
uv run python scripts/run_quantize.py --config configs/quant_dynamic.yaml
uv run python scripts/run_benchmark.py --config configs/smoke.yaml --checkpoints \
  baseline=checkpoints/baseline_resnet18_best.pt \
  prune50=checkpoints/prune_50_best.pt \
  student_distill=checkpoints/student_distill_best.pt \
  student_scratch=checkpoints/student_scratch_best.pt \
  quant_dynamic=checkpoints/baseline_resnet18_best.pt
uv run python scripts/plot_results.py
```

## Layout

```
configs/          experiment YAML
src/efflab/       shared library
scripts/          train / prune / distill / quant / benchmark / plot
results/          summary.csv, figures/, raw/
checkpoints/      gitignored weights
report.md         research writeup
```

## Hardware note

Fill after your runs: OS, CPU/GPU, torch version, device used for training vs quant (quant is CPU-only).

Example template:

```
Hardware: <machine>
Train device: <cuda|mps|cpu>
Quant device: cpu
torch: <version>
```

## License

Personal portfolio project. Use and adapt as you like for applications and learning.
