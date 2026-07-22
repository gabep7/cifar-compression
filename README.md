# CIFAR-10 compression experiments

I trained a ResNet-18 on CIFAR-10 and compared three ways to make it smaller or faster: magnitude pruning, knowledge distillation into MobileNetV3-Small, and dynamic quantization. The goal was to see what actually moves the needle on accuracy, parameter count, model size, and inference latency.

The full writeup with results and discussion is in [`report.md`](report.md).

## Setup

```bash
cd ~/Projects/Personal/cvproj
uv sync
```

Requires Python 3.11+. Uses `torch` and `torchvision`. CUDA or MPS is used automatically when available. I ran the full experiments on a Kaggle Tesla T4.

## Reproduce

```bash
# 1) Baseline (ResNet-18, 50 epochs)
uv run python scripts/train_baseline.py --config configs/baseline.yaml

# 2) Prune + fine-tune at 50%, 80%, 90%
uv run python scripts/run_prune.py --config configs/prune_50.yaml
uv run python scripts/run_prune.py --config configs/prune_80.yaml
uv run python scripts/run_prune.py --config configs/prune_90.yaml

# 3) Distillation + scratch control
uv run python scripts/run_distill.py --config configs/distill_mobilenet.yaml --mode distill
uv run python scripts/run_distill.py --config configs/distill_mobilenet.yaml --mode scratch

# 4) Dynamic quantization (CPU only)
uv run python scripts/run_quantize.py --config configs/quant_dynamic.yaml

# 5) Benchmark everything into one table + plots
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

Or just run `make all`.

## Results

| name | method | accuracy | sparsity | size_mb | latency_ms_p50 |
| --- | --- | --- | --- | --- | --- |
| baseline | ResNet-18 | 93.11% | 0% | 42.70 | 2.49 |
| prune50 | prune | 93.08% | 50% | 42.70 | 2.51 |
| prune80 | prune | 92.91% | 80% | 42.70 | 2.51 |
| prune90 | prune | 92.59% | 90% | 42.70 | 2.53 |
| student_distill | KD | 76.52% | 0% | 5.96 | 4.87 |
| student_scratch | scratch | 76.00% | 0% | 5.96 | 4.88 |
| quant_dynamic | qint8 | 93.09% | 0% | 42.69 | 17.88 (CPU) |

## Layout

```
configs/          experiment YAML files
src/efflab/       shared library
scripts/          train, prune, distill, quant, benchmark, plot
results/          summary.csv, figures/, raw/
report.md         full writeup with discussion
```

## Hardware

- Training: Kaggle Tesla T4 (16GB)
- Quantization eval: CPU (forced, dynamic quant requires it)
- torch 2.10.0+cu128, Python 3.12

## License

MIT. Use and adapt freely.