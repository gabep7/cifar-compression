# CIFAR-10 compression experiments

Three ways to make a ResNet-18 smaller or faster, measured end to end: global unstructured magnitude pruning, knowledge distillation into MobileNetV3-Small, and dynamic `qint8` quantization. Every run reports accuracy, nonzero parameters, serialized size, and inference latency, so the tradeoffs are visible instead of asserted.

Two of the three techniques did not do what the marketing says. That is the point of the repo.

## What the numbers say

**Pruning cuts parameters by 10x and buys nothing at runtime.** At 90% sparsity the model keeps 92.59% accuracy, down 0.52 points from baseline, with 1.13M nonzero weights left. Serialized size does not move (42.70 MB) and neither does latency (2.53 vs 2.49 ms), because zeros are still stored as float32 and dense kernels still multiply them. Unstructured sparsity needs sparse kernels, structured channel pruning, or a sparsity-aware runtime before it turns into wall-clock anything.

**Dynamic quantization is the wrong tool for a conv net.** Accuracy holds at 93.08%, but size drops 0.04% because ResNet-18 is convolution-heavy and dynamic quant only touches `nn.Linear`. Here that is a single fc layer. Static PTQ or QAT is what actually reaches convolutions.

**Distillation is the only thing that shrinks the model, and the scratch control eats most of the credit.** The MobileNetV3-Small student is 7.2x smaller at 5.96 MB but gives up 16.6 accuracy points. Trained with the ResNet-18 teacher it hits 76.52%; trained from scratch on an identical schedule it hits 76.00%. The teacher signal is worth 0.52 points, not the transformation it is usually sold as. The student is also *slower* than the teacher on a T4 (4.87 vs 2.49 ms) because depthwise convolutions map poorly to the hardware, so "smaller" and "faster" came apart completely.

At matched parameter count the ranking is unambiguous: pruned ResNet-18 at 1.13M nonzero weights holds 92.59%, while the 1.53M-parameter student manages 76.52%. Architecture family mattered more than either compression method.

## Results

From [`results/summary.csv`](results/summary.csv):

| name | method | accuracy | sparsity | params_nonzero | size_mb | p50 latency | device |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | ResNet-18 | 93.11% | 0% | 11,173,962 | 42.70 | 2.49 ms | cuda |
| prune50 | prune + finetune | 93.08% | 49.96% | 5,591,786 | 42.70 | 2.51 ms | cuda |
| prune80 | prune + finetune | 92.91% | 79.93% | 2,242,480 | 42.70 | 2.51 ms | cuda |
| prune90 | prune + finetune | 92.59% | 89.92% | 1,126,045 | 42.70 | 2.53 ms | cuda |
| student_distill | KD, T=4, α=0.3 | 76.52% | 0% | 1,528,105 | 5.96 | 4.87 ms | cuda |
| student_scratch | control, no teacher | 76.00% | 0% | 1,528,106 | 5.96 | 4.88 ms | cuda |
| quant_dynamic | Linear qint8 | 93.08% | 0% | 11,168,832 | 42.69 | 17.88 ms | cpu |

Latency protocol: 50 warmup passes, then 200 timed runs with device synchronize around the timed region. The quantized row is CPU-only because dynamic quant requires it, so its 17.88 ms is not comparable to the CUDA rows and is not evidence of a quantization slowdown.

Seed 42, single run per configuration, no error bars. Differences under roughly 0.3 points should be treated as noise.

Figures: [`acc_vs_size.png`](results/figures/acc_vs_size.png), [`acc_vs_latency.png`](results/figures/acc_vs_latency.png), [`sparsity_vs_acc.png`](results/figures/sparsity_vs_acc.png).

Full method details, per-technique discussion, and limitations are in [`report.md`](report.md).

## Setup

```bash
git clone https://github.com/gabep7/cifar-compression
cd cifar-compression
uv sync
```

Python 3.11+, `torch` 2.10.0+cu128, `torchvision`. CUDA or MPS is picked up automatically. Full experiments ran on a Kaggle Tesla T4 (16GB).

## Reproduce

```bash
# baseline ResNet-18, 50 epochs
uv run python scripts/train_baseline.py --config configs/baseline.yaml

# prune + fine-tune at 50/80/90% sparsity
uv run python scripts/run_prune.py --config configs/prune_50.yaml
uv run python scripts/run_prune.py --config configs/prune_80.yaml
uv run python scripts/run_prune.py --config configs/prune_90.yaml

# distillation and the scratch control
uv run python scripts/run_distill.py --config configs/distill_mobilenet.yaml --mode distill
uv run python scripts/run_distill.py --config configs/distill_mobilenet.yaml --mode scratch

# dynamic quantization, CPU only
uv run python scripts/run_quantize.py --config configs/quant_dynamic.yaml

# collapse everything into summary.csv + figures
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

`make all` runs the whole chain. `configs/smoke*.yaml` are short versions for checking the pipeline works before committing to 50 epochs.

## Layout

```
configs/          experiment YAML, one per run
src/efflab/       library: data, models, prune, quantize, distill, benchmark
scripts/          entry points for each stage
results/          summary.csv, figures/
report.md         full writeup
```

## Next

Structured channel pruning and QAT are the two changes that would turn the parameter reductions above into real latency wins. Beyond that: ONNX Runtime and TensorRT export numbers, multi-seed error bars, and a stronger student than MobileNetV3-Small.

## License

MIT.
