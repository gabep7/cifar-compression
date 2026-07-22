#!/usr/bin/env bash
set -euo pipefail

# Full experiment sequence: baseline -> prune x3 -> distill x2 -> quant -> benchmark -> plot.
# Pass --smoke to use smoke configs (single prune run) instead of the full configs.

SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
fi

if [[ "$SMOKE" -eq 1 ]]; then
  BASELINE_CFG="configs/smoke.yaml"
  PRUNE_CFGS=("configs/smoke_prune.yaml")
  DISTILL_CFG="configs/smoke_distill.yaml"
  QUANT_CFG="configs/smoke_quant.yaml"
  BENCH_CKPTS=(
    "baseline=checkpoints/baseline_resnet18_best.pt"
    "prune50=checkpoints/prune_50_best.pt"
    "student_distill=checkpoints/student_distill_best.pt"
    "student_scratch=checkpoints/student_scratch_best.pt"
    "quant_dynamic=checkpoints/quant_dynamic_meta.pt"
  )
else
  BASELINE_CFG="configs/baseline.yaml"
  PRUNE_CFGS=("configs/prune_50.yaml" "configs/prune_80.yaml" "configs/prune_90.yaml")
  DISTILL_CFG="configs/distill_mobilenet.yaml"
  QUANT_CFG="configs/quant_dynamic.yaml"
  BENCH_CKPTS=(
    "baseline=checkpoints/baseline_resnet18_best.pt"
    "prune50=checkpoints/prune_50_best.pt"
    "prune80=checkpoints/prune_80_best.pt"
    "prune90=checkpoints/prune_90_best.pt"
    "student_distill=checkpoints/student_distill_best.pt"
    "student_scratch=checkpoints/student_scratch_best.pt"
    "quant_dynamic=checkpoints/quant_dynamic_meta.pt"
  )
fi

run_step() {
  echo
  echo "================================================================"
  echo ">>> $*"
  echo "================================================================"
  "$@"
}

# Baseline
run_step uv run python scripts/train_baseline.py --config "$BASELINE_CFG"

# Prune (x3 full, x1 smoke)
for cfg in "${PRUNE_CFGS[@]}"; do
  run_step uv run python scripts/run_prune.py --config "$cfg"
done

# Distill x2: KD from teacher, then scratch control
run_step uv run python scripts/run_distill.py --config "$DISTILL_CFG" --mode distill
run_step uv run python scripts/run_distill.py --config "$DISTILL_CFG" --mode scratch

# Quant
run_step uv run python scripts/run_quantize.py --config "$QUANT_CFG"

# Benchmark
run_step uv run python scripts/run_benchmark.py --checkpoints "${BENCH_CKPTS[@]}"

# Plot
run_step uv run python scripts/plot_results.py