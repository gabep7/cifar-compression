.PHONY: setup smoke baseline experiments benchmark all clean clean-data

setup:
	uv sync

smoke:
	./run_all.sh --smoke

baseline:
	uv run python scripts/train_baseline.py --config configs/baseline.yaml

experiments:
	uv run python scripts/run_prune.py --config configs/prune_50.yaml
	uv run python scripts/run_prune.py --config configs/prune_80.yaml
	uv run python scripts/run_prune.py --config configs/prune_90.yaml
	uv run python scripts/run_distill.py --config configs/distill_mobilenet.yaml --mode distill
	uv run python scripts/run_distill.py --config configs/distill_mobilenet.yaml --mode scratch
	uv run python scripts/run_quantize.py --config configs/quant_dynamic.yaml

benchmark:
	uv run python scripts/run_benchmark.py --checkpoints \
		baseline=checkpoints/baseline_resnet18_best.pt \
		prune50=checkpoints/prune_50_best.pt \
		prune80=checkpoints/prune_80_best.pt \
		prune90=checkpoints/prune_90_best.pt \
		student_distill=checkpoints/student_distill_best.pt \
		student_scratch=checkpoints/student_scratch_best.pt \
		quant_dynamic=checkpoints/quant_dynamic_meta.pt
	uv run python scripts/plot_results.py

all: baseline experiments benchmark

clean:
	rm -rf checkpoints
	rm -rf results/raw
	rm -f results/summary.csv
	rm -f results/figures/*.png

clean-data:
	rm -rf data