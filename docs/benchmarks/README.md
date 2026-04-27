# Classical Model Benchmark Artifacts

This directory includes benchmark artifacts for PRISM.

## Classical-model workflow

1. Generate classical sklearn models and export ONNX:

```bash
poetry run python scripts/create_models.py
```

2. Benchmark PRISM API pipeline over generated ONNX models:

```bash
poetry run python scripts/benchmark_classical_models.py \
  --iterations 120 \
  --output-json docs/benchmarks/classical-models-latest.json
```

## Artifacts

- `classical-model-generation.json`: generated model list and sample sklearn outputs
- `classical-models-latest.json`: per-model latency/throughput plus exact sample API inputs/outputs and deviation metrics
- `latest.json`, `latest-high-latency.json`: existing module/route benchmark profiles
- `figures/`: generated benchmark plots
