# PRISM Benchmarks

This document tracks reproducible performance benchmarks for PRISM modules and routes, including the PRISM-13 batching delta.

---

## 1. Validation + Methodology

- Test validation (before and after benchmark changes): `83 passed, 0 failed`
- Script: `scripts/benchmark_models.py`
- Main command used:

```bash
poetry run python scripts/benchmark_models.py \
  --iterations 1000 \
  --burst-requests 300 \
  --burst-concurrency 60 \
  --simulated-container-latency-ms 2.0 \
  --simulated-container-per-row-ms 0.05 \
  --output-json docs/benchmarks/latest.json
```

- Metrics per benchmarked module/route:
  - Average latency (ms)
  - P95 latency (ms)
  - Throughput (ops/sec)

Notes:

- The batching comparison models a single-worker container path using a serialized forwarder lock (realistic for one model container handling one request at a time).
- The burst benchmark compares `batching=off` vs `batching=on (50ms window)` under identical load.

---

## 2. Environment

- Date: 2026-04-19
- OS: Linux
- Python: 3.13.11 (Poetry virtualenv)
- Runtime libraries: `onnxruntime` (CPUExecutionProvider), `numpy`, `joblib`

---

## 3. Module Coverage

Benchmarked modules/routes in this run:

1. `runtime.adapters.sklearn.predict`
2. `runtime.adapters.onnx.predict`
3. `app.core.input_contract.validate_payload`
4. `app.core.access_control.validate_api_key`
5. `app.core.access_control.enforce_rate_limit`
6. `app.batching.request_batcher.forward`
7. `app.routing.health.read_root`
8. `app.routing.health.monitor_health`
9. `app.routing.registry.list_registry`
10. `app.routing.inference.predict_model (batching=off)`
11. `app.routing.inference.predict_model (batching=on)`

---

## 4. Benchmark Results

### Runtime + Core Modules

| Module | Avg Latency (ms) | P95 Latency (ms) | Throughput (ops/s) | Ops |
| --- | ---: | ---: | ---: | ---: |
| `runtime.adapters.sklearn.predict` | 0.1236 | 0.1413 | 8,088.93 | 1000 |
| `runtime.adapters.onnx.predict` | 0.0610 | 0.0845 | 16,390.68 | 1000 |
| `app.core.input_contract.validate_payload` | 0.0469 | 0.0639 | 21,305.09 | 1000 |
| `app.core.access_control.validate_api_key` | 0.0105 | 0.0119 | 94,984.53 | 1000 |
| `app.core.access_control.enforce_rate_limit` | 0.0088 | 0.0145 | 113,445.31 | 1000 |
| `app.batching.request_batcher.forward` | 0.0622 | 0.0736 | 16,071.85 | 1000 |

### API Routes

| Route Module | Avg Latency (ms) | P95 Latency (ms) | Throughput (ops/s) | Ops |
| --- | ---: | ---: | ---: | ---: |
| `app.routing.health.read_root` | 92.9692 | 110.3131 | 10.76 | 1000 |
| `app.routing.health.monitor_health` | 5.1836 | 6.0301 | 192.92 | 1000 |
| `app.routing.registry.list_registry` | 5.7159 | 6.6740 | 174.95 | 1000 |
| `app.routing.inference.predict_model (batching=off)` | 211.5490 | 239.4168 | 4.73 | 300 |
| `app.routing.inference.predict_model (batching=on)` | 122.0231 | 156.6823 | 8.20 | 300 |

---

## 5. PRISM-13 Batching Performance Increase

From the measured burst scenario (`300 requests`, `concurrency=60`):

- Average latency reduction: **42.32%**
- Throughput increase: **73.37%**

Interpretation:

- Batching improves end-to-end inference performance under container-bound load by collapsing many serialized forwards into fewer batched forwards.
- A 50ms collection window adds queueing delay but still produces a strong net gain in the tested high-contention path.

---

## 6. Artifacts

- Latest JSON benchmark report: `docs/benchmarks/latest.json`
- Alternate profile report: `docs/benchmarks/latest-high-latency.json`
- Generated figures:
  - `docs/benchmarks/figures/module_avg_latency.png`
  - `docs/benchmarks/figures/module_throughput.png`
  - `docs/benchmarks/figures/batching_comparison_latest.png`
  - `docs/benchmarks/figures/batching_improvement_across_reports.png`

### Regenerate figures

```bash
poetry run python scripts/generate_benchmark_figures.py
```

Optional arguments:

```bash
poetry run python scripts/generate_benchmark_figures.py \
  --input-dir docs/benchmarks \
  --output-dir docs/benchmarks/figures \
  --primary-report latest
```

---

## 7. Classical ONNX Model Benchmarks (PRISM API Pipeline)

This run benchmarks PRISM inference on multiple classical models served through the existing ONNX path (no new runtime adapters).

- Model generation script: `scripts/create_models.py`
- Benchmark script: `scripts/benchmark_classical_models.py`
- Report artifact: `docs/benchmarks/classical-models-latest.json`

Commands used:

```bash
poetry run python scripts/create_models.py
poetry run python scripts/benchmark_classical_models.py \
  --iterations 120 \
  --output-json docs/benchmarks/classical-models-latest.json
```

### Models Covered

1. `logistic_regression`
2. `knn_classifier`
3. `kmeans`
4. `svm_classifier`
5. `random_forest_classifier`
6. `gaussian_nb`

### Latency / Throughput Results

| Model | Avg Latency (ms) | P95 (ms) | Throughput (rps) | Max Abs Deviation |
| --- | ---: | ---: | ---: | ---: |
| `gaussian_nb` | 11.0614 | 26.5580 | 90.40 | 0.0 |
| `kmeans` | 9.7589 | 13.2453 | 102.47 | 0.0 |
| `knn_classifier` | 10.1689 | 14.8328 | 98.34 | 0.0 |
| `logistic_regression` | 7.9149 | 11.1320 | 126.34 | 0.0 |
| `random_forest_classifier` | 9.1507 | 18.1315 | 109.28 | 0.0 |
| `svm_classifier` | 6.2590 | 8.2276 | 159.76 | 0.0 |

### Exact Input / Output Samples and Deviation

Each sample below is the exact JSON payload sent to `POST /models/{model_id}/predict` and the returned response for that payload.

- `gaussian_nb`
  - Input: `{"input": [[0.5532605648040771, 0.21760061383247375, -0.05798998847603798, -2.3189361095428467]]}`
  - Output: `{"predictions": {"output_label": [1], "output_probability": [{"0": 0.42018160223960876, "1": 0.564511239528656, "2": 0.015307186171412468}]}}`
  - Deviation (direct ONNX vs full PRISM pipeline): `compared_values=4`, `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `exact_match=true`

- `kmeans`
  - Input: `{"input": [[1.8267565965652466, -3.07833194732666, 0.9580639600753784, 0.0696372240781784]]}`
  - Output: `{"predictions": {"label": [0], "scores": [[13.505372047424316, 15.882755279541016, 15.406950950622559]]}}`
  - Deviation: `compared_values=4`, `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `exact_match=true`

- `knn_classifier`
  - Input: `{"input": [[0.0341927669942379, 1.3597475290298462, 1.224721074104309, -0.5103070735931396]]}`
  - Output: `{"predictions": {"output_label": [0], "output_probability": [{"0": 1.0, "1": 0.0, "2": 0.0}]}}`
  - Deviation: `compared_values=4`, `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `exact_match=true`

- `logistic_regression`
  - Input: `{"input": [[0.001230153371579945, 0.2987455427646637, -0.27413785457611084, -0.8905918598175049]]}`
  - Output: `{"predictions": {"output_label": [1], "output_probability": [{"0": 0.4413463771343231, "1": 0.5483062267303467, "2": 0.010347440838813782}]}}`
  - Deviation: `compared_values=4`, `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `exact_match=true`

- `random_forest_classifier`
  - Input: `{"input": [[-0.37002477049827576, 0.9940786957740784, 0.4158574938774109, -0.6181637048721313]]}`
  - Output: `{"predictions": {"output_label": [0], "output_probability": [{"0": 0.8942318558692932, "1": 0.10106945037841797, "2": 0.004698507487773895}]}}`
  - Deviation: `compared_values=4`, `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `exact_match=true`

- `svm_classifier`
  - Input: `{"input": [[1.1012624502182007, 0.3384312689304352, -0.5399715304374695, -1.2602418661117554]]}`
  - Output: `{"predictions": {"label": [1], "probabilities": [[1.1936345100402832, 2.2034807205200195, -0.24901126325130463]]}}`
  - Deviation: `compared_values=4`, `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `exact_match=true`

### Deviation Conclusion

For all six classical models, direct model outputs and full PRISM pipeline outputs were numerically identical for benchmarked payloads (`max_abs_diff = 0.0` across all runs), indicating no observed prediction deviation through the PRISM request pipeline.

