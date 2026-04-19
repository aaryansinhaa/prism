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

