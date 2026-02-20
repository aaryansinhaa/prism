# PRISM Benchmarks

This document tracks reproducible inference benchmarks for PRISM artifacts. The current measurements focus on "bare metal" execution (no container overhead) to provide a regression budget for future runtime layers.

---

## 1. Methodology

- Script: `scripts/benchmark_models.py`
- Command: `poetry run python scripts/benchmark_models.py --iterations 2000`
- Metrics per model:
  - Average latency (ms)
  - P95 latency (ms)
  - Throughput (requests per second)
- Each run performs a warm-up call before timing.

---

## 2. Environment

- OS: Linux 6.12.63-1-lts
- CPU: 12th Gen Intel(R) Core(TM) i5-12500H (x86_64)
- Python: 3.13.11 via Poetry
- Providers: scikit-learn 1.8.0, onnxruntime CPUExecutionProvider

> Note: Loading `linear_regression.pkl` triggers an `InconsistentVersionWarning` because it was serialized with scikit-learn 1.6.1. Re-exporting with the current version will eliminate the warning.

---

## 3. Results (Bare Metal)

| Model Artifact | Framework | Iterations | Avg Latency (ms) | P95 Latency (ms) | Throughput (req/s) |
| --- | --- | --- | --- | --- | --- |
| `linear_regression.pkl` | scikit-learn | 2000 | 0.0412 | 0.0458 | 24,270 |
| `linear_regression.onnx` | ONNX Runtime | 2000 | 0.0081 | 0.0115 | 123,551 |

Interpretation:

- ONNXRuntime (with CPUExecutionProvider) delivers ~5x lower latency than the scikit-learn estimator for this model.
- These numbers represent idealized host execution and therefore serve as the baseline when measuring PRISM container or routing overhead.

---

## 4. Next Steps

1. Re-export the scikit-learn artifact with version 1.8.0+ to remove warnings.
2. Add containerized benchmarks once the runtime orchestration is wired up.
3. Extend the script with dataset-driven inputs to complement the random feature vectors currently used.
