#!/usr/bin/env python3
"""Benchmark bare-metal inference latency for reference models."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import statistics
import joblib
import onnxruntime as ort
import pickle

MODEL_ROOT = Path(__file__).resolve().parent.parent / "model_store"


def _infer_feature_count(value: Any, fallback: int) -> int:
    if hasattr(value, "n_features_in_"):
        return int(value.n_features_in_)
    return fallback


def _infer_onnx_feature_count(session: ort.InferenceSession, fallback: int) -> int:
    for tensor in session.get_inputs():
        for dim in reversed(tensor.shape):
            if isinstance(dim, int) and dim > 0:
                return dim
    return fallback


def _timeit(func, iterations: int) -> Tuple[float, float, List[float]]:
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    total = sum(timings)
    return total, total / iterations, timings


def benchmark_sklearn(model_path: Path, iterations: int) -> Dict[str, Any]:
    try:
        estimator = joblib.load(model_path)
    except Exception:  # noqa: BLE001 - fallback to plain pickle
        with model_path.open("rb") as fh:
            estimator = pickle.load(fh)

    feature_count = _infer_feature_count(estimator, fallback=1)
    sample = np.random.rand(1, feature_count).astype(np.float32)

    estimator.predict(sample)  # warmup
    total, avg, timings = _timeit(lambda: estimator.predict(sample), iterations)

    p95 = statistics.quantiles(timings, n=20)[-1] if len(timings) >= 20 else max(timings)

    return {
        "model": model_path.name,
        "framework": "scikit-learn",
        "iterations": iterations,
        "avg_latency_ms": avg * 1000,
        "p95_latency_ms": p95 * 1000,
        "throughput_rps": iterations / total,
    }


def benchmark_onnx(model_path: Path, iterations: int) -> Dict[str, Any]:
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    feature_count = _infer_onnx_feature_count(session, fallback=1)
    sample = np.random.rand(1, feature_count).astype(np.float32)

    def run():
        session.run(None, {input_name: sample})

    run()
    total, avg, timings = _timeit(run, iterations)

    p95 = statistics.quantiles(timings, n=20)[-1] if len(timings) >= 20 else max(timings)

    return {
        "model": model_path.name,
        "framework": "ONNXRuntime",
        "iterations": iterations,
        "avg_latency_ms": avg * 1000,
        "p95_latency_ms": p95 * 1000,
        "throughput_rps": iterations / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000, help="Number of inference iterations per model")
    args = parser.parse_args()

    sklearn_path = MODEL_ROOT / "linear_regression.pkl"
    onnx_path = MODEL_ROOT / "linear_regression.onnx"

    results = [
        benchmark_sklearn(sklearn_path, args.iterations),
        benchmark_onnx(onnx_path, args.iterations),
    ]

    for item in results:
        print(
            f"{item['framework']:>14} | {item['model']:<22} | avg={item['avg_latency_ms']:.4f} ms | "
            f"p95={item['p95_latency_ms']:.4f} ms | throughput={item['throughput_rps']:.2f} rps"
        )


if __name__ == "__main__":
    main()
