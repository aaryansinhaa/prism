#!/usr/bin/env python3
"""Benchmark PRISM inference pipeline across multiple classical ONNX models.

Runs end-to-end API benchmark calls through `POST /models/{model_id}/predict`
using the existing PRISM inference route and ONNX runtime adapters.

Outputs:
- Console summary
- JSON report with exact request payloads, outputs, and deviation metrics
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from runtime.model_loaders import load_model

os.environ.setdefault("PRISM_ENABLE_HEALTH_MONITOR", "false")

from app.main import app

ROOT = Path(__file__).resolve().parents[1]
MODEL_STORE = ROOT / "model_store"


@dataclass
class ModelBenchmark:
    model_id: str
    iterations: int
    avg_latency_ms: float
    p95_latency_ms: float
    throughput_rps: float
    sample_input: dict[str, Any]
    sample_api_output: dict[str, Any]
    sample_direct_output: dict[str, Any]
    deviation: dict[str, Any]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * p
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fraction


def _flatten_numerics(value: Any) -> list[float]:
    numbers: list[float] = []

    if isinstance(value, dict):
        for item in value.values():
            numbers.extend(_flatten_numerics(item))
        return numbers

    if isinstance(value, list):
        for item in value:
            numbers.extend(_flatten_numerics(item))
        return numbers

    if isinstance(value, (int, float, np.integer, np.floating)):
        numbers.append(float(value))

    return numbers


def _deviation_metrics(expected: Any, actual: Any) -> dict[str, Any]:
    expected_flat = _flatten_numerics(expected)
    actual_flat = _flatten_numerics(actual)

    compared = min(len(expected_flat), len(actual_flat))
    if compared == 0:
        exact_match = expected == actual
        return {
            "compared_values": 0,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "exact_match": exact_match,
        }

    diffs = [abs(expected_flat[i] - actual_flat[i]) for i in range(compared)]
    return {
        "compared_values": compared,
        "max_abs_diff": float(max(diffs)),
        "mean_abs_diff": float(sum(diffs) / compared),
        "exact_match": expected == actual,
    }


def _canonical_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _discover_models() -> list[str]:
    ids: list[str] = []
    for path in sorted(MODEL_STORE.glob("*.onnx")):
        model_id = path.stem
        if model_id in {"linear_regression", "decision_trees"}:
            continue
        ids.append(model_id)
    return ids


def _sample_input_for_model(model_id: str) -> dict[str, Any]:
    seed_map = {
        "logistic_regression": 7,
        "knn_classifier": 11,
        "kmeans": 13,
        "svm_classifier": 17,
        "random_forest_classifier": 19,
        "gaussian_nb": 23,
    }
    seed = seed_map.get(model_id, 29)
    rng = np.random.default_rng(seed)
    row = rng.normal(loc=0.0, scale=1.0, size=(1, 4)).astype(np.float32)
    return {"input": row.tolist()}


def _prepare_registry(model_ids: list[str], registry_path: Path) -> dict[int, str]:
    models: dict[str, dict[str, Any]] = {}
    port_to_model: dict[int, str] = {}
    for idx, model_id in enumerate(model_ids):
        port = 9100 + idx
        port_to_model[port] = model_id
        models[model_id] = {
            "model_id": model_id,
            "container_id": f"benchmark-{model_id}",
            "port": port,
            "expected_input_json": json.dumps(
                {
                    "type": "object",
                    "required": ["input"],
                    "properties": {
                        "input": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                        }
                    },
                    "additionalProperties": False,
                }
            ),
        }

    registry_payload = {"models": models}
    registry_path.write_text(json.dumps(registry_payload, indent=2), encoding="utf-8")
    return port_to_model


def benchmark_models(iterations: int) -> list[ModelBenchmark]:
    import app.routing.inference as inference_module

    model_ids = _discover_models()
    if not model_ids:
        raise RuntimeError(
            "No classical ONNX models found. Run scripts/create_models.py first."
        )

    loaded = {
        model_id: load_model(MODEL_STORE / f"{model_id}.onnx") for model_id in model_ids
    }

    reports: list[ModelBenchmark] = []

    os.environ["PRISM_API_KEYS"] = "benchmark-key"
    os.environ["PRISM_RATE_LIMIT_REQUESTS"] = str(
        max(iterations * len(model_ids) * 2, 2000)
    )
    os.environ["PRISM_RATE_LIMIT_WINDOW_SECONDS"] = "60"

    temp_registry_dir = tempfile.TemporaryDirectory()
    benchmark_registry_path = (
        Path(temp_registry_dir.name) / "classical_models_registry.json"
    )
    port_to_model = _prepare_registry(model_ids, benchmark_registry_path)
    os.environ["MODEL_CONTAINER_REGISTRY_PATH"] = str(benchmark_registry_path)

    class _LocalForwarder:
        async def forward(
            self, container_url: str, payload: dict[str, Any]
        ) -> dict[str, Any]:
            port = int(container_url.rsplit(":", 1)[1].split("/")[0])
            model_id = port_to_model[port]
            model = loaded[model_id]
            validated = model.validate_input(payload)
            return model.predict(validated)

    original_batcher = inference_module.request_batcher
    inference_module.request_batcher = _LocalForwarder()

    headers = {"X-API-Key": "benchmark-key"}

    try:
        with TestClient(app) as client:
            for model_id in model_ids:
                payload = _sample_input_for_model(model_id)
                model = loaded[model_id]

                direct_output = model.predict(model.validate_input(payload))
                response = client.post(
                    f"/models/{model_id}/predict",
                    json=payload,
                    headers=headers,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Initial inference failed for {model_id}: {response.status_code} {response.text}"
                    )
                api_output = response.json()
                direct_output = _canonical_json(direct_output)
                api_output = _canonical_json(api_output)

                durations: list[float] = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    run_response = client.post(
                        f"/models/{model_id}/predict",
                        json=payload,
                        headers=headers,
                    )
                    durations.append(time.perf_counter() - start)
                    if run_response.status_code != 200:
                        raise RuntimeError(
                            f"Benchmark failed for {model_id}: {run_response.status_code} {run_response.text}"
                        )

                total = sum(durations)
                avg_ms = (total / iterations) * 1000.0
                p95_ms = _percentile(durations, 0.95) * 1000.0
                throughput = iterations / total if total > 0 else float("inf")
                deviation = _deviation_metrics(direct_output, api_output)

                reports.append(
                    ModelBenchmark(
                        model_id=model_id,
                        iterations=iterations,
                        avg_latency_ms=avg_ms,
                        p95_latency_ms=p95_ms,
                        throughput_rps=throughput,
                        sample_input=payload,
                        sample_api_output=api_output,
                        sample_direct_output=direct_output,
                        deviation=deviation,
                    )
                )
    finally:
        inference_module.request_batcher = original_batcher
        temp_registry_dir.cleanup()

    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "docs" / "benchmarks" / "classical-models-latest.json",
    )
    args = parser.parse_args()

    results = benchmark_models(iterations=args.iterations)

    print("\nClassical Model Benchmark Summary")
    print("=" * 72)
    for item in results:
        print(
            f"{item.model_id:<28} avg={item.avg_latency_ms:>8.3f} ms "
            f"p95={item.p95_latency_ms:>8.3f} ms throughput={item.throughput_rps:>9.2f} rps"
        )

    report_payload = {
        "meta": {
            "iterations": args.iterations,
            "models_count": len(results),
        },
        "models": [asdict(item) for item in results],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark report: {args.output_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
