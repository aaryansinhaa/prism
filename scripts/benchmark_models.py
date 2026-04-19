#!/usr/bin/env python3
"""Benchmark PRISM modules and batching performance.

Coverage:
- Runtime adapters (scikit-learn + ONNX bare-metal inference)
- Core module functions (`input_contract`, `access_control`, `request_batcher`)
- API routes (`/`, `/health/monitor`, `/registry`, `/models/{id}/predict`)
- Batching performance comparison (batching disabled vs enabled)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import joblib
import numpy as np
import onnxruntime as ort
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

MODEL_ROOT = Path(__file__).resolve().parent.parent / "model_store"


@dataclass
class Summary:
    name: str
    operations: int
    total_seconds: float
    avg_ms: float
    p95_ms: float
    throughput_rps: float


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * p
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fraction


def _summarize(name: str, durations: Sequence[float], operations: int) -> Summary:
    total = sum(durations)
    avg = total / operations if operations else 0.0
    p95 = _percentile(durations, 0.95)
    throughput = operations / total if total > 0 else float("inf")
    return Summary(
        name=name,
        operations=operations,
        total_seconds=total,
        avg_ms=avg * 1000,
        p95_ms=p95 * 1000,
        throughput_rps=throughput,
    )


def _timeit(func, iterations: int) -> Summary:
    durations: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - start)

    return _summarize(func.__name__, durations, iterations)


async def _timeit_async(coro_factory, iterations: int) -> Summary:
    durations: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        await coro_factory()
        durations.append(time.perf_counter() - start)

    return _summarize(coro_factory.__name__, durations, iterations)


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


def benchmark_runtime_adapters(iterations: int) -> list[Summary]:
    results: list[Summary] = []

    sklearn_path = MODEL_ROOT / "linear_regression.pkl"
    onnx_path = MODEL_ROOT / "linear_regression.onnx"

    try:
        estimator = joblib.load(sklearn_path)
    except Exception:  # noqa: BLE001
        with sklearn_path.open("rb") as fh:
            estimator = pickle.load(fh)

    sklearn_features = _infer_feature_count(estimator, fallback=1)
    sklearn_sample = np.random.rand(1, sklearn_features).astype(np.float32)

    def sklearn_predict():
        estimator.predict(sklearn_sample)

    sklearn_predict()
    summary = _timeit(sklearn_predict, iterations)
    summary.name = "runtime.adapters.sklearn.predict"
    results.append(summary)

    session = ort.InferenceSession(
        onnx_path.as_posix(), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    onnx_features = _infer_onnx_feature_count(session, fallback=1)
    onnx_sample = np.random.rand(1, onnx_features).astype(np.float32)

    def onnx_predict():
        session.run(None, {input_name: onnx_sample})

    onnx_predict()
    summary = _timeit(onnx_predict, iterations)
    summary.name = "runtime.adapters.onnx.predict"
    results.append(summary)

    return results


def _make_request(headers: dict[str, str]) -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/models/test/predict",
        "raw_path": b"/models/test/predict",
        "query_string": b"",
        "headers": [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in headers.items()
        ],
        "client": ("127.0.0.1", 8080),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
    }

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, _receive)


def benchmark_core_modules(iterations: int) -> list[Summary]:
    from app.batching.request_batcher import RequestBatcher
    from app.core import access_control
    from app.core.input_contract import validate_payload_against_expected_input_json

    results: list[Summary] = []

    schema_contract = json.dumps(
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
    )
    payload = {"input": [[1.0, 2.0], [3.0, 4.0]]}

    def input_contract_validate():
        is_valid, error = validate_payload_against_expected_input_json(
            schema_contract, payload
        )
        if not is_valid:
            raise RuntimeError(f"unexpected validation failure: {error}")

    summary = _timeit(input_contract_validate, iterations)
    summary.name = "app.core.input_contract.validate_payload"
    results.append(summary)

    os.environ["PRISM_API_KEYS"] = "benchmark-key"
    request = _make_request({"x-api-key": "benchmark-key"})

    def access_control_validate_key():
        access_control.validate_api_key(request)

    summary = _timeit(access_control_validate_key, iterations)
    summary.name = "app.core.access_control.validate_api_key"
    results.append(summary)

    previous_requests = os.environ.get("PRISM_RATE_LIMIT_REQUESTS")
    previous_window = os.environ.get("PRISM_RATE_LIMIT_WINDOW_SECONDS")
    os.environ["PRISM_RATE_LIMIT_REQUESTS"] = str(max(iterations * 2, 1000))
    os.environ["PRISM_RATE_LIMIT_WINDOW_SECONDS"] = "60"

    def access_control_rate_limit():
        principal = f"principal-{time.perf_counter_ns()}"
        access_control.enforce_rate_limit(principal)

    summary = _timeit(access_control_rate_limit, iterations)
    summary.name = "app.core.access_control.enforce_rate_limit"
    results.append(summary)

    access_control._rate_limit_buckets.clear()
    if previous_requests is None:
        os.environ.pop("PRISM_RATE_LIMIT_REQUESTS", None)
    else:
        os.environ["PRISM_RATE_LIMIT_REQUESTS"] = previous_requests
    if previous_window is None:
        os.environ.pop("PRISM_RATE_LIMIT_WINDOW_SECONDS", None)
    else:
        os.environ["PRISM_RATE_LIMIT_WINDOW_SECONDS"] = previous_window

    async def fake_forwarder(
        _url: str, payload_value: Dict[str, Any]
    ) -> Dict[str, Any]:
        rows = payload_value["input"]
        return {"predictions": [1.0 for _ in rows]}

    async def request_batcher_forward_once() -> None:
        batcher = RequestBatcher(batch_window_ms=0, forwarder=fake_forwarder)
        await batcher.forward("http://127.0.0.1:9999/predict", {"input": [[1.0, 2.0]]})

    summary = asyncio.run(_timeit_async(request_batcher_forward_once, iterations))
    summary.name = "app.batching.request_batcher.forward"
    results.append(summary)

    return results


class _ForwardNoBatch:
    def __init__(self, forwarder) -> None:
        self._forwarder = forwarder

    async def forward(
        self, container_url: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await self._forwarder(container_url, payload)


async def _benchmark_inference_burst(
    *,
    app,
    total_requests: int,
    concurrency: int,
    headers: Dict[str, str],
    model_id: str,
) -> Summary:
    semaphore = asyncio.Semaphore(concurrency)
    durations: list[float] = []

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:

        async def one_call() -> None:
            async with semaphore:
                start = time.perf_counter()
                response = await client.post(
                    f"/models/{model_id}/predict",
                    json={"input": [[1.0, 2.0, 3.0, 4.0]]},
                    headers=headers,
                )
                durations.append(time.perf_counter() - start)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"inference benchmark failed: {response.status_code} {response.text}"
                    )

        await asyncio.gather(*[one_call() for _ in range(total_requests)])

    return _summarize("inference_burst", durations, total_requests)


def benchmark_api_routes(
    iterations: int,
    burst_requests: int,
    burst_concurrency: int,
    simulated_container_latency_ms: float,
    simulated_container_per_row_ms: float,
) -> tuple[list[Summary], dict[str, Any]]:
    from app.batching.request_batcher import RequestBatcher
    from app.main import app
    import app.routing.inference as inference_module

    results: list[Summary] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        registry_file = tmp_path / "containers.json"
        model_id = "benchmark_model"
        registry_file.write_text(
            json.dumps(
                {
                    "models": {
                        model_id: {
                            "model_id": model_id,
                            "container_id": "benchmark-container",
                            "port": 9999,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        os.environ["MODEL_CONTAINER_REGISTRY_PATH"] = registry_file.as_posix()
        os.environ["PRISM_API_KEYS"] = "benchmark-key"
        os.environ["PRISM_RATE_LIMIT_REQUESTS"] = str(max(burst_requests * 10, 10000))
        os.environ["PRISM_RATE_LIMIT_WINDOW_SECONDS"] = "60"

        with TestClient(app) as client:

            def route_root():
                response = client.get("/")
                if response.status_code != 200:
                    raise RuntimeError(response.text)

            summary = _timeit(route_root, iterations)
            summary.name = "app.routing.health.read_root"
            results.append(summary)

            def route_monitor():
                response = client.get("/health/monitor")
                if response.status_code != 200:
                    raise RuntimeError(response.text)

            summary = _timeit(route_monitor, iterations)
            summary.name = "app.routing.health.monitor_health"
            results.append(summary)

            def route_registry_list():
                response = client.get("/registry")
                if response.status_code != 200:
                    raise RuntimeError(response.text)

            summary = _timeit(route_registry_list, iterations)
            summary.name = "app.routing.registry.list_registry"
            results.append(summary)

        container_lock = asyncio.Lock()

        async def simulated_container_forward(
            _url: str, payload: Dict[str, Any]
        ) -> Dict[str, Any]:
            rows = payload.get("input", [])
            row_count = len(rows)
            async with container_lock:
                await asyncio.sleep(
                    (simulated_container_latency_ms / 1000.0)
                    + ((simulated_container_per_row_ms * row_count) / 1000.0)
                )
            return {"predictions": [1.0 for _ in rows]}

        original_batcher = inference_module.request_batcher

        headers = {"X-API-Key": "benchmark-key"}
        no_batcher = _ForwardNoBatch(simulated_container_forward)
        inference_module.request_batcher = no_batcher
        no_batch_summary = asyncio.run(
            _benchmark_inference_burst(
                app=app,
                total_requests=burst_requests,
                concurrency=burst_concurrency,
                headers=headers,
                model_id=model_id,
            )
        )
        no_batch_summary.name = "app.routing.inference.predict_model (batching=off)"
        results.append(no_batch_summary)

        batched = RequestBatcher(
            batch_window_ms=50, forwarder=simulated_container_forward
        )
        inference_module.request_batcher = batched
        batch_summary = asyncio.run(
            _benchmark_inference_burst(
                app=app,
                total_requests=burst_requests,
                concurrency=burst_concurrency,
                headers=headers,
                model_id=model_id,
            )
        )
        batch_summary.name = "app.routing.inference.predict_model (batching=on)"
        results.append(batch_summary)

        inference_module.request_batcher = original_batcher

    delta_latency_pct = 0.0
    if no_batch_summary.avg_ms > 0:
        delta_latency_pct = (
            (no_batch_summary.avg_ms - batch_summary.avg_ms) / no_batch_summary.avg_ms
        ) * 100.0

    delta_throughput_pct = 0.0
    if no_batch_summary.throughput_rps > 0:
        delta_throughput_pct = (
            (batch_summary.throughput_rps - no_batch_summary.throughput_rps)
            / no_batch_summary.throughput_rps
        ) * 100.0

    comparison = {
        "baseline": {
            "avg_latency_ms": no_batch_summary.avg_ms,
            "p95_latency_ms": no_batch_summary.p95_ms,
            "throughput_rps": no_batch_summary.throughput_rps,
        },
        "batched": {
            "avg_latency_ms": batch_summary.avg_ms,
            "p95_latency_ms": batch_summary.p95_ms,
            "throughput_rps": batch_summary.throughput_rps,
        },
        "improvement": {
            "avg_latency_reduction_pct": delta_latency_pct,
            "throughput_increase_pct": delta_throughput_pct,
        },
        "burst": {
            "requests": burst_requests,
            "concurrency": burst_concurrency,
            "simulated_container_latency_ms": simulated_container_latency_ms,
            "simulated_container_per_row_ms": simulated_container_per_row_ms,
        },
    }

    return results, comparison


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_summaries(summaries: Iterable[Summary]) -> None:
    for item in summaries:
        print(
            f"{item.name:<56} "
            f"avg={item.avg_ms:>8.4f} ms | "
            f"p95={item.p95_ms:>8.4f} ms | "
            f"throughput={item.throughput_rps:>10.2f} rps | "
            f"ops={item.operations}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Iterations per sync micro-benchmark",
    )
    parser.add_argument(
        "--burst-requests",
        type=int,
        default=200,
        help="Total requests for inference burst benchmark",
    )
    parser.add_argument(
        "--burst-concurrency",
        type=int,
        default=50,
        help="Concurrent requests for inference burst benchmark",
    )
    parser.add_argument(
        "--simulated-container-latency-ms",
        type=float,
        default=2.0,
        help="Simulated single container forward latency for batching comparison",
    )
    parser.add_argument(
        "--simulated-container-per-row-ms",
        type=float,
        default=0.05,
        help="Additional simulated processing per row in a batched container call",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional file path to write benchmark report JSON",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTEST_CURRENT_TEST", "benchmark")

    _print_section("Runtime Adapter Benchmarks")
    runtime_results = benchmark_runtime_adapters(args.iterations)
    _print_summaries(runtime_results)

    _print_section("Core Module Benchmarks")
    core_results = benchmark_core_modules(args.iterations)
    _print_summaries(core_results)

    _print_section("API Route Benchmarks")
    route_results, batching_comparison = benchmark_api_routes(
        iterations=args.iterations,
        burst_requests=args.burst_requests,
        burst_concurrency=args.burst_concurrency,
        simulated_container_latency_ms=args.simulated_container_latency_ms,
        simulated_container_per_row_ms=args.simulated_container_per_row_ms,
    )
    _print_summaries(route_results)

    _print_section("Batching Performance Delta")
    print(
        "avg latency reduction: "
        f"{batching_comparison['improvement']['avg_latency_reduction_pct']:.2f}% | "
        "throughput increase: "
        f"{batching_comparison['improvement']['throughput_increase_pct']:.2f}%"
    )

    report = {
        "runtime": [asdict(summary) for summary in runtime_results],
        "core": [asdict(summary) for summary in core_results],
        "routes": [asdict(summary) for summary in route_results],
        "batching_comparison": batching_comparison,
        "meta": {
            "iterations": args.iterations,
            "burst_requests": args.burst_requests,
            "burst_concurrency": args.burst_concurrency,
            "simulated_container_latency_ms": args.simulated_container_latency_ms,
            "simulated_container_per_row_ms": args.simulated_container_per_row_ms,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "onnxruntime_version": ort.__version__,
        },
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {args.output_json}")


if __name__ == "__main__":
    main()
