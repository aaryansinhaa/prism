"""Benchmark script for load balancing performance.

Tests:
1. Single instance prediction latency
2. Multiple instances round-robin latency
3. Load balancer initialization overhead
4. Health tracking overhead
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.registry.container_registry import (
    register_container_instance,
    get_model_instances,
    load_registry,
    save_registry,
)
from app.services.load_balancer import LoadBalancer, get_load_balancer, reset_load_balancer


def benchmark_load_balancer_initialization():
    """Benchmark load balancer instance registration."""
    print("\n" + "=" * 60)
    print("BENCHMARK 1: Load Balancer Initialization")
    print("=" * 60)

    reset_load_balancer()
    lb = get_load_balancer()

    # Single instance registration
    start = time.perf_counter()
    for _ in range(1000):
        lb.register_instance("model", "v1", f"container", 9001, 0)
    single_instance_time = (time.perf_counter() - start) * 1000
    print(f"1000 single instance registrations: {single_instance_time:.2f}ms")
    print(f"  Average per registration: {single_instance_time / 1000:.4f}ms")

    # Reset and test 10 instances
    reset_load_balancer()
    lb = get_load_balancer()
    start = time.perf_counter()
    for i in range(10):
        lb.register_instance("model", "v1", f"container-{i}", 9001 + i, i)
    multi_instance_time = (time.perf_counter() - start) * 1000
    print(f"\n10 instance registrations: {multi_instance_time:.2f}ms")
    print(f"  Average per registration: {multi_instance_time / 10:.4f}ms")


def benchmark_round_robin_selection():
    """Benchmark round-robin instance selection."""
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Round-Robin Selection")
    print("=" * 60)

    reset_load_balancer()
    lb = get_load_balancer()

    # Register 3 instances
    for i in range(3):
        lb.register_instance("model", "v1", f"container-{i}", 9001 + i, i)

    # Benchmark: 10,000 selections
    start = time.perf_counter()
    for _ in range(10000):
        instance = lb.get_next_instance("model", "v1")
    selection_time = (time.perf_counter() - start) * 1000

    print(f"10,000 round-robin selections (3 instances): {selection_time:.2f}ms")
    print(f"  Average per selection: {selection_time / 10000 * 1000:.4f}μs")

    # Benchmark with 10 instances
    reset_load_balancer()
    lb = get_load_balancer()
    for i in range(10):
        lb.register_instance("model", "v1", f"container-{i}", 9001 + i, i)

    start = time.perf_counter()
    for _ in range(10000):
        instance = lb.get_next_instance("model", "v1")
    selection_time_10 = (time.perf_counter() - start) * 1000

    print(f"\n10,000 round-robin selections (10 instances): {selection_time_10:.2f}ms")
    print(f"  Average per selection: {selection_time_10 / 10000 * 1000:.4f}μs")


def benchmark_registry_operations(tmp_path_str):
    """Benchmark registry read/write operations."""
    print("\n" + "=" * 60)
    print("BENCHMARK 3: Registry Operations")
    print("=" * 60)

    import os
    tmp_path = Path(tmp_path_str)
    os.environ["MODEL_CONTAINER_REGISTRY_PATH"] = str(tmp_path / "registry.json")

    # Benchmark instance registration
    start = time.perf_counter()
    for i in range(100):
        register_container_instance(
            model_id="benchmark_model",
            version="v1",
            container_id=f"container-{i}",
            port=9001 + i,
            instance_index=i,
        )
    registration_time = (time.perf_counter() - start) * 1000
    print(f"100 instance registrations (with I/O): {registration_time:.2f}ms")
    print(f"  Average per registration: {registration_time / 100:.4f}ms")

    # Benchmark instance retrieval
    start = time.perf_counter()
    for _ in range(1000):
        instances = get_model_instances("benchmark_model", "v1")
    retrieval_time = (time.perf_counter() - start) * 1000
    print(f"\n1000 instance retrievals: {retrieval_time:.2f}ms")
    print(f"  Average per retrieval: {retrieval_time / 1000:.4f}ms")


def benchmark_health_tracking():
    """Benchmark health tracking operations."""
    print("\n" + "=" * 60)
    print("BENCHMARK 4: Health Tracking")
    print("=" * 60)

    reset_load_balancer()
    lb = get_load_balancer()

    # Register 10 instances
    for i in range(10):
        lb.register_instance("model", "v1", f"container-{i}", 9001 + i, i)

    state = lb.get_state("model", "v1")

    # Benchmark success marking
    start = time.perf_counter()
    for _ in range(10000):
        for instance in state.instances:
            lb.mark_success("model", instance, "v1")
    success_time = (time.perf_counter() - start) * 1000
    print(f"100,000 success marks (10 instances): {success_time:.2f}ms")
    print(f"  Average per mark: {success_time / 100000 * 1000:.4f}μs")

    # Benchmark failure marking
    start = time.perf_counter()
    for _ in range(10000):
        for instance in state.instances:
            lb.mark_failure("model", instance, "v1")
    failure_time = (time.perf_counter() - start) * 1000
    print(f"\n100,000 failure marks (10 instances): {failure_time:.2f}ms")
    print(f"  Average per mark: {failure_time / 100000 * 1000:.4f}μs")

    # Benchmark health summary
    start = time.perf_counter()
    for _ in range(1000):
        health = lb.get_health_summary("model", "v1")
    health_time = (time.perf_counter() - start) * 1000
    print(f"\n1000 health summaries: {health_time:.2f}ms")
    print(f"  Average per summary: {health_time / 1000:.4f}ms")


def benchmark_comparison():
    """Compare single vs multiple instance routing."""
    print("\n" + "=" * 60)
    print("BENCHMARK 5: Single vs Multiple Instance Routing")
    print("=" * 60)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        import os
        tmp_path = Path(tmp_path)
        os.environ["MODEL_CONTAINER_REGISTRY_PATH"] = str(
            tmp_path / "registry.json"
        )

        # Setup: Single instance
        register_container_instance(
            model_id="single",
            version="v1",
            container_id="container-1",
            port=9001,
            instance_index=0,
        )

        # Benchmark single instance retrieval and selection
        start = time.perf_counter()
        for _ in range(10000):
            instances = get_model_instances("single", "v1")
            if len(instances) == 1:
                port = instances[0]["port"]
        single_time = (time.perf_counter() - start) * 1000
        print(f"10,000 single instance selections: {single_time:.2f}ms")
        print(f"  Average per selection: {single_time / 10000 * 1000:.4f}μs")

        # Setup: Multiple instances
        for i in range(1, 10):
            register_container_instance(
                model_id="multi",
                version="v1",
                container_id=f"container-{i}",
                port=9001 + i,
                instance_index=i,
            )

        # Benchmark multiple instance retrieval and selection
        reset_load_balancer()
        lb = get_load_balancer()
        for i in range(10):
            lb.register_instance("multi", "v1", f"container-{i}", 9001 + i, i)

        start = time.perf_counter()
        for _ in range(10000):
            instances = get_model_instances("multi", "v1")
            if len(instances) > 1:
                instance = lb.get_next_instance("multi", "v1")
                port = instance.port if instance else instances[0]["port"]
        multi_time = (time.perf_counter() - start) * 1000
        print(f"\n10,000 multi-instance selections (10 instances): {multi_time:.2f}ms")
        print(f"  Average per selection: {multi_time / 10000 * 1000:.4f}μs")

        overhead = ((multi_time - single_time) / single_time) * 100
        print(f"\nLoad balancing overhead: {overhead:.2f}%")


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("PRISM-15 LOAD BALANCING BENCHMARK SUITE")
    print("=" * 60)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        benchmark_load_balancer_initialization()
        benchmark_round_robin_selection()
        benchmark_registry_operations(tmp_path)
        benchmark_health_tracking()
        benchmark_comparison()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
