"""Service for collecting and managing model performance metrics."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class MetricsSnapshot:
    """A single snapshot of metrics at a point in time."""

    timestamp: float
    latency_ms: float  # milliseconds
    throughput_rps: float  # requests per second
    requests_count: int  # total request count
    error_count: int = 0
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None


@dataclass
class ModelMetrics:
    """Aggregated metrics for a model."""

    model_id: str
    labels: list[str]  # Time labels (e.g., "1", "2", "3", ...)
    latency: list[float]  # Latency values in ms
    throughput: list[float]  # Throughput values in RPS
    requests: list[int]  # Request counts
    error_rate: Optional[list[float]] = None  # Error percentages
    cpu_usage: Optional[list[float]] = None  # CPU percentages
    memory_usage: Optional[list[float]] = None  # Memory in MB


class MetricsCollector:
    """Collects and manages metrics for a model container."""

    def __init__(self, model_id: str, container_id: str, window_size: int = 60):
        """Initialize metrics collector.

        Args:
            model_id: The model identifier
            container_id: Docker container ID
            window_size: Number of data points to keep (default 60 seconds)
        """
        self.model_id = model_id
        self.container_id = container_id
        self.window_size = window_size
        self._metrics: deque[MetricsSnapshot] = deque(maxlen=window_size)
        self._request_counter = 0
        self._error_counter = 0
        self._last_latency = 0.0
        self._last_throughput = 0.0

    def record_request(
        self,
        latency_ms: float,
        success: bool = True,
        cpu_percent: Optional[float] = None,
        memory_mb: Optional[float] = None,
    ) -> None:
        """Record a single request metric.

        Args:
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
        """
        self._request_counter += 1
        if not success:
            self._error_counter += 1

        self._last_latency = latency_ms

        snapshot = MetricsSnapshot(
            timestamp=time.time(),
            latency_ms=latency_ms,
            throughput_rps=0.0,  # Will be calculated by aggregate
            requests_count=self._request_counter,
            error_count=self._error_counter,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
        )
        self._metrics.append(snapshot)

    def get_aggregated_metrics(self) -> ModelMetrics:
        """Get aggregated metrics for dashboard display.

        Returns:
            ModelMetrics with time-series data suitable for charting
        """
        if not self._metrics:
            # Return empty metrics with sensible defaults
            return ModelMetrics(
                model_id=self.model_id,
                labels=list(str(i + 1) for i in range(20)),
                latency=[0.0] * 20,
                throughput=[0.0] * 20,
                requests=[0] * 20,
                error_rate=[0.0] * 20,
                cpu_usage=[0.0] * 20,
                memory_usage=[0.0] * 20,
            )

        snapshots = list(self._metrics)
        n = len(snapshots)

        # Create labels (1-indexed)
        labels = [str(i + 1) for i in range(n)]

        # Extract raw values
        latencies = [float(s.latency_ms) for s in snapshots]
        requests = [s.requests_count for s in snapshots]
        # error_counts = [s.error_count for s in snapshots]

        # Calculate throughput (requests per second in the window)
        throughputs = []
        for i, snapshot in enumerate(snapshots):
            if i == 0:
                throughputs.append(0.0)
            else:
                prev_snapshot = snapshots[i - 1]
                time_delta = snapshot.timestamp - prev_snapshot.timestamp
                req_delta = snapshot.requests_count - prev_snapshot.requests_count
                if time_delta > 0:
                    throughputs.append(req_delta / time_delta)
                else:
                    throughputs.append(0.0)

        # Calculate error rates
        error_rates = []
        for i, snapshot in enumerate(snapshots):
            if snapshot.requests_count == 0:
                error_rates.append(0.0)
            else:
                error_rate = snapshot.error_count / snapshot.requests_count * 100
                error_rates.append(min(error_rate, 100.0))

        # Extract CPU and memory if available
        cpu_usages = (
            [float(s.cpu_percent or 0.0) for s in snapshots]
            if any(s.cpu_percent is not None for s in snapshots)
            else None
        )
        memory_usages = (
            [float(s.memory_mb or 0.0) for s in snapshots]
            if any(s.memory_mb is not None for s in snapshots)
            else None
        )

        return ModelMetrics(
            model_id=self.model_id,
            labels=labels,
            latency=latencies,
            throughput=throughputs,
            requests=requests,
            error_rate=error_rates,
            cpu_usage=cpu_usages,
            memory_usage=memory_usages,
        )

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._request_counter = 0
        self._error_counter = 0


class MetricsRegistry:
    """Central registry for tracking metrics across all models."""

    _instance = None
    _collectors: dict[str, MetricsCollector] = {}

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> MetricsRegistry:
        """Get singleton instance."""
        return cls()

    def register_model(self, model_id: str, container_id: str) -> MetricsCollector:
        """Register a model for metrics collection.

        Args:
            model_id: Model identifier
            container_id: Docker container ID

        Returns:
            MetricsCollector instance for the model
        """
        if model_id not in self._collectors:
            self._collectors[model_id] = MetricsCollector(model_id, container_id)
        return self._collectors[model_id]

    def get_collector(self, model_id: str) -> Optional[MetricsCollector]:
        """Get metrics collector for a model.

        Args:
            model_id: Model identifier

        Returns:
            MetricsCollector or None if not registered
        """
        return self._collectors.get(model_id)

    def record_request(
        self,
        model_id: str,
        latency_ms: float,
        success: bool = True,
        cpu_percent: Optional[float] = None,
        memory_mb: Optional[float] = None,
    ) -> None:
        """Record a request for a model.

        Args:
            model_id: Model identifier
            latency_ms: Latency in milliseconds
            success: Whether request succeeded
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
        """
        collector = self._collectors.get(model_id)
        if collector:
            collector.record_request(
                latency_ms, success, cpu_percent=cpu_percent, memory_mb=memory_mb
            )

    def get_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        """Get aggregated metrics for a model.

        Args:
            model_id: Model identifier

        Returns:
            ModelMetrics or None if not registered
        """
        collector = self.get_collector(model_id)
        if collector:
            return collector.get_aggregated_metrics()
        return None

    def remove_model(self, model_id: str) -> None:
        """Remove metrics for a model.

        Args:
            model_id: Model identifier
        """
        self._collectors.pop(model_id, None)

    def get_all_metrics(self) -> dict[str, ModelMetrics]:
        """Get metrics for all models.

        Returns:
            Dictionary mapping model IDs to their metrics
        """
        result: dict[str, ModelMetrics] = {}
        for model_id in self._collectors.keys():
            metrics = self.get_metrics(model_id)
            if metrics is not None:
                result[model_id] = metrics
        return result
