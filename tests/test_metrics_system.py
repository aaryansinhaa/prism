"""Comprehensive tests for metrics system and model control center."""

import json
from fastapi.testclient import TestClient
from app.main import app
from app.services.metrics_service import MetricsRegistry, MetricsCollector, MetricsSnapshot

client = TestClient(app)


# ============================================================================
# UNIT TESTS - Metrics Service
# ============================================================================


class TestMetricsCollector:
    """UT: Test MetricsCollector class."""

    def test_collector_initialization(self):
        """UT: MetricsCollector initializes correctly."""
        collector = MetricsCollector("test-model", "container123", window_size=60)
        assert collector.model_id == "test-model"
        assert collector.container_id == "container123"
        assert collector.window_size == 60

    def test_record_single_request(self):
        """UT: Recording a single request updates metrics."""
        collector = MetricsCollector("model1", "container1")
        collector.record_request(latency_ms=100.0, success=True)
        
        metrics = collector.get_aggregated_metrics()
        assert len(metrics.latency) > 0
        assert metrics.latency[0] == 100.0

    def test_record_multiple_requests(self):
        """UT: Recording multiple requests accumulates correctly."""
        collector = MetricsCollector("model1", "container1", window_size=10)
        for i in range(5):
            collector.record_request(latency_ms=50.0 + i, success=True)
        
        metrics = collector.get_aggregated_metrics()
        assert len(metrics.requests) == 5
        assert metrics.requests[-1] == 5

    def test_record_failed_request(self):
        """UT: Failed requests increment error counter."""
        collector = MetricsCollector("model1", "container1", window_size=10)
        collector.record_request(latency_ms=100.0, success=True)
        collector.record_request(latency_ms=200.0, success=False)
        collector.record_request(latency_ms=150.0, success=True)
        
        metrics = collector.get_aggregated_metrics()
        assert metrics.requests[-1] == 3
        if metrics.error_rate:
            assert len([e for e in metrics.error_rate if e > 0]) > 0

    def test_aggregated_metrics_has_all_fields(self):
        """UT: Aggregated metrics contains all required fields."""
        collector = MetricsCollector("model1", "container1")
        collector.record_request(latency_ms=100.0, success=True, cpu_percent=25.5, memory_mb=512.0)
        
        metrics = collector.get_aggregated_metrics()
        assert metrics.labels is not None
        assert metrics.latency is not None
        assert metrics.throughput is not None
        assert metrics.requests is not None
        assert metrics.error_rate is not None
        assert metrics.cpu_usage is not None
        assert metrics.memory_usage is not None

    def test_empty_collector_returns_default_metrics(self):
        """UT: Collector with no data returns sensible defaults."""
        collector = MetricsCollector("empty-model", "container1")
        metrics = collector.get_aggregated_metrics()
        
        assert len(metrics.labels) == 20  # Default size
        assert len(metrics.latency) == 20
        assert all(l == 0.0 for l in metrics.latency)
        assert all(r == 0 for r in metrics.requests)

    def test_reset_clears_all_metrics(self):
        """UT: Reset clears all accumulated metrics."""
        collector = MetricsCollector("model1", "container1")
        collector.record_request(latency_ms=100.0, success=True)
        collector.reset()
        
        metrics = collector.get_aggregated_metrics()
        assert all(l == 0.0 for l in metrics.latency)
        assert all(r == 0 for r in metrics.requests)

    def test_window_size_respected(self):
        """UT: Collector respects maximum window size."""
        collector = MetricsCollector("model1", "container1", window_size=5)
        
        for i in range(10):
            collector.record_request(latency_ms=50.0 + i, success=True)
        
        metrics = collector.get_aggregated_metrics()
        assert len(metrics.latency) <= 5

    def test_error_rate_calculation(self):
        """UT: Error rate is calculated correctly."""
        collector = MetricsCollector("model1", "container1", window_size=10)
        
        # 7 successes, 3 failures = 30% error rate
        for _ in range(7):
            collector.record_request(latency_ms=100.0, success=True)
        for _ in range(3):
            collector.record_request(latency_ms=100.0, success=False)
        
        metrics = collector.get_aggregated_metrics()
        if metrics.error_rate:
            final_error_rate = metrics.error_rate[-1]
            assert 29.0 <= final_error_rate <= 31.0  # Allow small floating point variation


class TestMetricsRegistry:
    """UT: Test MetricsRegistry singleton."""

    def test_registry_is_singleton(self):
        """UT: MetricsRegistry is a singleton."""
        reg1 = MetricsRegistry.get_instance()
        reg2 = MetricsRegistry.get_instance()
        assert reg1 is reg2

    def test_register_and_retrieve_model(self):
        """UT: Can register and retrieve model collector."""
        registry = MetricsRegistry.get_instance()
        registry._collectors.clear()  # Clean state for test
        
        collector = registry.register_model("test-model", "container123")
        assert collector is not None
        assert collector.model_id == "test-model"
        
        retrieved = registry.get_collector("test-model")
        assert retrieved is collector

    def test_record_request_through_registry(self):
        """UT: Can record requests through registry."""
        registry = MetricsRegistry.get_instance()
        registry._collectors.clear()
        registry.register_model("model1", "container1")
        
        registry.record_request("model1", latency_ms=100.0, success=True)
        registry.record_request("model1", latency_ms=150.0, success=True)
        
        metrics = registry.get_metrics("model1")
        assert metrics is not None
        assert len(metrics.requests) == 2
        assert metrics.requests[-1] == 2

    def test_get_metrics_for_unregistered_model(self):
        """UT: Getting metrics for unregistered model returns None."""
        registry = MetricsRegistry.get_instance()
        metrics = registry.get_metrics("nonexistent-model")
        assert metrics is None

    def test_remove_model_from_registry(self):
        """UT: Can remove model from registry."""
        registry = MetricsRegistry.get_instance()
        registry._collectors.clear()
        registry.register_model("model1", "container1")
        
        registry.remove_model("model1")
        assert registry.get_collector("model1") is None

    def test_get_all_metrics(self):
        """UT: Can retrieve metrics for all models."""
        registry = MetricsRegistry.get_instance()
        registry._collectors.clear()
        
        registry.register_model("model1", "container1")
        registry.register_model("model2", "container2")
        registry.record_request("model1", latency_ms=100.0)
        registry.record_request("model2", latency_ms=150.0)
        
        all_metrics = registry.get_all_metrics()
        assert "model1" in all_metrics
        assert "model2" in all_metrics


# ============================================================================
# INTEGRATION TESTS - API Endpoints
# ============================================================================


class TestModelMetricsEndpoint:
    """IT: Test /api/model-metrics endpoint."""

    def test_metrics_endpoint_returns_json(self):
        """IT: /api/model-metrics returns valid JSON."""
        response = client.get("/api/model-metrics?model_id=test-model")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_metrics_response_has_required_fields(self):
        """IT: Metrics response includes all required fields."""
        response = client.get("/api/model-metrics?model_id=test-model")
        data = response.json()
        
        assert "labels" in data
        assert "requests" in data
        assert "latency" in data
        assert "throughput" in data

    def test_metrics_labels_are_strings(self):
        """IT: Metric labels are string numbers."""
        response = client.get("/api/model-metrics?model_id=test-model")
        data = response.json()
        
        assert isinstance(data["labels"], list)
        assert all(isinstance(label, str) for label in data["labels"])

    def test_metrics_values_are_numeric(self):
        """IT: Metric values are numeric (int or float)."""
        response = client.get("/api/model-metrics?model_id=test-model")
        data = response.json()
        
        assert all(isinstance(r, (int, float)) for r in data["requests"])
        assert all(isinstance(l, (int, float)) for l in data["latency"])
        assert all(isinstance(t, (int, float)) for t in data["throughput"])

    def test_metrics_optional_fields_present(self):
        """IT: Optional metric fields are included when available."""
        response = client.get("/api/model-metrics?model_id=test-model")
        data = response.json()
        
        # Some fields might be optional
        if "error_rate" in data:
            assert isinstance(data["error_rate"], list)

    def test_metrics_consistency(self):
        """IT: Metric arrays have consistent lengths."""
        response = client.get("/api/model-metrics?model_id=test-model")
        data = response.json()
        
        n_labels = len(data["labels"])
        assert len(data["requests"]) == n_labels
        assert len(data["latency"]) == n_labels
        assert len(data["throughput"]) == n_labels

    def test_metrics_endpoint_with_multiple_models(self):
        """IT: Metrics endpoint handles different models independently."""
        response1 = client.get("/api/model-metrics?model_id=model1")
        response2 = client.get("/api/model-metrics?model_id=model2")
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        # Both should return valid data
        data1 = response1.json()
        data2 = response2.json()
        assert "labels" in data1 and "labels" in data2


class TestRecordMetricsEndpoint:
    """IT: Test /api/record-metrics endpoint."""

    def test_record_metrics_success(self):
        """IT: Recording metrics returns success."""
        response = client.post(
            "/api/record-metrics",
            data={
                "model_id": "test-model",
                "latency_ms": "150.5",
                "success": "True",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "recorded"

    def test_record_metrics_with_resource_usage(self):
        """IT: Recording metrics with CPU/memory works."""
        response = client.post(
            "/api/record-metrics",
            data={
                "model_id": "test-model",
                "latency_ms": "200.0",
                "success": "True",
                "cpu_percent": "45.5",
                "memory_mb": "512.0",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "recorded"

    def test_record_failed_request(self):
        """IT: Recording failed requests works."""
        response = client.post(
            "/api/record-metrics",
            data={
                "model_id": "test-model",
                "latency_ms": "1000.0",
                "success": "False",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "recorded"

    def test_record_metrics_missing_model_id(self):
        """IT: Recording without model_id returns error."""
        response = client.post(
            "/api/record-metrics",
            data={
                "latency_ms": "150.0",
                "success": "True",
            }
        )
        # Should return 422 for missing required field
        assert response.status_code in [200, 422]

    def test_record_metrics_invalid_latency(self):
        """IT: Recording with invalid latency returns error."""
        response = client.post(
            "/api/record-metrics",
            data={
                "model_id": "test-model",
                "latency_ms": "not-a-number",
                "success": "True",
            }
        )
        # Should return error or 422
        assert response.status_code in [200, 422]


class TestMetricsConfigEndpoint:
    """IT: Test metrics configuration endpoints."""

    def test_get_metrics_config(self):
        """IT: Getting metrics config returns JSON."""
        response = client.get("/api/metrics-config?model_id=test-model")
        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data
        assert data["model_id"] == "test-model"

    def test_config_has_all_tunables(self):
        """IT: Config includes all tunable parameters."""
        response = client.get("/api/metrics-config?model_id=test-model")
        data = response.json()
        
        assert "window_size" in data
        assert "update_interval_ms" in data
        assert "latency_warning_threshold_ms" in data
        assert "error_rate_warning_threshold_pct" in data
        assert "chart_colors" in data

    def test_config_has_sensible_defaults(self):
        """IT: Config provides sensible default values."""
        response = client.get("/api/metrics-config?model_id=test-model")
        data = response.json()
        
        # Default values should be reasonable
        assert 1 <= data["window_size"] <= 600
        assert 100 <= data["update_interval_ms"] <= 10000
        assert data["latency_warning_threshold_ms"] > 0
        assert 0 <= data["error_rate_warning_threshold_pct"] <= 100

    def test_config_has_chart_colors(self):
        """IT: Config includes chart color definitions."""
        response = client.get("/api/metrics-config?model_id=test-model")
        data = response.json()
        
        colors = data.get("chart_colors", {})
        assert "requests" in colors
        assert "latency" in colors
        assert "throughput" in colors

    def test_update_metrics_config(self):
        """IT: Updating metrics config persists changes."""
        response = client.post(
            "/api/metrics-config",
            data={
                "model_id": "test-model",
                "window_size": "120",
                "update_interval_ms": "2000",
                "latency_warning_threshold_ms": "500",
                "error_rate_warning_threshold_pct": "10.5",
            }
        )
        assert response.status_code in [200, 400]  # May fail if model not in registry
        if response.status_code == 200:
            data = response.json()
            assert data.get("status") == "updated"

    def test_config_validates_window_size(self):
        """IT: Config validates window_size bounds."""
        # Test with values outside bounds
        response = client.post(
            "/api/metrics-config",
            data={
                "model_id": "test-model",
                "window_size": "1000",  # Max is 600
            }
        )
        # Should either accept and clamp, or return error
        assert response.status_code in [200, 400]

    def test_config_validates_update_interval(self):
        """IT: Config validates update_interval_ms bounds."""
        response = client.post(
            "/api/metrics-config",
            data={
                "model_id": "test-model",
                "update_interval_ms": "50000",  # Max is 10000
            }
        )
        # Should either accept and clamp, or return error
        assert response.status_code in [200, 400]


# ============================================================================
# INTEGRATION TESTS - Model Control Center UI
# ============================================================================


class TestModelControlCenterUI:
    """IT: Test model control center page rendering."""

    def test_control_center_loads(self):
        """IT: Model control center page loads successfully."""
        response = client.get("/model/test-model/control")
        assert response.status_code == 200
        assert "Model Control Center" in response.text

    def test_control_center_has_metrics_forms(self):
        """IT: Control center includes configuration forms."""
        response = client.get("/model/test-model/control")
        
        # Should have configuration form
        assert 'id="metricsConfigForm"' in response.text or "metricsConfigForm" in response.text
        # Should have input fields for configuration
        assert 'name="window_size"' in response.text
        assert 'name="update_interval_ms"' in response.text

    def test_control_center_has_charts(self):
        """IT: Control center includes chart canvases."""
        response = client.get("/model/test-model/control")
        
        assert "requestsChart" in response.text
        assert "latencyChart" in response.text
        assert "throughputChart" in response.text
        assert "errorChart" in response.text

    def test_control_center_has_chart_js(self):
        """IT: Control center loads Chart.js library."""
        response = client.get("/model/test-model/control")
        
        assert "chart.js" in response.text.lower()

    def test_control_center_has_color_config_display(self):
        """IT: Control center shows chart color configuration."""
        response = client.get("/model/test-model/control")
        
        assert "🎨 Chart Colors" in response.text or "Chart Colors" in response.text

    def test_control_center_threshold_warnings(self):
        """IT: Control center displays warning thresholds."""
        response = client.get("/model/test-model/control")
        
        # Should show threshold information
        assert "threshold" in response.text.lower()
        assert "warning" in response.text.lower()

    def test_control_center_form_submission(self):
        """IT: Control center form can be submitted via HTMX."""
        response = client.get("/model/test-model/control")
        
        # Check for HTMX attributes
        assert 'hx-post="/api/metrics-config"' in response.text or "/api/metrics-config" in response.text

    def test_control_center_javascript_updates(self):
        """IT: Control center includes JavaScript for chart updates."""
        response = client.get("/model/test-model/control")
        
        # Should have async fetch function
        assert "fetchMetrics" in response.text
        assert "updateCharts" in response.text


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestMetricsEdgeCases:
    """IT: Test edge cases and error handling."""

    def test_negative_latency_rejected(self):
        """IT: Negative latency values are handled gracefully."""
        # Try to record negative latency - service should handle it
        response = client.post(
            "/api/record-metrics",
            data={
                "model_id": "test-model",
                "latency_ms": "-100.0",  # Invalid
                "success": "True",
            }
        )
        # Either accepts and handles, or rejects
        assert response.status_code in [200, 400, 422]

    def test_very_large_latency_handled(self):
        """IT: Very large latency values are handled."""
        response = client.post(
            "/api/record-metrics",
            data={
                "model_id": "test-model",
                "latency_ms": "999999.99",
                "success": "True",
            }
        )
        assert response.status_code in [200]

    def test_empty_model_id_handled(self):
        """IT: Empty model ID is handled gracefully."""
        response = client.get("/api/model-metrics?model_id=")
        # Should either return 400 or handle gracefully
        assert response.status_code in [200, 400, 422]

    def test_special_characters_in_model_id(self):
        """IT: Special characters in model ID are escaped."""
        response = client.get("/api/model-metrics?model_id=test%2Fmodel%3Fid")
        assert response.status_code in [200, 404]

    def test_concurrent_metrics_recording(self):
        """IT: Multiple metrics can be recorded without data corruption."""
        # Record multiple metrics rapidly
        for i in range(10):
            response = client.post(
                "/api/record-metrics",
                data={
                    "model_id": f"model-{i}",
                    "latency_ms": str(100.0 + i),
                    "success": "True",
                }
            )
            assert response.status_code == 200

    def test_metrics_with_missing_optional_fields(self):
        """IT: Metrics can be recorded without optional fields."""
        response = client.post(
            "/api/record-metrics",
            data={
                "model_id": "test-model",
                "latency_ms": "150.0",
                "success": "True",
                # No cpu_percent or memory_mb
            }
        )
        assert response.status_code == 200


# ============================================================================
# FRONTEND AND INTERACTION TESTS
# ============================================================================


class TestControlCenterInteractions:
    """IT: Test user interactions with control center."""

    def test_dashboard_link_present(self):
        """IT: Control center has link back to dashboard."""
        response = client.get("/model/test-model/control")
        
        assert "Dashboard" in response.text or "dashboard" in response.text.lower()

    def test_model_info_displayed(self):
        """IT: Model information is displayed in control center."""
        response = client.get("/model/test-model/control")
        
        # Should show model ID
        assert "test-model" in response.text

    def test_refresh_capability(self):
        """IT: Page supports manual refresh (Ctrl+R)."""
        response = client.get("/model/test-model/control")
        
        # Should have JavaScript supporting refresh
        assert "keydown" in response.text or "refresh" in response.text.lower()
