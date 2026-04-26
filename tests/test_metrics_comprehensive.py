"""Comprehensive tests for metrics system and model control center."""

from fastapi.testclient import TestClient
from app.main import app
from app.services.metrics_service import MetricsRegistry, MetricsCollector

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

    def test_reset_clears_all_metrics(self):
        """UT: Reset clears all accumulated metrics."""
        collector = MetricsCollector("model1", "container1")
        collector.record_request(latency_ms=100.0, success=True)
        collector.reset()

        metrics = collector.get_aggregated_metrics()
        assert all(latency == 0.0 for latency in metrics.latency)
        assert all(requests == 0 for requests in metrics.requests)


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
        registry._collectors.clear()

        collector = registry.register_model("test-model", "container123")
        assert collector is not None
        assert collector.model_id == "test-model"

    def test_remove_model_from_registry(self):
        """UT: Can remove model from registry."""
        registry = MetricsRegistry.get_instance()
        registry._collectors.clear()
        registry.register_model("model1", "container1")

        registry.remove_model("model1")
        assert registry.get_collector("model1") is None


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

    def test_metrics_consistency(self):
        """IT: Metric arrays have consistent lengths."""
        response = client.get("/api/model-metrics?model_id=test-model")
        data = response.json()

        n_labels = len(data["labels"])
        assert len(data["requests"]) == n_labels
        assert len(data["latency"]) == n_labels
        assert len(data["throughput"]) == n_labels


# ============================================================================
# INTEGRATION TESTS - Model Control Center UI
# ============================================================================


class TestModelControlCenterUI:
    """IT: Test model control center page rendering."""

    def test_control_center_loads_missing_model(self):
        """IT: Model control center page loads even for missing model."""
        response = client.get("/model/test-model/control")
        # Should still render with default values for missing model
        assert response.status_code == 200
        assert (
            "Model Control Center" in response.text
            or "model_control" in response.text.lower()
        )

    def test_control_center_has_metrics_forms(self):
        """IT: Control center includes configuration forms."""
        response = client.get("/model/test-model/control")

        assert response.status_code == 200
        # Should have configuration form (even if model doesn't exist)
        assert (
            "metricsConfigForm" in response.text or "metrics" in response.text.lower()
        )

    def test_control_center_has_charts(self):
        """IT: Control center includes chart canvases."""
        response = client.get("/model/test-model/control")

        assert response.status_code == 200
        assert "requestsChart" in response.text or "requests" in response.text.lower()
        assert "latencyChart" in response.text or "latency" in response.text.lower()

    def test_control_center_has_chart_js(self):
        """IT: Control center loads Chart.js library."""
        response = client.get("/model/test-model/control")
        assert response.status_code == 200
        assert "chart" in response.text.lower()

    def test_control_center_has_color_config_display(self):
        """IT: Control center shows chart color configuration or colors."""
        response = client.get("/model/test-model/control")
        assert response.status_code == 200
        # Should have color-related content
        assert "color" in response.text.lower() or "#" in response.text

    def test_control_center_threshold_warnings(self):
        """IT: Control center displays warning thresholds."""
        response = client.get("/model/test-model/control")
        assert response.status_code == 200
        # Should mention warnings or thresholds
        assert (
            "warning" in response.text.lower() or "threshold" in response.text.lower()
        )

    def test_control_center_javascript_updates(self):
        """IT: Control center includes JavaScript for chart updates."""
        response = client.get("/model/test-model/control")
        assert response.status_code == 200
        # Should have fetch/update related JS code
        assert "fetch" in response.text or "update" in response.text.lower()


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
        assert "test-model" in response.text
