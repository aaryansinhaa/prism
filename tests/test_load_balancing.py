"""Tests for load balancing and multiple container instances."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.registry.container_registry import (
    register_container_instance,
    get_model_instances,
)
from app.services.load_balancer import (
    LoadBalancer,
    LoadBalancerState,
    ContainerInstance,
    reset_load_balancer,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestLoadBalancer:
    """Test the load balancer core functionality."""

    def test_load_balancer_initialization(self):
        """Test load balancer initializes with empty state."""
        lb = LoadBalancer()
        assert lb.get_state("sentiment") is None

    def test_register_instance(self):
        """Test registering container instances."""
        lb = LoadBalancer()
        state = lb.register_instance(
            "sentiment", "v1", "container-1", 9001, instance_index=0
        )

        assert state.model_id == "sentiment"
        assert state.version == "v1"
        assert len(state.instances) == 1
        assert state.instances[0].container_id == "container-1"
        assert state.instances[0].port == 9001

    def test_round_robin_distribution(self):
        """Test round-robin selection across instances."""
        lb = LoadBalancer()

        # Register 3 instances
        for i in range(3):
            lb.register_instance(
                "sentiment",
                "v1",
                f"container-{i}",
                9001 + i,
                instance_index=i,
            )

        # Get 6 selections and verify round-robin order
        selected_ports = []
        for _ in range(6):
            instance = lb.get_next_instance("sentiment", "v1")
            selected_ports.append(instance.port)

        # Should cycle through ports in order
        expected = [9001, 9002, 9003, 9001, 9002, 9003]
        assert selected_ports == expected

    def test_instance_health_tracking(self):
        """Test marking instances as healthy/unhealthy."""
        lb = LoadBalancer()
        state = lb.register_instance(
            "sentiment", "v1", "container-1", 9001, instance_index=0
        )

        instance = state.instances[0]
        assert instance.healthy

        # Mark 3 failures
        for _ in range(3):
            instance.mark_failure()

        assert not instance.healthy
        assert instance.consecutive_failures == 3

        # Mark success
        instance.mark_success()
        assert instance.healthy
        assert instance.consecutive_failures == 0

    def test_health_summary(self):
        """Test getting health summary."""
        lb = LoadBalancer()
        lb.register_instance("sentiment", "v1", "container-1", 9001, instance_index=0)
        lb.register_instance("sentiment", "v1", "container-2", 9002, instance_index=1)

        health = lb.get_health_summary("sentiment", "v1")
        assert health["total_instances"] == 2
        assert health["healthy_instances"] == 2
        assert len(health["instances"]) == 2


class TestRegistryInstancesSupport:
    """Test registry support for multiple instances."""

    def test_register_container_instance(self, monkeypatch, tmp_path):
        """Test registering container instances in registry."""
        registry_file = tmp_path / "containers.json"
        monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

        # Register 3 instances
        for i in range(3):
            register_container_instance(
                model_id="sentiment",
                version="v1",
                container_id=f"container-{i}",
                port=9001 + i,
                instance_index=i,
            )

        data = json.loads(registry_file.read_text(encoding="utf-8"))
        sentiment = data["models"]["sentiment"]
        instances = sentiment["versions"]["v1"]["instances"]

        assert len(instances) == 3
        assert instances[0]["port"] == 9001
        assert instances[1]["port"] == 9002
        assert instances[2]["port"] == 9003

    def test_get_model_instances(self, monkeypatch, tmp_path):
        """Test retrieving instances for a model."""
        registry_file = tmp_path / "containers.json"
        monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

        # Register instances
        for i in range(2):
            register_container_instance(
                model_id="sentiment",
                version="v1",
                container_id=f"container-{i}",
                port=9001 + i,
                instance_index=i,
            )

        instances = get_model_instances("sentiment", "v1")
        assert len(instances) == 2
        assert instances[0]["port"] == 9001
        assert instances[1]["port"] == 9002

    def test_multiple_versions_separate_instances(self, monkeypatch, tmp_path):
        """Test that different versions have separate instances."""
        registry_file = tmp_path / "containers.json"
        monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

        # Register v1 instances
        register_container_instance(
            model_id="sentiment",
            version="v1",
            container_id="container-v1",
            port=9001,
            instance_index=0,
        )

        # Register v2 instances
        register_container_instance(
            model_id="sentiment",
            version="v2",
            container_id="container-v2",
            port=9002,
            instance_index=0,
        )

        v1_instances = get_model_instances("sentiment", "v1")
        v2_instances = get_model_instances("sentiment", "v2")

        assert len(v1_instances) == 1
        assert len(v2_instances) == 1
        assert v1_instances[0]["port"] == 9001
        assert v2_instances[0]["port"] == 9002


class TestLoadBalancedPrediction:
    """Test prediction routing through load balancer."""

    def test_single_instance_prediction(self, monkeypatch, tmp_path):
        """Test prediction with single instance."""
        registry_file = tmp_path / "containers.json"
        monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))
        monkeypatch.setenv("PRISM_API_KEYS", "test-key")

        registry_data = {
            "models": {
                "sentiment": {
                    "model_id": "sentiment",
                    "active_version": "v1",
                    "versions": {
                        "v1": {
                            "model_id": "sentiment",
                            "version": "v1",
                            "instances": [
                                {
                                    "container_id": "container-1",
                                    "port": 9001,
                                    "instance_index": 0,
                                }
                            ],
                        }
                    },
                }
            }
        }
        registry_file.write_text(json.dumps(registry_data), encoding="utf-8")

        calls: list[str] = []

        async def fake_forward(
            container_url: str, payload: dict[str, object]
        ) -> dict[str, object]:
            calls.append(container_url)
            return {"predictions": [42]}

        monkeypatch.setattr("app.routing.inference.request_batcher.forward", fake_forward)

        with TestClient(app) as client:
            response = client.post(
                "/models/sentiment/predict",
                json={"input": [1.0]},
                headers={"X-API-Key": "test-key"},
            )

        assert response.status_code == 200
        assert calls == ["http://127.0.0.1:9001/predict"]

    def test_multiple_instances_round_robin(self, monkeypatch, tmp_path):
        """Test prediction distributes across instances with round-robin."""
        registry_file = tmp_path / "containers.json"
        monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))
        monkeypatch.setenv("PRISM_API_KEYS", "test-key")

        registry_data = {
            "models": {
                "sentiment": {
                    "model_id": "sentiment",
                    "active_version": "v1",
                    "versions": {
                        "v1": {
                            "model_id": "sentiment",
                            "version": "v1",
                            "instances": [
                                {
                                    "container_id": "container-1",
                                    "port": 9001,
                                    "instance_index": 0,
                                },
                                {
                                    "container_id": "container-2",
                                    "port": 9002,
                                    "instance_index": 1,
                                },
                                {
                                    "container_id": "container-3",
                                    "port": 9003,
                                    "instance_index": 2,
                                },
                            ],
                        }
                    },
                }
            }
        }
        registry_file.write_text(json.dumps(registry_data), encoding="utf-8")

        calls: list[str] = []

        async def fake_forward(
            container_url: str, payload: dict[str, object]
        ) -> dict[str, object]:
            calls.append(container_url)
            return {"predictions": [42]}

        monkeypatch.setattr("app.routing.inference.request_batcher.forward", fake_forward)
        reset_load_balancer()

        with TestClient(app) as client:
            # Make 6 requests
            for _ in range(6):
                response = client.post(
                    "/models/sentiment/predict",
                    json={"input": [1.0]},
                    headers={"X-API-Key": "test-key"},
                )
                assert response.status_code == 200

        # Verify round-robin distribution
        expected = [
            "http://127.0.0.1:9001/predict",
            "http://127.0.0.1:9002/predict",
            "http://127.0.0.1:9003/predict",
            "http://127.0.0.1:9001/predict",
            "http://127.0.0.1:9002/predict",
            "http://127.0.0.1:9003/predict",
        ]
        assert calls == expected

    def test_instances_list_endpoint(self, monkeypatch, tmp_path):
        """Test GET /models/{model_id}/instances endpoint."""
        registry_file = tmp_path / "containers.json"
        monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

        registry_data = {
            "models": {
                "sentiment": {
                    "model_id": "sentiment",
                    "active_version": "v1",
                    "versions": {
                        "v1": {
                            "model_id": "sentiment",
                            "version": "v1",
                            "instances": [
                                {
                                    "container_id": "container-1",
                                    "port": 9001,
                                    "instance_index": 0,
                                },
                                {
                                    "container_id": "container-2",
                                    "port": 9002,
                                    "instance_index": 1,
                                },
                            ],
                        }
                    },
                }
            }
        }
        registry_file.write_text(json.dumps(registry_data), encoding="utf-8")

        with TestClient(app) as client:
            response = client.get("/models/sentiment/instances")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "sentiment"
        assert data["instance_count"] == 2
        assert len(data["instances"]) == 2
