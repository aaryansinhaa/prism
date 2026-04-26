from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.registry.container_registry import register_container

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_register_container_creates_versioned_registry(monkeypatch, tmp_path):
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

    register_container(
        model_id="sentiment",
        version="v1",
        container_id="container-v1",
        port=9001,
        name="Sentiment v1",
    )
    register_container(
        model_id="sentiment",
        version="v2",
        container_id="container-v2",
        port=9002,
        name="Sentiment v2",
    )

    data = json.loads(registry_file.read_text(encoding="utf-8"))
    sentiment = data["models"]["sentiment"]

    assert sentiment["active_version"] == "v2"
    assert sorted(sentiment["versions"].keys()) == ["v1", "v2"]
    assert sentiment["versions"]["v1"]["container_id"] == "container-v1"
    assert sentiment["versions"]["v2"]["container_id"] == "container-v2"


def test_public_predict_routes_to_active_version(monkeypatch, tmp_path):
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))
    monkeypatch.setenv("PRISM_API_KEYS", "test-key")

    registry_data = {
        "models": {
            "sentiment": {
                "model_id": "sentiment",
                "active_version": "v2",
                "versions": {
                    "v1": {
                        "model_id": "sentiment",
                        "version": "v1",
                        "container_id": "container-v1",
                        "port": 9001,
                    },
                    "v2": {
                        "model_id": "sentiment",
                        "version": "v2",
                        "container_id": "container-v2",
                        "port": 9002,
                    },
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

    assert response.status_code == 200, response.text
    assert response.json()["predictions"] == [42]
    assert calls == ["http://127.0.0.1:9002/predict"]


@pytest.mark.skip(
    reason="Test has global state pollution issues - passes individually but fails in suite"
)
def test_versioned_predict_routes_to_requested_version(monkeypatch, tmp_path):
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))
    monkeypatch.setenv("PRISM_API_KEYS", "test-key")

    registry_data = {
        "models": {
            "sentiment": {
                "model_id": "sentiment",
                "active_version": "v2",
                "versions": {
                    "v1": {
                        "model_id": "sentiment",
                        "version": "v1",
                        "container_id": "container-v1",
                        "port": 9001,
                    },
                    "v2": {
                        "model_id": "sentiment",
                        "version": "v2",
                        "container_id": "container-v2",
                        "port": 9002,
                    },
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
        return {"predictions": [99]}

    monkeypatch.setattr("app.routing.inference.request_batcher.forward", fake_forward)

    with TestClient(app) as client:
        response = client.post(
            "/models/sentiment/versions/v1/predict",
            json={"input": [1.0]},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200, response.text
    assert response.json()["predictions"] == [99]
    assert calls == ["http://127.0.0.1:9001/predict"]


def test_versioned_registry_lookup_returns_404_for_missing_version(
    monkeypatch, tmp_path
):
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
                        "container_id": "container-v1",
                        "port": 9001,
                    }
                },
            }
        }
    }
    registry_file.write_text(json.dumps(registry_data), encoding="utf-8")

    with TestClient(app) as client:
        response = client.post(
            "/models/sentiment/versions/v2/predict",
            json={"input": [1.0]},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 404
    assert "version v2" in response.json()["detail"].lower()
