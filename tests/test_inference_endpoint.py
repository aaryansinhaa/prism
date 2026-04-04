from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


def test_inference_returns_404_for_missing_model(tmp_path):
    """Test that inference endpoint returns 404 for unknown model."""
    with TestClient(app) as client:
        response = client.post(
            "/models/nonexistent_model/predict",
            json={"x": [1.0, 2.0, 3.0]},
        )

    assert response.status_code == 404
    assert "not found in registry" in response.json()["detail"].lower()


def test_inference_forwards_to_container(monkeypatch, tmp_path):
    """Test that inference endpoint forwards request to container and returns response."""
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

    # Create a registry entry
    model_id = "test_model_123"
    registry_data = {
        "models": {
            model_id: {
                "model_id": model_id,
                "container_id": "fake_container_123",
                "port": 9999,
            }
        }
    }
    registry_file.write_text(json.dumps(registry_data))

    # Mock httpx.AsyncClient
    class FakeResponse:
        def __init__(self, status_code, json_data):
            self.status_code = status_code
            self._json_data = json_data

        async def json(self):
            return self._json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"Status {self.status_code}")

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def post(self, url, **kwargs):
            # Return mock prediction response
            return FakeResponse(200, {"predictions": [0.5, 1.5, 2.5]})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    import httpx

    monkeypatch.setattr("app.routing.inference.httpx.AsyncClient", FakeAsyncClient)

    with TestClient(app) as client:
        response = client.post(
            f"/models/{model_id}/predict",
            json={"x": [1.0, 2.0, 3.0]},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predictions"] == [0.5, 1.5, 2.5]


def test_inference_handles_connection_error(monkeypatch, tmp_path):
    """Test that inference endpoint handles connection errors gracefully."""
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

    # Create a registry entry
    model_id = "test_model_456"
    registry_data = {
        "models": {
            model_id: {
                "model_id": model_id,
                "container_id": "fake_container_456",
                "port": 9998,
            }
        }
    }
    registry_file.write_text(json.dumps(registry_data))

    # Mock httpx.AsyncClient to raise ConnectError
    import httpx

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def post(self, url, **kwargs):
            raise httpx.ConnectError("Cannot connect")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr("app.routing.inference.httpx.AsyncClient", FakeAsyncClient)

    with TestClient(app) as client:
        response = client.post(
            f"/models/{model_id}/predict",
            json={"x": [1.0, 2.0, 3.0]},
        )

    assert response.status_code == 503
    assert "cannot connect" in response.json()["detail"].lower()
