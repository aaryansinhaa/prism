from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from app.main import app

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _auth_headers(api_key: str = "test-api-key") -> dict[str, str]:
    return {"X-API-Key": api_key}


def test_inference_returns_404_for_missing_model(monkeypatch, tmp_path):
    """Test that inference endpoint returns 404 for unknown model."""
    monkeypatch.setenv("PRISM_API_KEYS", "test-api-key")
    with TestClient(app) as client:
        response = client.post(
            "/models/nonexistent_model/predict",
            json={"x": [1.0, 2.0, 3.0]},
            headers=_auth_headers(),
        )

    assert response.status_code == 404
    assert "not found in registry" in response.json()["detail"].lower()


def test_inference_forwards_to_container(monkeypatch, tmp_path):
    """Test that inference endpoint forwards request to container and returns response."""
    monkeypatch.setenv("PRISM_API_KEYS", "test-api-key")
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

        def json(self):
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

    monkeypatch.setattr("app.routing.inference.httpx.AsyncClient", FakeAsyncClient)

    with TestClient(app) as client:
        response = client.post(
            f"/models/{model_id}/predict",
            json={"x": [1.0, 2.0, 3.0]},
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predictions"] == [0.5, 1.5, 2.5]


def test_inference_handles_connection_error(monkeypatch, tmp_path):
    """Test that inference endpoint handles connection errors gracefully."""
    monkeypatch.setenv("PRISM_API_KEYS", "test-api-key")
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
            headers=_auth_headers(),
        )

    assert response.status_code == 503
    assert "cannot connect" in response.json()["detail"].lower()


def test_inference_requires_api_key(monkeypatch, tmp_path):
    """Test that public inference endpoint requires an API key when configured."""
    monkeypatch.setenv("PRISM_API_KEYS", "required-key")
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))
    registry_file.write_text(json.dumps({"models": {}}))

    with TestClient(app) as client:
        response = client.post(
            "/models/any-model/predict",
            json={"x": [1.0]},
        )

    assert response.status_code == 401
    assert "api key" in response.json()["detail"].lower()


def test_inference_rate_limit(monkeypatch, tmp_path):
    """Test that public inference endpoint enforces configured rate limit."""
    model_id = "rate_limit_model"
    api_key = "rate-limit-key"

    monkeypatch.setenv("PRISM_API_KEYS", api_key)
    monkeypatch.setenv("PRISM_RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("PRISM_RATE_LIMIT_WINDOW_SECONDS", "60")

    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))
    registry_data = {
        "models": {
            model_id: {
                "model_id": model_id,
                "container_id": "fake_container_rate",
                "port": 9997,
            }
        }
    }
    registry_file.write_text(json.dumps(registry_data))

    class FakeResponse:
        def __init__(self, status_code, json_data):
            self.status_code = status_code
            self._json_data = json_data

        def json(self):
            return self._json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"Status {self.status_code}")

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def post(self, url, **kwargs):
            return FakeResponse(200, {"predictions": [1]})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr("app.routing.inference.httpx.AsyncClient", FakeAsyncClient)

    with TestClient(app) as client:
        first = client.post(
            f"/models/{model_id}/predict",
            json={"x": [1.0]},
            headers=_auth_headers(api_key),
        )
        second = client.post(
            f"/models/{model_id}/predict",
            json={"x": [1.0]},
            headers=_auth_headers(api_key),
        )
        third = client.post(
            f"/models/{model_id}/predict",
            json={"x": [1.0]},
            headers=_auth_headers(api_key),
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert "rate limit" in third.json()["detail"].lower()


def test_inference_rejects_payload_not_matching_expected_contract(
    monkeypatch, tmp_path
):
    """Test contract validation rejects payloads that violate expected input schema."""
    model_id = "contract_model"
    api_key = "contract-key"

    monkeypatch.setenv("PRISM_API_KEYS", api_key)
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

    contract = {
        "type": "object",
        "required": ["input"],
        "properties": {
            "input": {
                "type": "array",
                "items": {"type": "number"},
            }
        },
        "additionalProperties": False,
    }

    registry_data = {
        "models": {
            model_id: {
                "model_id": model_id,
                "container_id": "fake_container_contract",
                "port": 9996,
                "expected_input_json": json.dumps(contract),
            }
        }
    }
    registry_file.write_text(json.dumps(registry_data))

    class FakeResponse:
        def __init__(self, status_code, json_data):
            self.status_code = status_code
            self._json_data = json_data

        def json(self):
            return self._json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"Status {self.status_code}")

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def post(self, url, **kwargs):
            return FakeResponse(200, {"predictions": [1.0]})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr("app.routing.inference.httpx.AsyncClient", FakeAsyncClient)

    with TestClient(app) as client:
        response = client.post(
            f"/models/{model_id}/predict",
            json={"wrong": [1.0]},
            headers=_auth_headers(api_key),
        )

    assert response.status_code == 400
    assert "expected format" in response.json()["detail"].lower()
