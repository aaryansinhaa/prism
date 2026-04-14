from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


def test_post_models_creates_and_runs_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_UPLOAD_ROOT", str(tmp_path))
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

    def fake_run(command, **kwargs):
        if command[:3] == ["docker", "build", "-t"]:
            return subprocess.CompletedProcess(command, 0, stdout="built", stderr="")
        if command[:3] == ["docker", "run", "-d"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="container-lifecycle", stderr=""
            )
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr("subprocess.run", fake_run)

    with TestClient(app) as client:
        response = client.post(
            "/models",
            data={
                "model_name": "Lifecycle Model",
                "model_description": "Created through PRISM-11 endpoint",
                "expected_input_json": '{"input": [1.0, 2.0]}',
            },
            files={
                "file": (
                    "uploaded_model.onnx",
                    io.BytesIO(b"model-bytes"),
                    "application/octet-stream",
                )
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["container_id"] == "container-lifecycle"
    assert payload["registry"]["name"] == "Lifecycle Model"


def test_get_models_lists_registry_records(monkeypatch, tmp_path):
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

    registry_data = {
        "models": {
            "abc123": {
                "model_id": "abc123",
                "container_id": "container-abc",
                "port": 9001,
                "name": "Model ABC",
            }
        }
    }
    registry_file.write_text(json.dumps(registry_data), encoding="utf-8")

    with TestClient(app) as client:
        response = client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["models"][0]["model_id"] == "abc123"
    assert payload["models"][0]["container_id"] == "container-abc"


def test_get_model_by_id_returns_record(monkeypatch, tmp_path):
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

    registry_data = {
        "models": {
            "xyz789": {
                "model_id": "xyz789",
                "container_id": "container-xyz",
                "port": 9002,
                "description": "single-model lookup",
            }
        }
    }
    registry_file.write_text(json.dumps(registry_data), encoding="utf-8")

    with TestClient(app) as client:
        response = client.get("/models/xyz789")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == "xyz789"
    assert payload["container_id"] == "container-xyz"
    assert payload["port"] == 9002


def test_get_model_by_id_returns_404_when_missing(monkeypatch, tmp_path):
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))
    registry_file.write_text(json.dumps({"models": {}}), encoding="utf-8")

    with TestClient(app) as client:
        response = client.get("/models/not-there")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_delete_model_by_id_stops_container_and_removes_registry(monkeypatch, tmp_path):
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

    registry_data = {
        "models": {
            "delete-me": {
                "model_id": "delete-me",
                "container_id": "container-delete",
                "port": 9003,
            }
        }
    }
    registry_file.write_text(json.dumps(registry_data), encoding="utf-8")

    def fake_run(command, **kwargs):
        if command[:3] == ["docker", "stop", "container-delete"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["docker", "rm", "container-delete"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr("subprocess.run", fake_run)

    with TestClient(app) as client:
        response = client.delete("/models/delete-me")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["success"] is True
    assert payload["model_id"] == "delete-me"

    with registry_file.open("r", encoding="utf-8") as file:
        updated_registry = json.load(file)
    assert "delete-me" not in updated_registry.get("models", {})


def test_delete_model_by_id_returns_404_when_missing(monkeypatch, tmp_path):
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))
    registry_file.write_text(json.dumps({"models": {}}), encoding="utf-8")

    with TestClient(app) as client:
        response = client.delete("/models/not-there")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
