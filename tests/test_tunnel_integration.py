"""Tests for reverse tunnel integration."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


def test_upload_and_run_without_tunnel(monkeypatch, tmp_path):
    """Test that upload-and-run works without tunnel enabled (default behavior)."""
    monkeypatch.setenv("MODEL_UPLOAD_ROOT", str(tmp_path))
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))
    monkeypatch.delenv("ENABLE_TUNNEL", raising=False)  # Ensure tunnel is disabled

    call_count = {"build": 0, "run": 0}

    def fake_run(command, **kwargs):
        import subprocess

        if command[:3] == ["docker", "build", "-t"]:
            call_count["build"] += 1
            cwd = kwargs.get("cwd")
            assert cwd is not None
            assert Path(cwd).exists()
            return subprocess.CompletedProcess(command, 0, stdout="built", stderr="")
        if command[:3] == ["docker", "run", "-d"]:
            call_count["run"] += 1
            assert "--name" in command
            assert "-p" in command
            return subprocess.CompletedProcess(
                command, 0, stdout="container123", stderr=""
            )
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr("subprocess.run", fake_run)

    with TestClient(app) as client:
        response = client.post(
            "/models/upload-and-run",
            files={
                "file": (
                    "model.onnx",
                    io.BytesIO(b"model-bytes"),
                    "application/octet-stream",
                )
            },
        )

    assert response.status_code == 200
    payload = response.json()

    # Verify tunnel field is present but None
    assert "tunnel_url" in payload
    assert payload["tunnel_url"] is None

    # Verify registry doesn't have tunnel info
    with registry_file.open("r") as f:
        registry = json.load(f)
    model_id = payload["model_id"]
    assert "tunnel_url" not in registry["models"][model_id]


def test_upload_and_run_with_tunnel_enabled(monkeypatch, tmp_path):
    """Test that upload-and-run starts tunnel when ENABLE_TUNNEL=true."""
    monkeypatch.setenv("MODEL_UPLOAD_ROOT", str(tmp_path))
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))
    monkeypatch.setenv("ENABLE_TUNNEL", "true")

    call_count = {"build": 0, "run": 0, "tunnel": 0}

    def fake_run(command, **kwargs):
        import subprocess

        if command[:3] == ["docker", "build", "-t"]:
            call_count["build"] += 1
            cwd = kwargs.get("cwd")
            assert cwd is not None
            assert Path(cwd).exists()
            return subprocess.CompletedProcess(command, 0, stdout="built", stderr="")
        if command[:3] == ["docker", "run", "-d"]:
            call_count["run"] += 1
            assert "--name" in command
            assert "-p" in command
            return subprocess.CompletedProcess(
                command, 0, stdout="container123", stderr=""
            )
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr("subprocess.run", fake_run)

    # Mock the tunnel startup
    async def fake_start_tunnel(local_port, model_id):
        call_count["tunnel"] += 1
        return f"https://testmodel{model_id[:6]}.ngrok.io", None


    monkeypatch.setattr("app.routing.models.start_tunnel", fake_start_tunnel)

    with TestClient(app) as client:
        response = client.post(
            "/models/upload-and-run",
            files={
                "file": (
                    "model.onnx",
                    io.BytesIO(b"model-bytes"),
                    "application/octet-stream",
                )
            },
        )

    assert response.status_code == 200
    payload = response.json()
    model_id = payload["model_id"]

    # Verify tunnel was started
    assert call_count["tunnel"] == 1

    # Verify response includes tunnel info
    assert payload["tunnel_url"] is not None
    assert payload["tunnel_url"].endswith(".ngrok.io")

    # Verify registry stores tunnel info
    with registry_file.open("r") as f:
        registry = json.load(f)
    assert "tunnel_url" in registry["models"][model_id]
    assert registry["models"][model_id]["tunnel_url"].endswith(".ngrok.io")


def test_upload_and_run_tunnel_startup_failure_is_non_fatal(monkeypatch, tmp_path):
    """Test that tunnel startup failure doesn't block model deployment."""
    monkeypatch.setenv("MODEL_UPLOAD_ROOT", str(tmp_path))
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))
    monkeypatch.setenv("ENABLE_TUNNEL", "true")

    def fake_run(command, **kwargs):
        import subprocess

        if command[:3] == ["docker", "build", "-t"]:
            cwd = kwargs.get("cwd")
            assert cwd is not None
            return subprocess.CompletedProcess(command, 0, stdout="built", stderr="")
        if command[:3] == ["docker", "run", "-d"]:
            assert "--name" in command
            return subprocess.CompletedProcess(
                command, 0, stdout="container123", stderr=""
            )
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr("subprocess.run", fake_run)

    # Mock tunnel to fail
    async def fake_start_tunnel_fail(local_port, model_id):
        raise RuntimeError("ngrok connection failed")

    monkeypatch.setattr("app.routing.models.start_tunnel", fake_start_tunnel_fail)

    with TestClient(app) as client:
        response = client.post(
            "/models/upload-and-run",
            files={
                "file": (
                    "model.onnx",
                    io.BytesIO(b"model-bytes"),
                    "application/octet-stream",
                )
            },
        )

    # Container should still be deployed even if tunnel fails
    assert response.status_code == 200
    payload = response.json()

    # Verify container info is returned
    assert "container_id" in payload
    assert "host_port" in payload

    # Verify tunnel field is None (since it failed)
    assert payload["tunnel_url"] is None


def test_registry_contains_tunnel_info(monkeypatch, tmp_path):
    """Test that registry correctly stores and retrieves tunnel metadata."""
    from app.registry.container_registry import register_container

    registry_file = tmp_path / "registry.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

    # Register container with tunnel info
    record = register_container(
        model_id="test_model_001",
        container_id="container_abc123",
        port=9000,
        tunnel_url="https://testmodel001.ngrok.io",
    )

    assert record["tunnel_url"] == "https://testmodel001.ngrok.io"

    # Verify persisted in file
    with registry_file.open("r") as f:
        data = json.load(f)

    assert (
        data["models"]["test_model_001"]["tunnel_url"]
        == "https://testmodel001.ngrok.io"
    )


def test_registry_without_tunnel_info(monkeypatch, tmp_path):
    """Test that registry works fine without tunnel info (backward compatibility)."""
    from app.registry.container_registry import register_container

    registry_file = tmp_path / "registry.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

    # Register container without tunnel info
    record = register_container(
        model_id="test_model_002",
        container_id="container_def456",
        port=9001,
    )

    assert "tunnel_url" not in record

    # Verify it's not in the file either
    with registry_file.open("r") as f:
        data = json.load(f)

    assert "tunnel_url" not in data["models"]["test_model_002"]
