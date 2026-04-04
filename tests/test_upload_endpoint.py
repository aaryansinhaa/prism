from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


def test_upload_builds_model_image(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_UPLOAD_ROOT", str(tmp_path))

    def fake_run(command, cwd, check, capture_output, text, timeout):
        assert command[:3] == ["docker", "build", "-t"]
        assert command[3].startswith("prism_model_")
        assert command[4] == "."
        assert Path(cwd).exists()
        return subprocess.CompletedProcess(command, 0, stdout="built", stderr="")

    monkeypatch.setattr("app.main.subprocess.run", fake_run)

    with TestClient(app) as client:
        response = client.post(
            "/upload",
            files={"file": ("uploaded_model.onnx", io.BytesIO(b"model-bytes"), "application/octet-stream")},
        )

    assert response.status_code == 200, response.text
    payload = response.json()

    model_id = payload["model_id"]
    model_dir = tmp_path / model_id

    assert payload["image_tag"] == f"prism_model_{model_id}"
    assert (model_dir / "uploaded_model.onnx").exists()
    assert (model_dir / "Dockerfile").exists()
    assert (model_dir / "runtime.py").exists()
    assert (model_dir / "requirements.txt").exists()
    assert (model_dir / "entrypoint.sh").exists()


def test_upload_rejects_unsupported_extension(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_UPLOAD_ROOT", str(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/upload",
            files={"file": ("bad_model.txt", io.BytesIO(b"not-a-model"), "text/plain")},
        )

    assert response.status_code == 400
    assert "Unsupported model format" in response.json()["detail"]


def test_upload_and_run_starts_container(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_UPLOAD_ROOT", str(tmp_path))

    def fake_run(command, **kwargs):
        if command[:3] == ["docker", "build", "-t"]:
            cwd = kwargs.get("cwd")
            assert cwd is not None
            assert Path(cwd).exists()
            return subprocess.CompletedProcess(command, 0, stdout="built", stderr="")
        if command[:3] == ["docker", "run", "-d"]:
            assert "--name" in command
            assert "-p" in command
            return subprocess.CompletedProcess(command, 0, stdout="container123", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr("app.main.subprocess.run", fake_run)

    with TestClient(app) as client:
        response = client.post(
            "/upload-and-run",
            files={"file": ("uploaded_model.onnx", io.BytesIO(b"model-bytes"), "application/octet-stream")},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["container_name"].startswith("prism_model_")
    assert payload["container_id"] == "container123"
    assert isinstance(payload["host_port"], int)
    assert payload["predict_url"].startswith("http://127.0.0.1:")


def test_upload_and_run_returns_500_on_run_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_UPLOAD_ROOT", str(tmp_path))

    def fake_run(command, **kwargs):
        if command[:3] == ["docker", "build", "-t"]:
            return subprocess.CompletedProcess(command, 0, stdout="built", stderr="")
        if command[:3] == ["docker", "run", "-d"]:
            raise subprocess.CalledProcessError(1, command, output="", stderr="docker run boom")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr("app.main.subprocess.run", fake_run)

    with TestClient(app) as client:
        response = client.post(
            "/upload-and-run",
            files={"file": ("uploaded_model.onnx", io.BytesIO(b"model-bytes"), "application/octet-stream")},
        )

    assert response.status_code == 500
    assert "docker run boom" in response.json()["detail"]
