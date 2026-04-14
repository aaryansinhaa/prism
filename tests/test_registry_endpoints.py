from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from app.main import app

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_prune_stale_registry_entries_removes_missing_containers(monkeypatch, tmp_path):
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

    registry_data = {
        "models": {
            "alive-model": {
                "model_id": "alive-model",
                "container_id": "container-alive",
                "port": 9001,
            },
            "stale-model": {
                "model_id": "stale-model",
                "container_id": "container-stale",
                "port": 9002,
            },
        }
    }
    registry_file.write_text(json.dumps(registry_data), encoding="utf-8")

    def fake_container_exists(container_id: str) -> bool:
        return container_id == "container-alive"

    monkeypatch.setattr(
        "app.services.dashboard_service.container_exists",
        fake_container_exists,
    )

    with TestClient(app) as client:
        response = client.post("/registry/prune-stale")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["removed_count"] == 1
    assert payload["removed_model_ids"] == ["stale-model"]

    updated = json.loads(registry_file.read_text(encoding="utf-8"))
    assert "alive-model" in updated["models"]
    assert "stale-model" not in updated["models"]


def test_prune_stale_registry_entries_noop_when_all_exist(monkeypatch, tmp_path):
    registry_file = tmp_path / "containers.json"
    monkeypatch.setenv("MODEL_CONTAINER_REGISTRY_PATH", str(registry_file))

    registry_data = {
        "models": {
            "model-a": {
                "model_id": "model-a",
                "container_id": "container-a",
                "port": 9001,
            }
        }
    }
    registry_file.write_text(json.dumps(registry_data), encoding="utf-8")

    monkeypatch.setattr(
        "app.services.dashboard_service.container_exists",
        lambda container_id: True,
    )

    with TestClient(app) as client:
        response = client.post("/registry/prune-stale")

    assert response.status_code == 200
    payload = response.json()
    assert payload["removed_count"] == 0
    assert payload["removed_model_ids"] == []

    updated = json.loads(registry_file.read_text(encoding="utf-8"))
    assert "model-a" in updated["models"]
