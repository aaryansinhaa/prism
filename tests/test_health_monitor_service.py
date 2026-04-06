from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.model import ModelMetadata
from app.services.health_monitor_service import HealthMonitorService


def test_monitor_cycle_restarts_dead_container(monkeypatch):
    models = {
        "model-a": ModelMetadata(
            model_id="model-a",
            container_id="container-dead",
            port=9000,
        )
    }

    class DeadStatus:
        is_running = False

    monkeypatch.setattr(
        "app.services.health_monitor_service.ModelRegistryService.load_all_models",
        lambda: models,
    )
    monkeypatch.setattr(
        "app.services.health_monitor_service.ModelRegistryService.prune_stale_models",
        lambda: [],
    )
    monkeypatch.setattr(
        "app.services.health_monitor_service.get_container_status",
        lambda container_id: DeadStatus(),
    )

    restarted_ids: list[str] = []

    def fake_restart(container_id: str):
        restarted_ids.append(container_id)
        return True, "ok"

    monkeypatch.setattr(
        "app.services.health_monitor_service.restart_container",
        fake_restart,
    )

    result = asyncio.run(HealthMonitorService.run_monitor_cycle())

    assert result.scanned == 1
    assert result.restarted == 1
    assert restarted_ids == ["container-dead"]


def test_monitor_cycle_skips_running_container(monkeypatch):
    models = {
        "model-a": ModelMetadata(
            model_id="model-a",
            container_id="container-running",
            port=9001,
        )
    }

    class RunningStatus:
        is_running = True

    monkeypatch.setattr(
        "app.services.health_monitor_service.ModelRegistryService.load_all_models",
        lambda: models,
    )
    monkeypatch.setattr(
        "app.services.health_monitor_service.ModelRegistryService.prune_stale_models",
        lambda: [],
    )
    monkeypatch.setattr(
        "app.services.health_monitor_service.get_container_status",
        lambda container_id: RunningStatus(),
    )

    restart_calls = {"count": 0}

    def fake_restart(container_id: str):
        restart_calls["count"] += 1
        return True, "ok"

    monkeypatch.setattr(
        "app.services.health_monitor_service.restart_container",
        fake_restart,
    )

    result = asyncio.run(HealthMonitorService.run_monitor_cycle())

    assert result.scanned == 1
    assert result.restarted == 0
    assert restart_calls["count"] == 0


def test_monitor_cycle_handles_restart_failure(monkeypatch):
    models = {
        "model-a": ModelMetadata(
            model_id="model-a",
            container_id="container-failing",
            port=9002,
        )
    }

    class DeadStatus:
        is_running = False

    monkeypatch.setattr(
        "app.services.health_monitor_service.ModelRegistryService.load_all_models",
        lambda: models,
    )
    monkeypatch.setattr(
        "app.services.health_monitor_service.ModelRegistryService.prune_stale_models",
        lambda: [],
    )
    monkeypatch.setattr(
        "app.services.health_monitor_service.get_container_status",
        lambda container_id: DeadStatus(),
    )
    monkeypatch.setattr(
        "app.services.health_monitor_service.restart_container",
        lambda container_id: (False, "failed"),
    )

    result = asyncio.run(HealthMonitorService.run_monitor_cycle())

    assert result.scanned == 1
    assert result.restarted == 0


def test_monitor_cycle_prunes_stale_entries_before_scan(monkeypatch):
    calls = {"count": 0}

    def fake_prune():
        return ["stale-model"]

    def fake_load_models():
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "alive-model": ModelMetadata(
                    model_id="alive-model",
                    container_id="container-alive",
                    port=9003,
                )
            }
        return {
            "alive-model": ModelMetadata(
                model_id="alive-model",
                container_id="container-alive",
                port=9003,
            )
        }

    class RunningStatus:
        is_running = True

    monkeypatch.setattr(
        "app.services.health_monitor_service.ModelRegistryService.prune_stale_models",
        fake_prune,
    )
    monkeypatch.setattr(
        "app.services.health_monitor_service.ModelRegistryService.load_all_models",
        fake_load_models,
    )
    monkeypatch.setattr(
        "app.services.health_monitor_service.get_container_status",
        lambda container_id: RunningStatus(),
    )

    result = asyncio.run(HealthMonitorService.run_monitor_cycle())

    assert result.scanned == 1
    assert result.restarted == 0
