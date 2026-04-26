from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient
from app.main import app

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_health_monitor_endpoint_returns_snapshot(monkeypatch):
    expected_snapshot = {
        "running": True,
        "interval_seconds": 10,
        "last_cycle": {
            "scanned": 2,
            "restarted": 1,
            "completed_at": "2026-04-05T10:00:00+00:00",
        },
        "last_error": None,
    }

    monkeypatch.setattr(
        "app.routing.health.HealthMonitorService.get_status_snapshot",
        lambda: expected_snapshot,
    )

    with TestClient(app) as client:
        response = client.get("/health/monitor")

    assert response.status_code == 200
    assert response.json() == expected_snapshot
