"""Health and status endpoints."""

from typing import Any, Dict

from fastapi import APIRouter

from app.services.health_monitor_service import HealthMonitorService

router = APIRouter()


@router.get("/")
def read_root() -> Dict[str, str]:
    return {"service": "prism-runtime"}


@router.get("/health/monitor")
def monitor_health() -> Dict[str, Any]:
    """Return health monitor runtime status and latest cycle information."""
    return HealthMonitorService.get_status_snapshot()
