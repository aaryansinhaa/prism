"""Background health monitor for deployed model containers."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from app.services.dashboard_service import ModelRegistryService
from app.utils.docker_utils import get_container_status, restart_container


def _monitor_interval_seconds() -> int:
    raw = os.environ.get("PRISM_HEALTH_MONITOR_INTERVAL_SECONDS", "10")
    try:
        parsed = int(raw)
        return max(1, parsed)
    except ValueError:
        return 10


@dataclass(slots=True)
class MonitorCycleResult:
    scanned: int
    restarted: int


class HealthMonitorService:
    """Runs periodic container health checks and restarts dead containers."""

    _last_cycle_result: MonitorCycleResult = MonitorCycleResult(scanned=0, restarted=0)
    _last_cycle_at: str | None = None
    _last_error: str | None = None
    _running: bool = False

    @staticmethod
    def get_status_snapshot() -> Dict[str, Any]:
        return {
            "running": HealthMonitorService._running,
            "interval_seconds": _monitor_interval_seconds(),
            "last_cycle": {
                "scanned": HealthMonitorService._last_cycle_result.scanned,
                "restarted": HealthMonitorService._last_cycle_result.restarted,
                "completed_at": HealthMonitorService._last_cycle_at,
            },
            "last_error": HealthMonitorService._last_error,
        }

    @staticmethod
    def _record_cycle_result(result: MonitorCycleResult) -> None:
        HealthMonitorService._last_cycle_result = result
        HealthMonitorService._last_cycle_at = datetime.now(timezone.utc).isoformat()
        HealthMonitorService._last_error = None

    @staticmethod
    async def run_monitor_cycle() -> MonitorCycleResult:
        removed_model_ids = await asyncio.to_thread(
            ModelRegistryService.prune_stale_models
        )
        if removed_model_ids:
            print(
                "Health monitor: pruned stale registry entries: "
                + ", ".join(removed_model_ids)
            )

        models = ModelRegistryService.load_all_models()
        restarted = 0

        for metadata in models.values():
            status = await asyncio.to_thread(
                get_container_status, metadata.container_id
            )
            if status.is_running:
                continue

            success, message = await asyncio.to_thread(
                restart_container, metadata.container_id
            )
            if success:
                restarted += 1
                print(
                    f"Health monitor: restarted container {metadata.container_id[:12]}"
                )
            else:
                print(
                    "Health monitor: failed to restart "
                    f"{metadata.container_id[:12]}: {message}"
                )

        return MonitorCycleResult(scanned=len(models), restarted=restarted)

    @staticmethod
    async def run_forever(stop_event: asyncio.Event) -> None:
        HealthMonitorService._running = True
        while not stop_event.is_set():
            try:
                result = await HealthMonitorService.run_monitor_cycle()
                HealthMonitorService._record_cycle_result(result)
                if result.scanned > 0:
                    print(
                        "Health monitor cycle complete: "
                        f"scanned={result.scanned}, restarted={result.restarted}"
                    )
            except Exception as exc:
                HealthMonitorService._last_error = str(exc)
                print(f"Health monitor error: {exc}")

            interval = _monitor_interval_seconds()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except TimeoutError:
                continue

        HealthMonitorService._running = False
