import asyncio
import os

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from app.routing import frontend, health, inference, models, predict, registry
from app.services.health_monitor_service import HealthMonitorService
from runtime.model_loaders import load_model

# Load environment variables from .env file
load_dotenv()


def _health_monitor_enabled() -> bool:
    configured = os.environ.get("PRISM_ENABLE_HEALTH_MONITOR", "true").lower()
    if configured in {"0", "false", "no", "off"}:
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at application startup and manage health monitor lifecycle."""
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        predict.set_loaded_model(None)
        print("No MODEL_PATH configured; skipping default runtime model load.")
    else:
        try:
            model = load_model(model_path)
            predict.set_loaded_model(model)
            print(f"Loaded model from: {model_path}")
        except Exception as exc:
            predict.set_loaded_model(None)
            print(f"Failed to load model from {model_path}: {exc}")

    global _health_monitor_stop_event
    global _health_monitor_task

    if _health_monitor_enabled():
        _health_monitor_stop_event = asyncio.Event()
        _health_monitor_task = asyncio.create_task(
            HealthMonitorService.run_forever(_health_monitor_stop_event)
        )
        print("Health monitor started")
    else:
        _health_monitor_stop_event = None
        _health_monitor_task = None

    yield

    if _health_monitor_stop_event is not None:
        _health_monitor_stop_event.set()

    if _health_monitor_task is not None:
        try:
            await _health_monitor_task
        except Exception as exc:
            print(f"Error while stopping health monitor: {exc}")
        finally:
            _health_monitor_task = None

    _health_monitor_stop_event = None


app = FastAPI(lifespan=lifespan)

_health_monitor_stop_event: asyncio.Event | None = None
_health_monitor_task: asyncio.Task | None = None


# Include modular routers
app.include_router(frontend.router)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(models.router)
app.include_router(inference.router)
app.include_router(registry.router)
