import asyncio

from fastapi import FastAPI

from app.routing import frontend, health, inference, models, predict, registry
from app.services.health_monitor_service import HealthMonitorService
from runtime.model_loaders import load_model, default_model_path

app = FastAPI()

_health_monitor_stop_event: asyncio.Event | None = None
_health_monitor_task: asyncio.Task | None = None


@app.on_event("startup")
async def load_runtime_model() -> None:
    """Load model at application startup.

    The loader uses `MODEL_PATH` env var if set, otherwise falls back to
    `model_store/linear_regression.onnx` in the repository.
    """
    import os
    from typing import Any, Dict

    model_path = os.environ.get("MODEL_PATH") or default_model_path()
    try:
        model = load_model(model_path)
        predict.set_loaded_model(model)
        print(f"Loaded model from: {model_path}")
    except Exception as exc:
        print(f"Failed to load model from {model_path}: {exc}")

        class _FallbackModel:
            def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {"predictions": input_data}

        predict.set_loaded_model(_FallbackModel())
        print("Registered fallback model for development/testing.")

    global _health_monitor_stop_event
    global _health_monitor_task

    _health_monitor_stop_event = asyncio.Event()
    _health_monitor_task = asyncio.create_task(
        HealthMonitorService.run_forever(_health_monitor_stop_event)
    )
    print("Health monitor started")


@app.on_event("shutdown")
async def stop_health_monitor() -> None:
    global _health_monitor_stop_event
    global _health_monitor_task

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


# Include modular routers
app.include_router(frontend.router)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(models.router)
app.include_router(inference.router)
app.include_router(registry.router)
