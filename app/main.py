from fastapi import FastAPI

from app.routing import frontend, health, inference, models, predict, registry
from runtime.model_loaders import load_model, default_model_path

app = FastAPI()


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


# Include modular routers
app.include_router(frontend.router)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(models.router)
app.include_router(inference.router)
app.include_router(registry.router)
