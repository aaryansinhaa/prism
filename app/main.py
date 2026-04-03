import os
from typing import Any, Dict

from fastapi import FastAPI, Body, HTTPException
from starlette import status

from runtime.model_loaders import load_model, default_model_path
from runtime.adapters.base import ModelPredictError, ModelLoadError


app = FastAPI()


@app.on_event("startup")
async def load_runtime_model() -> None:
    """Load model at application startup and attach to app.state.model.

    The loader uses `MODEL_PATH` env var if set, otherwise falls back to
    `model_store/linear_regression.onnx` in the repository.
    """
    model_path = os.environ.get("MODEL_PATH") or default_model_path()
    try:
        model = load_model(model_path)
        app.state.model = model
        print(f"Loaded model from: {model_path}")
    except Exception as exc:  # keep startup resilient and surface load errors
        # If the real model cannot be loaded (missing optional deps, etc.)
        # install a small in-memory fallback model so the service can be
        # exercised during development.
        print(f"Failed to load model from {model_path}: {exc}")

        class _FallbackModel:
            def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                # Echo input to allow quick smoke-testing via curl.
                return {"predictions": input_data}

        app.state.model = _FallbackModel()
        print("Registered fallback model for development/testing.")


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"service": "prism-runtime"}


@app.post("/predict")
async def predict(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Run prediction using the loaded model.

    This endpoint accepts a JSON mapping in the body and returns a
    JSON-serializable response produced by the adapter.
    """
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    try:
        normalized_payload = getattr(model, "validate_input", lambda value: value)(payload)
        result = model.predict(normalized_payload)
    except ModelPredictError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except ModelLoadError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return result
