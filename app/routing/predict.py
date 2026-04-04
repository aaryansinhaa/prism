"""Model prediction routing."""

from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException
from starlette import status

from runtime.adapters.base import ModelLoadError, ModelPredictError

router = APIRouter(tags=["predict"])

# Module-level model storage
_loaded_model = None


def get_loaded_model():
    """Get the currently loaded model."""
    global _loaded_model
    return _loaded_model


def set_loaded_model(model):
    """Set the loaded model."""
    global _loaded_model
    _loaded_model = model


@router.post("/predict")
async def predict(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Run prediction using the loaded model.

    This endpoint accepts a JSON mapping in the body and returns a
    JSON-serializable response produced by the adapter.
    """
    model = get_loaded_model()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        normalized_payload = getattr(model, "validate_input", lambda value: value)(
            payload
        )
        result = model.predict(normalized_payload)
    except ModelPredictError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except ModelLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return result
