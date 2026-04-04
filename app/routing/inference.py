"""Model inference routing - forwards requests to deployed containers."""

import httpx
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException
from starlette import status

from app.registry.container_registry import registry_path

router = APIRouter(prefix="/models", tags=["inference"])


def _load_registry() -> Dict[str, Any]:
    """Load the container registry."""
    path = registry_path()
    if not path.exists():
        return {"models": {}}

    try:
        import json
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {"models": {}}
    except (IOError, json.JSONDecodeError):
        return {"models": {}}


@router.post("/{model_id}/predict")
async def predict_model(
    model_id: str,
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    """Forward prediction request to deployed model container.
    
    Looks up the model's container port in the registry and forwards
    the request to the running container's /predict endpoint.
    """
    registry = _load_registry()
    models = registry.get("models", {})

    if model_id not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found in registry",
        )

    model_entry = models[model_id]
    port = model_entry.get("port")
    if port is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model {model_id} has no port configured",
        )

    container_url = f"http://127.0.0.1:{port}/predict"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(container_url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot connect to model container at {container_url}",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Container returned error: {exc.response.text}",
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error communicating with container: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(exc)}",
        )
