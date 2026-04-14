"""Model inference routing - forwards requests to deployed containers."""

import httpx
from time import perf_counter
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Request
from starlette import status

from app.core.access_control import enforce_rate_limit, log_access, validate_api_key
from app.core.input_contract import validate_payload_against_expected_input_json
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
    request: Request,
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    """Forward prediction request to deployed model container.

    Looks up the model's container port in the registry and forwards
    the request to the running container's /predict endpoint.
    """
    start_time = perf_counter()
    client_ip = request.client.host if request and request.client else "unknown"
    principal = "unknown"
    response_status = status.HTTP_500_INTERNAL_SERVER_ERROR

    try:
        principal = validate_api_key(request)
        enforce_rate_limit(principal)

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

        is_valid, error = validate_payload_against_expected_input_json(
            model_entry.get("expected_input_json"),
            payload,
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input does not match expected format: {error}",
            )

        container_url = f"http://127.0.0.1:{port}/predict"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(container_url, json=payload)
            response.raise_for_status()
            response_status = status.HTTP_200_OK
            return response.json()
    except httpx.ConnectError:
        response_status = status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot connect to model container at {container_url}",
        )
    except httpx.HTTPStatusError as exc:
        response_status = exc.response.status_code
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Container returned error: {exc.response.text}",
        )
    except httpx.RequestError as exc:
        response_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error communicating with container: {str(exc)}",
        )
    except HTTPException as exc:
        response_status = exc.status_code
        raise
    except Exception as exc:
        response_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(exc)}",
        )
    finally:
        latency_ms = (perf_counter() - start_time) * 1000
        log_access(
            model_id=model_id,
            principal=principal,
            client_ip=client_ip,
            status_code=response_status,
            latency_ms=latency_ms,
        )
