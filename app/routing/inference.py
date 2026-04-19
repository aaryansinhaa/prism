"""Model inference routing - forwards requests to deployed containers."""

import httpx
from time import perf_counter
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Request
from starlette import status

from app.batching.request_batcher import request_batcher
from app.core.access_control import enforce_rate_limit, log_access, validate_api_key
from app.core.input_contract import validate_payload_against_expected_input_json
from app.registry.container_registry import (
    registry_path,
    resolve_model_version_entry,
    get_model_instances,
)
from app.services.load_balancer import get_load_balancer

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


def _build_versioned_prediction_url(model_id: str, version: str) -> str:
    return f"http://127.0.0.1:8000/models/{model_id}/versions/{version}/predict"


async def _predict_model(
    model_id: str,
    request: Request,
    payload: Dict[str, Any] = Body(...),
    version: str | None = None,
) -> Dict[str, Any]:
    """Forward prediction request to deployed model container.

    Uses load balancing to distribute requests across multiple instances
    of the same model version using round-robin scheduling.
    """
    start_time = perf_counter()
    client_ip = request.client.host if request and request.client else "unknown"
    principal = "unknown"
    response_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    container_url = None

    try:
        principal = validate_api_key(request)
        enforce_rate_limit(principal)

        registry = _load_registry()
        try:
            model_entry, resolved_version, _ = resolve_model_version_entry(
                model_id,
                version=version,
                registry=registry,
            )
        except KeyError:
            if version:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model {model_id} version {version} not found in registry",
                )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found in registry",
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

        # Get instances for load balancing
        instances = get_model_instances(model_id, resolved_version, registry)
        if not instances:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"No container instances available for {model_id}",
            )

        # Use load balancer to select instance
        lb = get_load_balancer()
        selected_port = None

        # If single instance, use directly
        if len(instances) == 1:
            selected_port = instances[0]["port"]
        else:
            # Multiple instances: initialize load balancer if not already done
            state = lb.get_state(model_id, resolved_version)
            if not state or state.get_total_count() == 0:
                # First time: register all instances in load balancer
                for inst in instances:
                    lb.register_instance(
                        model_id,
                        resolved_version,
                        inst["container_id"],
                        inst["port"],
                        inst.get("instance_index", 0),
                    )

            # Get next instance via round-robin
            lb_instance = lb.get_next_instance(model_id, resolved_version)
            if lb_instance:
                selected_port = lb_instance.port

        if not selected_port:
            selected_port = instances[0]["port"]

        container_url = f"http://127.0.0.1:{selected_port}/predict"

        response = await request_batcher.forward(container_url, payload)
        response_status = status.HTTP_200_OK
        return response
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


@router.post("/{model_id}/predict")
async def predict_model(
    model_id: str,
    request: Request,
    payload: Dict[str, Any] = Body(...),
    version: str | None = None,
) -> Dict[str, Any]:
    return await _predict_model(
        model_id=model_id, request=request, payload=payload, version=version
    )


@router.post("/{model_id}/versions/{version}/predict")
async def predict_model_versioned(
    model_id: str,
    version: str,
    request: Request,
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    return await _predict_model(
        model_id=model_id, request=request, payload=payload, version=version
    )


@router.get("/{model_id}/load-balancer/health")
async def get_load_balancer_health(
    model_id: str,
    version: str | None = None,
) -> Dict[str, Any]:
    """Get load balancer health status for a model version."""
    lb = get_load_balancer()
    health = lb.get_health_summary(model_id, version)

    if not health:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No load balancer state for {model_id}",
        )

    return health


@router.get("/{model_id}/instances")
async def list_model_instances(
    model_id: str,
    version: str | None = None,
) -> Dict[str, Any]:
    """List all instances for a model version."""
    registry = _load_registry()
    instances = get_model_instances(model_id, version, registry)

    if not instances:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No instances for {model_id}",
        )

    return {
        "model_id": model_id,
        "version": version or "v1",
        "instances": instances,
        "instance_count": len(instances),
    }

