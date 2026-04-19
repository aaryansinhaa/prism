"""Model container registry routing."""

import json
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from starlette import status

from app.registry.container_registry import registry_path, resolve_model_version_entry
from app.services.dashboard_service import ModelRegistryService

router = APIRouter(prefix="/registry", tags=["registry"])


def _load_registry() -> Dict[str, Any]:
    path = registry_path()
    if not path.exists():
        return {"models": {}}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data if isinstance(data, dict) else {"models": {}}


@router.get("")
def list_registry() -> Dict[str, Any]:
    """List all registered model containers."""
    return _load_registry()


@router.get("/{model_id}")
def get_model_registry(model_id: str) -> Dict[str, Any]:
    """Get registry entry for a specific model."""
    registry = _load_registry()
    try:
        model_entry, resolved_version, active_version = resolve_model_version_entry(
            model_id,
            registry=registry,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found in registry",
        )

    response = dict(model_entry)
    response["model_id"] = model_id
    response["version"] = resolved_version
    if active_version:
        response["active_version"] = active_version
    return response


@router.get("/{model_id}/versions/{version}")
def get_model_registry_version(model_id: str, version: str) -> Dict[str, Any]:
    """Get a specific versioned registry entry for a model."""
    registry = _load_registry()
    try:
        model_entry, resolved_version, active_version = resolve_model_version_entry(
            model_id,
            version=version,
            registry=registry,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} version {version} not found in registry",
        )

    response = dict(model_entry)
    response["model_id"] = model_id
    response["version"] = resolved_version
    if active_version:
        response["active_version"] = active_version
    return response


@router.post("/prune-stale")
def prune_stale_registry_entries() -> Dict[str, Any]:
    """Remove registry entries for containers that no longer exist."""
    removed = ModelRegistryService.prune_stale_models()
    return {
        "removed_count": len(removed),
        "removed_model_ids": removed,
    }
