"""Model container registry routing."""

import json
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from starlette import status

from app.registry.container_registry import registry_path

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
    models = registry.get("models", {})

    if model_id not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found in registry",
        )

    return models[model_id]
