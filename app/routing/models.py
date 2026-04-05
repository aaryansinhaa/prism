"""Model upload and container launch routing."""

import asyncio
import json
import os
from typing import Any, Dict

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette import status

from app.core import run_model_container
from app.core.tunnel import start_tunnel
from app.models import ingest_upload_and_build, validate_upload_extension
from app.registry.container_registry import register_container, registry_path
from app.services.dashboard_service import ContainerService, ModelRegistryService

router = APIRouter(prefix="/models", tags=["models"])


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _validate_expected_input_json(raw_value: str | None) -> str | None:
    cleaned = _clean_text(raw_value)
    if cleaned is None:
        return None

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid expected input JSON format: {exc}",
        )

    return json.dumps(parsed, ensure_ascii=False)


@router.post("/upload")
async def upload_model(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload and build model without launching container."""
    file_name = file.filename or ""
    validate_upload_extension(file_name)
    return await ingest_upload_and_build(file)


@router.post("/upload-and-run")
async def upload_and_run_model(
    file: UploadFile = File(...),
    model_name: str | None = Form(None),
    model_description: str | None = Form(None),
    expected_input_json: str | None = Form(None),
) -> Dict[str, Any]:
    """Upload, build, and launch model container in one step."""
    file_name = file.filename or ""
    validate_upload_extension(file_name)
    cleaned_name = _clean_text(model_name)
    cleaned_description = _clean_text(model_description)
    validated_expected_input_json = _validate_expected_input_json(expected_input_json)

    upload_result = await ingest_upload_and_build(file)

    model_id = upload_result["model_id"]
    image_tag = upload_result["image_tag"]

    try:
        container_name, host_port, container_id = await asyncio.to_thread(
            run_model_container,
            model_id,
            image_tag,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    # Optionally start tunnel if enabled
    tunnel_url = None
    enable_tunnel = os.environ.get("ENABLE_TUNNEL", "false").lower() == "true"
    
    if enable_tunnel:
        try:
            tunnel_url, _ = await start_tunnel(host_port, model_id)
        except RuntimeError as exc:
            # Tunnel startup is not fatal - log and continue
            print(f"Warning: Failed to start tunnel for {model_id}: {exc}")

    registry_record = register_container(
        model_id=model_id,
        container_id=container_id,
        port=host_port,
        name=cleaned_name,
        description=cleaned_description,
        expected_input_json=validated_expected_input_json,
        tunnel_url=tunnel_url,
    )

    upload_result.update(
        {
            "container_name": container_name,
            "container_id": container_id,
            "host_port": host_port,
            "predict_url": f"http://127.0.0.1:{host_port}/predict",
            "tunnel_url": tunnel_url,
            "registry": registry_record,
            "registry_path": str(registry_path()),
        }
    )
    return upload_result


@router.post("")
async def create_model(
    file: UploadFile = File(...),
    model_name: str | None = Form(None),
    model_description: str | None = Form(None),
    expected_input_json: str | None = Form(None),
) -> Dict[str, Any]:
    """PRISM-11 create endpoint: upload, build, and run model."""
    return await upload_and_run_model(
        file=file,
        model_name=model_name,
        model_description=model_description,
        expected_input_json=expected_input_json,
    )


@router.get("")
def list_models() -> Dict[str, Any]:
    """PRISM-11 list endpoint: return all deployed model metadata."""
    models = ModelRegistryService.load_all_models()
    items: list[Dict[str, Any]] = []

    for model_id, metadata in sorted(models.items()):
        item: Dict[str, Any] = {
            "model_id": model_id,
            "container_id": metadata.container_id,
            "port": metadata.port,
        }
        if metadata.name:
            item["name"] = metadata.name
        if metadata.description:
            item["description"] = metadata.description
        if metadata.expected_input_json:
            item["expected_input_json"] = metadata.expected_input_json
        if metadata.tunnel_url:
            item["tunnel_url"] = metadata.tunnel_url
        items.append(item)

    return {"models": items, "count": len(items)}


@router.get("/{model_id}")
def get_model(model_id: str) -> Dict[str, Any]:
    """PRISM-11 get endpoint: return one deployed model metadata record."""
    models = ModelRegistryService.load_all_models()
    metadata = models.get(model_id)
    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found in registry",
        )

    item: Dict[str, Any] = {
        "model_id": model_id,
        "container_id": metadata.container_id,
        "port": metadata.port,
    }
    if metadata.name:
        item["name"] = metadata.name
    if metadata.description:
        item["description"] = metadata.description
    if metadata.expected_input_json:
        item["expected_input_json"] = metadata.expected_input_json
    if metadata.tunnel_url:
        item["tunnel_url"] = metadata.tunnel_url

    return item


@router.delete("/{model_id}")
async def delete_model(model_id: str) -> Dict[str, Any]:
    """PRISM-11 delete endpoint: stop/remove container and registry record."""
    models = ModelRegistryService.load_all_models()
    metadata = models.get(model_id)
    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found in registry",
        )

    result = await ContainerService.delete_model_async(
        model_id=model_id,
        container_id=metadata.container_id,
    )
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.message,
        )

    return {
        "success": True,
        "model_id": model_id,
        "deleted_count": result.deleted_count,
        "message": result.message,
    }
