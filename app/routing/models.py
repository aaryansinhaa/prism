"""Model upload and container launch routing."""

import asyncio
import os
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette import status

from app.core import run_model_container
from app.core.tunnel import start_tunnel
from app.models import ingest_upload_and_build, validate_upload_extension
from app.registry.container_registry import register_container, registry_path

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/upload")
async def upload_model(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload and build model without launching container."""
    file_name = file.filename or ""
    validate_upload_extension(file_name)
    return await ingest_upload_and_build(file)


@router.post("/upload-and-run")
async def upload_and_run_model(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload, build, and launch model container in one step."""
    file_name = file.filename or ""
    validate_upload_extension(file_name)

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
