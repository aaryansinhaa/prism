"""UI routes split from legacy frontend module for cleaner structure."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from app.core import run_model_container
from app.core.tunnel import start_tunnel
from app.models import ingest_upload_and_build, validate_upload_extension
from app.registry.container_registry import register_container, registry_path
from app.routing.ui.templates import (
    base_layout,
    predict_page,
    prediction_error_component,
    prediction_result_component,
    upload_page,
    upload_success_response,
)

router = APIRouter(tags=["frontend"])


@router.get("/", response_class=HTMLResponse)
async def upload_ui() -> str:
    return base_layout("PRISM - Model Upload", upload_page())


@router.post("/api/upload-and-run-ui", response_class=HTMLResponse)
async def upload_and_run_ui(
    file: UploadFile = File(...),
    enable_tunnel: bool = Form(False),
) -> str:
    try:
        file_name = file.filename or ""
        validate_upload_extension(file_name)

        upload_result = await ingest_upload_and_build(file)
        model_id = upload_result["model_id"]
        image_tag = upload_result["image_tag"]

        container_name, host_port, container_id = await asyncio.to_thread(
            run_model_container,
            model_id,
            image_tag,
        )

        tunnel_url = None
        tunnel_warning = None
        if enable_tunnel:
            try:
                tunnel_url, _ = await start_tunnel(host_port, model_id)
            except RuntimeError as exc:
                tunnel_warning = str(exc)
                print(f"Warning: Failed to start tunnel for {model_id}: {exc}")

        register_container(
            model_id=model_id,
            container_id=container_id,
            port=host_port,
            tunnel_url=tunnel_url,
        )

        return upload_success_response(
            model_id=model_id,
            port=host_port,
            tunnel_url=tunnel_url,
            tunnel_warning=tunnel_warning,
        )
    except HTTPException as exc:
        return f"<div class=\"alert-error\">Error: {exc.detail}</div>"
    except RuntimeError as exc:
        return f"<div class=\"alert-error\">Error: {str(exc)}</div>"
    except Exception as exc:
        return f"<div class=\"alert-error\">Unexpected error: {str(exc)}</div>"


@router.get("/predict", response_class=HTMLResponse)
async def predict_ui(model_id: str | None = None) -> str:
    return base_layout("PRISM - Make Predictions", predict_page())


@router.post("/predict-result", response_class=HTMLResponse)
async def predict_result(
    model_id: str = Form(...),
    input_data: str = Form(...),
) -> str:
    try:
        payload = json.loads(input_data)
    except json.JSONDecodeError as exc:
        return prediction_error_component(f"Invalid JSON: {exc}", model_id)

    path = registry_path()
    if not path.exists():
        return prediction_error_component("Model registry not found", model_id)

    try:
        with path.open("r", encoding="utf-8") as file:
            data: Dict[str, Any] = json.load(file)
    except (OSError, json.JSONDecodeError):
        return prediction_error_component("Failed to read model registry", model_id)

    models = data.get("models", {})
    model_entry = models.get(model_id)
    if not model_entry:
        return prediction_error_component(f"Model not found: {model_id}", model_id)

    port = model_entry.get("port")
    if port is None:
        return prediction_error_component("Model port not found in registry", model_id)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"http://127.0.0.1:{port}/predict", json=payload)
        if response.status_code != 200:
            return prediction_error_component(
                f"Server returned {response.status_code}: {response.text}",
                model_id,
            )
        return prediction_result_component(response.json(), model_id)
    except httpx.RequestError as exc:
        return prediction_error_component(f"Cannot connect to model: {exc}", model_id)
    except Exception as exc:
        return prediction_error_component(str(exc), model_id)
