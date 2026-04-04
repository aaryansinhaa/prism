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
from app.dto.dto import (
    HtmlResponseDTO,
    ModelCardDTO,
)
from app.models import ingest_upload_and_build, validate_upload_extension
from app.registry.container_registry import register_container, registry_path
from app.routing.ui.templates import (
    base_layout,
    dashboard_page_with_cards,
    model_logs_modal,
    predict_page,
    prediction_error_component,
    prediction_result_component,
    upload_page,
    upload_model_page,
    upload_success_response,
)
from app.services.dashboard_service import (
    ContainerLogsService,
    ContainerService,
    DashboardService,
    ModelRegistryService,
)
from app.utils.docker_utils import delete_container, get_container_logs

router = APIRouter(tags=["frontend"])


@router.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    """Render main dashboard with all deployed models."""
    dashboard_dto = DashboardService.build_dashboard_dto()
    html_content = dashboard_page_with_cards(dashboard_dto.model_cards, dashboard_dto.has_models)
    return base_layout("PRISM - Dashboard", html_content, show_sidebar=True, active_nav="dashboard")


@router.get("/upload-model", response_class=HTMLResponse)
async def upload_model_ui() -> str:
    """Render upload model page."""
    return base_layout("PRISM - Upload Model", upload_model_page(), show_sidebar=True, active_nav="upload")


@router.get("/model-logs", response_class=HTMLResponse)
async def model_logs_ui() -> str:
    """Render model logs view."""
    models = ModelRegistryService.load_all_models()

    log_entries = []
    for model_id, metadata in sorted(models.items()):
        container_id = metadata.container_id
        log_entries.append(f"""
    <div class="bg-white rounded-lg p-4 border border-gray-200 mb-4 cursor-pointer hover:border-blue-500" onclick="htmx.ajax('GET', '/api/model-logs?container_id={container_id}', '#modal-logs')">
        <div class="flex items-center justify-between">
            <div>
                <p class="font-semibold text-gray-900">{model_id}</p>
                <p class="text-xs text-gray-600">Container: {container_id[:12]}</p>
            </div>
            <span class="text-gray-400">→</span>
        </div>
    </div>
""")

    logs_content = f"""
<div class="max-w-4xl mx-auto">
    <h1 class="text-3xl font-bold text-gray-900 mb-2">📋 Model Logs</h1>
    <p class="text-gray-600 mb-8">View logs from your deployed model containers</p>

    <div class="space-y-2">
        {("".join(log_entries) if log_entries else '<div class="alert-info">No models deployed yet. Upload one from the Upload Model page.</div>')}
    </div>

    <div id="modal-logs"></div>
</div>
"""
    return base_layout("PRISM - Model Logs", logs_content, show_sidebar=True, active_nav="logs")


@router.get("/api/model-logs", response_class=HTMLResponse)
async def get_model_logs(container_id: str) -> str:
    """Get logs for a specific container."""
    return model_logs_modal(container_id)


@router.post("/api/restart-model", response_class=HTMLResponse)
async def restart_model(container_id: str = Form(...)) -> str:
    """Restart a model container."""
    try:
        success, message = await ContainerService.restart_model_async(container_id)
        if success:
            return f"""<div class="model-card">
    <div class="alert-success mb-4">✓ Container restarted successfully!</div>
    <a href="/" class="btn-secondary">Return to Dashboard</a>
</div>"""
        else:
            return f"""<div class="model-card">
    <div class="alert-warning mb-4">⚠ {message}</div>
    <a href="/" class="btn-secondary">Return to Dashboard</a>
</div>"""
    except Exception as e:
        return f"""<div class="model-card">
    <div class="alert-error mb-4">✗ Error restarting container: {str(e)}</div>
    <a href="/" class="btn-secondary">Return to Dashboard</a>
</div>"""


@router.delete("/api/delete-model", response_class=HTMLResponse)
async def delete_model(model_id: str = Form(...), container_id: str = Form(...)) -> str:
    """Delete a model and stop its container."""
    try:
        result_dto = await ContainerService.delete_model_async(model_id, container_id)

        if not result_dto.success:
            return f"""<div class="model-card">
    <div class="alert-error mb-4">✗ {result_dto.message}</div>
    <a href="/" class="btn-secondary">Return to Dashboard</a>
</div>"""

        return f"""<div class="bg-white rounded-lg p-6 text-center">
    <div class="alert-success mb-4">✓ Model '{model_id}' deleted successfully!</div>
    <p class="text-gray-600 mb-4">Container has been stopped and removed.</p>
    <a href="/" class="btn-secondary">Return to Dashboard</a>
</div>"""
    except Exception as e:
        return f"""<div class="model-card">
    <div class="alert-error mb-4">✗ Error deleting model: {str(e)}</div>
    <a href="/" class="btn-secondary">Return to Dashboard</a>
</div>"""


@router.delete("/api/kill-all-models", response_class=HTMLResponse)
async def kill_all_models() -> str:
    """Delete all models and stop all containers."""
    try:
        result_dto = await ContainerService.kill_all_models_async()

        return f"""<div class="bg-white rounded-lg p-6 text-center">
    <div class="alert-success mb-4">✓ {result_dto.message}</div>
    <p class="text-gray-600 mb-4">All containers have been stopped and removed.</p>
    <a href="/" class="btn-secondary">Return to Dashboard</a>
</div>"""
    except Exception as e:
        return f"""<div class="bg-white rounded-lg p-6">
    <div class="alert-error mb-4">✗ Error deleting all models: {str(e)}</div>
    <a href="/" class="btn-secondary">Return to Dashboard</a>
</div>"""


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

        success_html = upload_success_response(
            model_id=model_id,
            port=host_port,
            tunnel_url=tunnel_url,
            tunnel_warning=tunnel_warning,
        )
        # Add button to return to dashboard
        return success_html + '<div class="mt-6"><a href="/" hx-boost="true" class="btn-secondary w-full text-center block">Return to Dashboard</a></div>'
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
