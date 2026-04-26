"""UI routes split from legacy frontend module for cleaner structure."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from app.core import run_model_container
from app.core.input_contract import validate_payload_against_expected_input_json
from app.core.tunnel import start_tunnel
from app.models import ingest_upload_and_build, validate_upload_extension
from app.registry.container_registry import (
    register_container,
    registry_path,
    resolve_model_version_entry,
)
from app.registry.container_registry import (
    register_container_instance,
    load_registry,
    save_registry,
)
from app.routing.ui.templates import (
    base_layout,
    dashboard_page_with_cards,
    model_logs_modal,
    predict_page,
    prediction_error_component,
    prediction_result_component,
    upload_model_page,
    upload_success_response,
    model_control_center_page,
)
from app.services.dashboard_service import (
    ContainerService,
    DashboardService,
    ModelRegistryService,
)
from app.services.metrics_service import MetricsRegistry
from app.utils.qr_utils import generate_qr_data_uri

router = APIRouter(tags=["frontend"])


def _single_active_mode_enabled() -> bool:
    env_value = os.environ.get("PRISM_SINGLE_ACTIVE_MODEL", "true").lower()
    return env_value not in {"0", "false", "no", "off"}


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _validate_expected_input_json(raw_value: str | None) -> str | None:
    """Validate expected input format JSON and return normalized string."""
    cleaned = _clean_text(raw_value)
    if cleaned is None:
        return None

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid expected input JSON format: {exc}",
        )

    return json.dumps(parsed, ensure_ascii=False)


def _public_app_port() -> int:
    raw_port = os.environ.get("PRISM_PUBLIC_APP_PORT", "8000")
    try:
        return int(raw_port)
    except (TypeError, ValueError):
        return 8000


@router.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    """Render main dashboard with all deployed models."""
    dashboard_dto = DashboardService.build_dashboard_dto()
    html_content = dashboard_page_with_cards(
        dashboard_dto.model_cards, dashboard_dto.has_models
    )
    return base_layout(
        "PRISM - Dashboard", html_content, show_sidebar=True, active_nav="dashboard"
    )


@router.get("/upload-model", response_class=HTMLResponse)
async def upload_model_ui() -> str:
    """Render upload model page."""
    return base_layout(
        "PRISM - Upload Model",
        upload_model_page(),
        show_sidebar=True,
        active_nav="upload",
    )


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
    return base_layout(
        "PRISM - Model Logs", logs_content, show_sidebar=True, active_nav="logs"
    )


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
            return """<div class="model-card">
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
    model_id: str | None = Form(None),
    version: str | None = Form(None),
    model_name: str | None = Form(None),
    model_description: str | None = Form(None),
    expected_input_json: str | None = Form(None),
    enable_tunnel: bool = Form(False),
    replicas: int = Form(1),
    load_balancing_strategy: str | None = Form("round-robin"),
    enable_caching: bool = Form(False),
    cache_ttl: int | None = Form(None),
    enable_batching: bool = Form(False),
    batch_size: int | None = Form(None),
    batch_timeout: int | None = Form(None),
) -> str:
    try:
        file_name = file.filename or ""
        validate_upload_extension(file_name)

        cleaned_model_id = _clean_text(model_id)
        cleaned_version = _clean_text(version)
        cleaned_name = _clean_text(model_name)
        cleaned_description = _clean_text(model_description)
        validated_expected_input_json = _validate_expected_input_json(
            expected_input_json
        )

        upload_result = await ingest_upload_and_build(
            file,
            model_id=cleaned_model_id,
            version=cleaned_version,
        )
        model_id = upload_result["model_id"]
        version = upload_result.get("version")
        deployment_key = upload_result["deployment_key"]
        image_tag = upload_result["image_tag"]

        # Support deploying multiple replicas for simple load balancing
        instances: list[dict] = []
        hosts: list[int] = []
        for i in range(max(1, int(replicas or 1))):
            instance_key = deployment_key if i == 0 else f"{deployment_key}__{i}"
            container_name, host_port, container_id = await asyncio.to_thread(
                run_model_container,
                instance_key,
                image_tag,
            )
            instances.append(
                {
                    "container_id": container_id,
                    "port": host_port,
                    "instance_index": i,
                }
            )
            hosts.append(host_port)

        tunnel_url = None
        tunnel_warning = None
        qr_data_uri = None
        if enable_tunnel:
            try:
                # Tunnel the main PRISM app so public URL matches local UI path format
                tunnel_url, _ = await start_tunnel(_public_app_port(), deployment_key)
                if tunnel_url:
                    tunnel_prediction_url = (
                        f"{tunnel_url.rstrip('/')}/predict?model_id={model_id}"
                    )
                    qr_data_uri = generate_qr_data_uri(tunnel_prediction_url)
            except RuntimeError as exc:
                tunnel_warning = str(exc)
                print(f"Warning: Failed to start tunnel for {deployment_key}: {exc}")

        if _single_active_mode_enabled():
            await ContainerService.kill_all_models_async()

        # Register either as single container or multiple instances
        if len(instances) == 1:
            inst = instances[0]
            register_container(
                model_id=model_id,
                version=version,
                container_id=inst["container_id"],
                port=inst["port"],
                name=cleaned_name,
                description=cleaned_description,
                expected_input_json=validated_expected_input_json,
                tunnel_url=tunnel_url,
            )
        else:
            # Use register_container_instance to add instances per version
            for inst in instances:
                register_container_instance(
                    model_id=model_id,
                    container_id=inst["container_id"],
                    port=inst["port"],
                    version=version,
                    instance_index=inst["instance_index"],
                    name=cleaned_name,
                    description=cleaned_description,
                    expected_input_json=validated_expected_input_json,
                    tunnel_url=tunnel_url,
                )

        # Persist configuration metadata in registry under model -> versions -> <version> -> config
        try:
            data = load_registry()
            models = data.setdefault("models", {})
            entry = models.get(model_id)
            model_version = version or "v1"
            if isinstance(entry, dict):
                versions = (
                    entry.get("versions")
                    if isinstance(entry.get("versions"), dict)
                    else None
                )
                if versions and model_version in versions:
                    version_entry = versions[model_version]
                else:
                    # fallback to legacy shape
                    version_entry = entry
                cfg = {
                    "load_balancing": {
                        "replicas": int(replicas or 1),
                        "strategy": load_balancing_strategy,
                    },
                    "caching": {
                        "enabled": bool(enable_caching),
                        "ttl": int(cache_ttl) if cache_ttl is not None else None,
                    },
                    "batching": {
                        "enabled": bool(enable_batching),
                        "batch_size": int(batch_size) if batch_size else None,
                        "batch_timeout_ms": (
                            int(batch_timeout) if batch_timeout else None
                        ),
                    },
                }
                version_entry["config"] = cfg
                save_registry(data)
        except Exception:
            # Non-critical: don't block deployment if registry saving fails
            pass

        success_html = upload_success_response(
            model_id=model_id,
            port=host_port,
            tunnel_url=tunnel_url,
            tunnel_warning=tunnel_warning,
            qr_data_uri=qr_data_uri,
        )
        # Add button to return to dashboard
        return (
            success_html
            + '<div class="mt-6"><a href="/" hx-boost="true" class="btn-secondary w-full text-center block">Return to Dashboard</a></div>'
        )
    except HTTPException as exc:
        return f'<div class="alert-error">Error: {exc.detail}</div>'
    except RuntimeError as exc:
        return f'<div class="alert-error">Error: {str(exc)}</div>'
    except Exception as exc:
        return f'<div class="alert-error">Unexpected error: {str(exc)}</div>'


@router.get("/predict", response_class=HTMLResponse)
async def predict_ui(model_id: str | None = None, version: str | None = None) -> str:
    model_name: str | None = None
    model_description: str | None = None
    expected_input_json: str | None = None

    if model_id:
        path = registry_path()
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as file:
                    data: Dict[str, Any] = json.load(file)
                model_entry, resolved_version, _ = resolve_model_version_entry(
                    model_id,
                    version=version,
                    registry=data,
                )
                if isinstance(model_entry, dict):
                    model_name = model_entry.get("name")
                    model_description = model_entry.get("description")
                    expected_input_json = model_entry.get("expected_input_json")
                    version = resolved_version
            except (OSError, json.JSONDecodeError, KeyError):
                pass

    return base_layout(
        "PRISM - Make Predictions",
        predict_page(
            model_name=model_name,
            model_description=model_description,
            expected_input_json=expected_input_json,
            version=version,
        ),
        show_sidebar=False,
    )


@router.get("/model/{model_id}/control", response_class=HTMLResponse)
async def model_control_page(model_id: str, version: str | None = None) -> str:
    """Render a minimal control center for a specific model."""
    try:
        data = load_registry()
        # Resolve versioned entry if available
        model_entry, resolved_version, _ = resolve_model_version_entry(
            model_id, version=version, registry=data
        )
        html = model_control_center_page(model_id, model_entry, resolved_version)
    except Exception:
        html = model_control_center_page(model_id, None, version)

    return base_layout(f"PRISM - Control {model_id}", html, show_sidebar=True)


@router.get("/api/model-metrics")
async def api_model_metrics(model_id: str) -> JSONResponse:
    """Return real metrics for the model from the metrics registry.

    Returns:
        JSON with keys:
        - labels: Time labels (1, 2, 3, ...)
        - requests: Request counts over time
        - latency: Latency values in milliseconds
        - throughput: Requests per second
        - error_rate: Error percentage (optional)
        - cpu_usage: CPU percentage (optional)
        - memory_usage: Memory in MB (optional)
    """
    metrics_registry = MetricsRegistry.get_instance()
    metrics = metrics_registry.get_metrics(model_id)

    if metrics is None:
        # Model not in registry, return empty metrics with placeholder data
        return JSONResponse(
            {
                "labels": [str(i + 1) for i in range(20)],
                "requests": [0] * 20,
                "latency": [0.0] * 20,
                "throughput": [0.0] * 20,
                "error_rate": [0.0] * 20,
            }
        )

    response_data = {
        "labels": metrics.labels,
        "requests": metrics.requests,
        "latency": metrics.latency,
        "throughput": metrics.throughput,
    }

    if metrics.error_rate is not None:
        response_data["error_rate"] = metrics.error_rate
    if metrics.cpu_usage is not None:
        response_data["cpu_usage"] = metrics.cpu_usage
    if metrics.memory_usage is not None:
        response_data["memory_usage"] = metrics.memory_usage

    return JSONResponse(response_data)


@router.post("/api/record-metrics")
async def record_metrics(
    model_id: str = Form(...),
    latency_ms: float = Form(...),
    success: bool = Form(True),
    cpu_percent: float | None = Form(None),
    memory_mb: float | None = Form(None),
) -> JSONResponse:
    """Record a single request metric for a model.

    Args:
        model_id: Model identifier
        latency_ms: Request latency in milliseconds
        success: Whether request succeeded
        cpu_percent: CPU usage percentage (optional)
        memory_mb: Memory usage in MB (optional)

    Returns:
        JSON acknowledgment
    """
    metrics_registry = MetricsRegistry.get_instance()
    metrics_registry.record_request(
        model_id=model_id,
        latency_ms=latency_ms,
        success=success,
        cpu_percent=cpu_percent,
        memory_mb=memory_mb,
    )
    return JSONResponse({"status": "recorded"})


@router.post("/api/register-model-metrics")
async def register_model_metrics(
    model_id: str = Form(...),
    container_id: str = Form(...),
) -> JSONResponse:
    """Register a model for metrics tracking.

    Args:
        model_id: Model identifier
        container_id: Docker container ID

    Returns:
        JSON acknowledgment
    """
    metrics_registry = MetricsRegistry.get_instance()
    metrics_registry.register_model(model_id, container_id)
    return JSONResponse({"status": "registered", "model_id": model_id})


@router.get("/api/metrics-config")
async def get_metrics_config(model_id: str) -> JSONResponse:
    """Get current metrics configuration for a model.

    This endpoint returns the current metrics settings that can be adjusted
    from the model control center. Users can tune:
    - Display window size (number of data points)
    - Update interval
    - Thresholds for warnings/alerts
    - Which metrics to display

    Args:
        model_id: Model identifier

    Returns:
        JSON with current configuration
    """
    # Load from registry if available
    try:
        data = load_registry()
        model_entry, _, _ = resolve_model_version_entry(model_id, registry=data)
        config = model_entry.get("metrics_config", {})
    except Exception:
        config = {}

    # Provide sensible defaults
    return JSONResponse(
        {
            "model_id": model_id,
            "window_size": config.get("window_size", 60),  # 60 data points
            "update_interval_ms": config.get("update_interval_ms", 1000),  # 1 second
            "display_metrics": config.get(
                "display_metrics", ["latency", "throughput", "requests", "error_rate"]
            ),
            "latency_warning_threshold_ms": config.get(
                "latency_warning_threshold_ms", 1000
            ),
            "error_rate_warning_threshold_pct": config.get(
                "error_rate_warning_threshold_pct", 5.0
            ),
            "chart_colors": config.get(
                "chart_colors",
                {
                    "requests": "#000000",
                    "latency": "#ff9900",
                    "throughput": "#0066ff",
                    "error_rate": "#ff0000",
                    "cpu_usage": "#00cc00",
                    "memory_usage": "#ff6600",
                },
            ),
        }
    )


@router.post("/api/metrics-config")
async def update_metrics_config(
    model_id: str = Form(...),
    window_size: int | None = Form(None),
    update_interval_ms: int | None = Form(None),
    latency_warning_threshold_ms: int | None = Form(None),
    error_rate_warning_threshold_pct: float | None = Form(None),
) -> JSONResponse:
    """Update metrics configuration for a model from the control center.

    Args:
        model_id: Model identifier
        window_size: Number of data points to display (1-600)
        update_interval_ms: How often to fetch metrics (100-10000 ms)
        latency_warning_threshold_ms: Warn if latency exceeds this (ms)
        error_rate_warning_threshold_pct: Warn if error rate exceeds this (%)

    Returns:
        JSON with updated configuration
    """
    try:
        # Validate inputs
        if window_size is not None:
            window_size = max(1, min(600, int(window_size)))
        if update_interval_ms is not None:
            update_interval_ms = max(100, min(10000, int(update_interval_ms)))
        if latency_warning_threshold_ms is not None:
            latency_warning_threshold_ms = max(1, int(latency_warning_threshold_ms))
        if error_rate_warning_threshold_pct is not None:
            error_rate_warning_threshold_pct = max(
                0.0, min(100.0, float(error_rate_warning_threshold_pct))
            )

        # Load and update registry
        data = load_registry()
        model_entry, resolved_version, _ = resolve_model_version_entry(
            model_id, registry=data
        )

        # Build metrics config
        metrics_config = model_entry.get("metrics_config", {})
        if window_size is not None:
            metrics_config["window_size"] = window_size
        if update_interval_ms is not None:
            metrics_config["update_interval_ms"] = update_interval_ms
        if latency_warning_threshold_ms is not None:
            metrics_config["latency_warning_threshold_ms"] = (
                latency_warning_threshold_ms
            )
        if error_rate_warning_threshold_pct is not None:
            metrics_config["error_rate_warning_threshold_pct"] = (
                error_rate_warning_threshold_pct
            )

        model_entry["metrics_config"] = metrics_config
        save_registry(data)

        return JSONResponse(
            {
                "status": "updated",
                "model_id": model_id,
                "config": metrics_config,
            }
        )
    except Exception as e:
        return JSONResponse(
            {
                "status": "error",
                "error": str(e),
            },
            status_code=400,
        )


@router.post("/predict-result", response_class=HTMLResponse)
async def predict_result(
    model_id: str = Form(...),
    version: str | None = Form(None),
    input_data: str = Form(...),
) -> str:
    try:
        payload = json.loads(input_data)
    except json.JSONDecodeError as exc:
        return prediction_error_component(f"Invalid JSON: {exc}", model_id, version)

    path = registry_path()
    if not path.exists():
        return prediction_error_component("Model registry not found", model_id, version)

    try:
        with path.open("r", encoding="utf-8") as file:
            data: Dict[str, Any] = json.load(file)
    except (OSError, json.JSONDecodeError):
        return prediction_error_component(
            "Failed to read model registry", model_id, version
        )

    try:
        model_entry, resolved_version, _ = resolve_model_version_entry(
            model_id,
            version=version,
            registry=data,
        )
    except KeyError:
        if version:
            return prediction_error_component(
                f"Model not found: {model_id} version {version}",
                model_id,
                version,
            )
        return prediction_error_component(f"Model not found: {model_id}", model_id)

    is_valid, error = validate_payload_against_expected_input_json(
        model_entry.get("expected_input_json"),
        payload,
    )
    if not is_valid:
        return prediction_error_component(
            f"Input does not match expected format: {error}",
            model_id,
            resolved_version,
        )

    port = model_entry.get("port")
    if port is None:
        return prediction_error_component(
            "Model port not found in registry",
            model_id,
            resolved_version,
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/predict", json=payload
            )
        if response.status_code != 200:
            return prediction_error_component(
                f"Server returned {response.status_code}: {response.text}",
                model_id,
                resolved_version,
            )
        return prediction_result_component(response.json(), model_id, resolved_version)
    except httpx.RequestError as exc:
        return prediction_error_component(
            f"Cannot connect to model: {exc}", model_id, resolved_version
        )
    except Exception as exc:
        return prediction_error_component(str(exc), model_id, resolved_version)
