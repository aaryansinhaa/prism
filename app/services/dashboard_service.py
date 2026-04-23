"""Service layer for dashboard business logic."""

from __future__ import annotations

import asyncio
import json

from app.dto.dto import (
    ContainerLogsDTO,
    DashboardDTO,
    DeleteResultDTO,
    ModelCardDTO,
)
from app.models.model import ModelMetadata
from app.registry.container_registry import (
    load_registry,
    remove_model_version,
    registry_path,
)
from app.utils.docker_utils import (
    build_api_url,
    build_prediction_url,
    container_exists,
    delete_container,
    get_container_logs,
    get_container_status,
    restart_container,
)


class ModelRegistryService:
    """Service for managing model registry."""

    @staticmethod
    def load_all_models() -> dict[str, ModelMetadata]:
        """Load all models from registry."""
        try:
            data = load_registry()
            models_data = data.get("models", {})
            if not isinstance(models_data, dict):
                return {}

            result: dict[str, ModelMetadata] = {}
            for model_id, entry in models_data.items():
                if not isinstance(entry, dict):
                    continue

                versions = entry.get("versions")
                if isinstance(versions, dict) and versions:
                    active_version = entry.get("active_version")
                    if not isinstance(active_version, str) or not active_version:
                        active_version = next(iter(versions.keys()))
                    active_entry = versions.get(active_version)
                    if not isinstance(active_entry, dict):
                        active_entry = next(iter(versions.values()))
                        active_version = str(active_entry.get("version", "v1"))

                    result[model_id] = ModelMetadata(
                        model_id=model_id,
                        container_id=active_entry.get("container_id", "unknown"),
                        port=int(active_entry.get("port", 0) or 0),
                        version=str(active_entry.get("version", active_version)),
                        active_version=active_version,
                        available_versions=sorted(versions.keys()),
                        name=active_entry.get("name"),
                        description=active_entry.get("description"),
                        expected_input_json=active_entry.get("expected_input_json"),
                        tunnel_url=active_entry.get("tunnel_url"),
                    )
                    continue

                version = entry.get("version")
                result[model_id] = ModelMetadata(
                    model_id=model_id,
                    container_id=entry.get("container_id", "unknown"),
                    port=entry.get("port", 0),
                    version=version if isinstance(version, str) else None,
                    active_version=version if isinstance(version, str) else None,
                    available_versions=[version] if isinstance(version, str) else None,
                    name=entry.get("name"),
                    description=entry.get("description"),
                    expected_input_json=entry.get("expected_input_json"),
                    tunnel_url=entry.get("tunnel_url"),
                )

            return result
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def remove_model_from_registry(model_id: str, version: str | None = None) -> bool:
        """Remove model from registry."""
        return remove_model_version(model_id, version=version)

    @staticmethod
    def clear_all_models() -> int:
        """Clear all models from registry. Returns count."""
        try:
            data = load_registry()
            count = len(data.get("models", {}))
            data["models"] = {}
            with registry_path().open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)
            return count
        except (OSError, json.JSONDecodeError):
            return 0

    @staticmethod
    def prune_stale_models() -> list[str]:
        """Remove registry entries whose Docker containers no longer exist."""
        try:
            data = load_registry()
            models = data.get("models", {})
            if not isinstance(models, dict):
                return []

            removed_model_ids: list[str] = []
            for model_id, metadata in list(models.items()):
                if not isinstance(metadata, dict):
                    continue

                versions = metadata.get("versions")
                if isinstance(versions, dict) and versions:
                    removed_versions: list[str] = []
                    for version, version_metadata in list(versions.items()):
                        if not isinstance(version_metadata, dict):
                            continue
                        container_id = version_metadata.get("container_id")
                        if not isinstance(container_id, str) or not container_id:
                            continue
                        if not container_exists(container_id):
                            del versions[version]
                            removed_versions.append(version)
                    if removed_versions:
                        removed_model_ids.extend(
                            [f"{model_id}:{version}" for version in removed_versions]
                        )
                    if not versions:
                        del models[model_id]
                    elif metadata.get("active_version") not in versions:
                        metadata["active_version"] = next(iter(versions.keys()))
                    continue

                container_id = metadata.get("container_id")
                if not isinstance(container_id, str) or not container_id:
                    continue

                if not container_exists(container_id):
                    del models[model_id]
                    removed_model_ids.append(model_id)

            if removed_model_ids:
                with registry_path().open("w", encoding="utf-8") as file:
                    json.dump(data, file, indent=2)

            return removed_model_ids
        except (OSError, json.JSONDecodeError):
            return []


class ContainerService:
    """Service for Docker container operations."""

    @staticmethod
    async def delete_model_async(model_id: str, container_id: str) -> DeleteResultDTO:
        """Delete a model and its container."""
        # Delete container
        success, message = await asyncio.to_thread(delete_container, container_id)

        if not success:
            return DeleteResultDTO(success=False, message=message, deleted_count=0)

        # Remove from registry
        ModelRegistryService.remove_model_from_registry(model_id)

        return DeleteResultDTO(
            success=True,
            message=f"Model '{model_id}' deleted successfully",
            deleted_count=1,
        )

    @staticmethod
    async def kill_all_models_async() -> DeleteResultDTO:
        """Delete all models and their containers."""
        models = ModelRegistryService.load_all_models()
        deleted_count = 0
        failed_count = 0

        for model_id, metadata in models.items():
            success, _ = await asyncio.to_thread(
                delete_container, metadata.container_id
            )
            if success:
                deleted_count += 1
            else:
                failed_count += 1

        # Clear registry
        ModelRegistryService.clear_all_models()

        message = f"Deleted {deleted_count} model(s)"
        if failed_count > 0:
            message += f" ({failed_count} failed)"

        return DeleteResultDTO(
            success=True, message=message, deleted_count=deleted_count
        )

    @staticmethod
    async def restart_model_async(container_id: str) -> tuple[bool, str]:
        """Restart a model container."""
        return await asyncio.to_thread(restart_container, container_id)


class DashboardService:
    """Service for dashboard rendering and data."""

    @staticmethod
    def build_dashboard_dto() -> DashboardDTO:
        """Build complete dashboard DTO."""
        models = ModelRegistryService.load_all_models()
        model_cards = []

        for model_id, metadata in sorted(models.items()):
            status = get_container_status(metadata.container_id)
            tunnel_prediction_url = None
            if metadata.tunnel_url:
                tunnel_prediction_url = (
                    f"{metadata.tunnel_url.rstrip('/')}/predict?model_id={model_id}"
                )
            card = ModelCardDTO(
                model_id=model_id,
                model_name=metadata.name or model_id,
                description=metadata.description,
                expected_input_json=metadata.expected_input_json,
                container_id=metadata.container_id,
                port=metadata.port,
                status_text=status.status_text,
                status_class=status.badge_class,
                indicator_class=status.indicator_class,
                predict_url=build_prediction_url(
                    model_id, version=metadata.active_version
                ),
                api_url=build_api_url(
                    model_id,
                    base_url=f"http://127.0.0.1:{metadata.port}",
                    version=metadata.active_version,
                ),
                tunnel_url=metadata.tunnel_url,
                tunnel_prediction_url=tunnel_prediction_url,
            )
            model_cards.append(card)

        return DashboardDTO(
            model_cards=model_cards,
            has_models=len(model_cards) > 0,
        )


class ContainerLogsService:
    """Service for retrieving container logs."""

    @staticmethod
    def get_container_logs_dto(container_id: str, lines: int = 50) -> ContainerLogsDTO:
        """Get container logs as DTO."""
        logs = get_container_logs(container_id, lines)
        return ContainerLogsDTO(container_id=container_id, logs=logs)
