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
from app.registry.container_registry import registry_path
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
        path = registry_path()
        if not path.exists():
            return {}

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
                models_data = data.get("models", {})
                return {
                    model_id: ModelMetadata(
                        model_id=model_id,
                        container_id=m.get("container_id", "unknown"),
                        port=m.get("port", 0),
                        name=m.get("name"),
                        description=m.get("description"),
                        expected_input_json=m.get("expected_input_json"),
                        tunnel_url=m.get("tunnel_url"),
                    )
                    for model_id, m in models_data.items()
                }
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def remove_model_from_registry(model_id: str) -> bool:
        """Remove model from registry."""
        path = registry_path()
        if not path.exists():
            return False

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if model_id in data.get("models", {}):
                del data["models"][model_id]
                with path.open("w", encoding="utf-8") as file:
                    json.dump(data, file, indent=2)
                return True
            return False
        except (OSError, json.JSONDecodeError):
            return False

    @staticmethod
    def clear_all_models() -> int:
        """Clear all models from registry. Returns count."""
        path = registry_path()
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            count = len(data.get("models", {}))
            data["models"] = {}
            with path.open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)
            return count
        except (OSError, json.JSONDecodeError):
            return 0

    @staticmethod
    def prune_stale_models() -> list[str]:
        """Remove registry entries whose Docker containers no longer exist."""
        path = registry_path()
        if not path.exists():
            return []

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            models = data.get("models", {})
            if not isinstance(models, dict):
                return []

            removed_model_ids: list[str] = []
            for model_id, metadata in list(models.items()):
                if not isinstance(metadata, dict):
                    continue

                container_id = metadata.get("container_id")
                if not isinstance(container_id, str) or not container_id:
                    continue

                if not container_exists(container_id):
                    del models[model_id]
                    removed_model_ids.append(model_id)

            if removed_model_ids:
                with path.open("w", encoding="utf-8") as file:
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
                tunnel_prediction_url = f"{metadata.tunnel_url.rstrip('/')}/predict"
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
                predict_url=build_prediction_url(model_id),
                api_url=build_api_url(
                    model_id, base_url=f"http://127.0.0.1:{metadata.port}"
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
