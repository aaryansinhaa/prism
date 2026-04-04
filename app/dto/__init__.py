"""DTO (Data Transfer Object) schemas for PRISM API."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response after model upload."""

    model_id: str = Field(..., description="Unique model identifier")
    image_tag: str = Field(..., description="Docker image tag")
    model_path: str = Field(..., description="Path to saved model file")
    dockerfile_path: str = Field(..., description="Path to generated Dockerfile")
    build_context: str = Field(..., description="Build context directory")
    build_output: str = Field(..., description="Docker build output")


class ContainerRegistry(BaseModel):
    """Container registry entry."""

    model_id: str = Field(..., description="Model identifier")
    container_id: str = Field(..., description="Docker container ID")
    port: int = Field(..., description="Host port mapped to container")


class UploadAndRunResponse(UploadResponse):
    """Response after upload and container launch."""

    container_name: str = Field(..., description="Docker container name")
    container_id: str = Field(..., description="Docker container ID")
    host_port: int = Field(..., description="Host port for inference")
    predict_url: str = Field(..., description="URL to call /predict endpoint")
    tunnel_url: str | None = Field(None, description="Public tunnel URL (if tunnel enabled)")
    registry: ContainerRegistry = Field(..., description="Registry record")
    registry_path: str = Field(..., description="Path to registry file")


# Import clean architecture DTOs
from app.dto.dto import (
    ContainerLogsDTO,
    DashboardDTO,
    DeleteResultDTO,
    DeploymentResultDTO,
    HtmlResponseDTO,
    ModelCardDTO,
    UploadModelDTO,
)

__all__ = [
    "UploadResponse",
    "ContainerRegistry",
    "UploadModelDTO",
    "ModelCardDTO",
    "DashboardDTO",
    "DeploymentResultDTO",
    "ContainerLogsDTO",
    "DeleteResultDTO",
    "HtmlResponseDTO",
]