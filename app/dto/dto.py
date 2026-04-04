"""Data Transfer Objects (DTOs) for PRISM application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class UploadModelDTO:
    """DTO for model upload request."""
    filename: str
    enable_tunnel: bool = False


@dataclass
class ModelCardDTO:
    """DTO for rendering a model card on dashboard."""
    model_id: str
    container_id: str
    port: int
    status_text: str
    status_class: str
    indicator_class: str
    predict_url: str
    api_url: str
    tunnel_url: Optional[str] = None


@dataclass
class DashboardDTO:
    """DTO for dashboard page data."""
    model_cards: list[ModelCardDTO]
    has_models: bool

    @property
    def empty_state_shown(self) -> bool:
        """Whether to show empty state message."""
        return not self.has_models


@dataclass
class DeploymentResultDTO:
    """DTO for model deployment result."""
    model_id: str
    container_id: str
    port: int
    tunnel_url: Optional[str] = None
    tunnel_warning: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class ContainerLogsDTO:
    """DTO for container logs response."""
    container_id: str
    logs: str
    error: Optional[str] = None

    @property
    def has_error(self) -> bool:
        """Whether logs retrieval failed."""
        return self.error is not None


@dataclass
class DeleteResultDTO:
    """DTO for deletion result."""
    success: bool
    message: str
    deleted_count: int = 1


@dataclass
class HtmlResponseDTO:
    """Generic HTML response DTO."""
    html_content: str
    status_code: int = 200
