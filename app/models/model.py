"""Domain models (entities) for PRISM application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelMetadata:
    """Core model metadata entity."""

    model_id: str
    container_id: str
    port: int
    version: Optional[str] = None
    active_version: Optional[str] = None
    available_versions: Optional[list[str]] = None
    name: Optional[str] = None
    description: Optional[str] = None
    expected_input_json: Optional[str] = None
    tunnel_url: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        result = {
            "model_id": self.model_id,
            "container_id": self.container_id,
            "port": self.port,
        }
        if self.version:
            result["version"] = self.version
        if self.active_version:
            result["active_version"] = self.active_version
        if self.available_versions:
            result["available_versions"] = self.available_versions
        if self.name:
            result["name"] = self.name
        if self.description:
            result["description"] = self.description
        if self.expected_input_json:
            result["expected_input_json"] = self.expected_input_json
        if self.tunnel_url:
            result["tunnel_url"] = self.tunnel_url
        return result


@dataclass
class ContainerStatus:
    """Container runtime status."""

    is_running: bool
    status_text: str

    @property
    def badge_class(self) -> str:
        """CSS class for status badge."""
        return "status-running" if self.is_running else "status-stopped"

    @property
    def indicator_class(self) -> str:
        """CSS class for status indicator."""
        return "running" if self.is_running else "stopped"


@dataclass
class PredictionRequest:
    """Request for model prediction."""

    model_id: str
    input_data: dict

    @classmethod
    def from_json(
        cls, model_id: str, json_str: str
    ) -> tuple[bool, PredictionRequest | str]:
        """Parse JSON prediction request. Returns (success, result)."""
        try:
            import json

            payload = json.loads(json_str)
            return True, cls(model_id=model_id, input_data=payload)
        except Exception as e:
            return False, f"Invalid JSON: {e}"


@dataclass
class PredictionResult:
    """Response from model prediction."""

    model_id: str
    output: dict
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Whether prediction succeeded."""
        return self.error is None
