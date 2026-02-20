"""Common abstractions for PRISM runtime model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class ModelAdapterError(RuntimeError):
	"""Base exception for adapter-specific failures."""


class ModelLoadError(ModelAdapterError):
	"""Raised when a model artifact cannot be loaded."""


class ModelPredictError(ModelAdapterError):
	"""Raised when prediction fails for a loaded model."""


class BaseModel(ABC):
	"""All model adapters must implement this contract."""

	def __init__(self, metadata: Dict[str, Any] | None = None) -> None:
		self.metadata: Dict[str, Any] = metadata or {}

	@classmethod
	@abstractmethod
	def from_path(
		cls,
		model_path: str | Path,
		**kwargs: Any,
	) -> "BaseModel":
		"""Create a model adapter from an on-disk artifact."""

	@abstractmethod
	def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Run inference and return a JSON-serializable dict."""


__all__ = [
	"BaseModel",
	"ModelAdapterError",
	"ModelLoadError",
	"ModelPredictError",
]
