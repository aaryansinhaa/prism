"""Model detection and loader helpers for PRISM runtime.

This module detects model artifact types and instantiates the
appropriate adapter from `runtime.adapters`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os

from runtime.adapters.base import BaseModel, ModelLoadError


def detect_model_type(model_path: str | Path) -> str | None:
    """Return a short type name for a model file based on its extension.

    Supported types:
    - "onnx" for `.onnx` files
    - "sklearn" for `.pkl` / `.joblib` files
    """
    path = Path(model_path)
    ext = path.suffix.lower()
    if ext == ".onnx":
        return "onnx"
    if ext in {".pkl", ".joblib"}:
        return "sklearn"
    return None


def load_model(model_path: str | Path, **kwargs: Any) -> BaseModel:
    """Load a model artifact and return a framework adapter instance.

    The loader selects an adapter implementation by file extension and
    delegates loading to the adapter's `from_path` classmethod.
    """
    model_type = detect_model_type(model_path)
    if model_type is None:
        raise ModelLoadError(f"Unsupported model artifact: {model_path}")

    if model_type == "onnx":
        # Import lazily to avoid optional dependency requirements at module
        # import time.
        from runtime.adapters.onnx_adapters import ONNXModel

        return ONNXModel.from_path(model_path, **kwargs)

    if model_type == "sklearn":
        from runtime.adapters.scikitlearn_adapters import SklearnModel

        return SklearnModel.from_path(model_path, **kwargs)

    # Fallback — should not be reachable due to early check
    raise ModelLoadError(f"Could not load model for {model_path}")


def default_model_path() -> str:
    """Return a sensible default model path.

    Priority:
    1. `MODEL_PATH` environment variable
    2. `model_store/linear_regression.onnx` inside the repo
    """
    env = os.environ.get("MODEL_PATH")
    if env:
        return env

    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "model_store" / "linear_regression.onnx"
    return str(candidate)


__all__ = ["detect_model_type", "load_model", "default_model_path"]
