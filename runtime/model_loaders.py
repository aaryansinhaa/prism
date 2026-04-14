"""Model detection and loader helpers for PRISM runtime.

This module detects model artifact types and instantiates the
appropriate adapter from `runtime.adapters`.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from runtime.adapters.base import BaseModel, ModelLoadError

_MODEL_TYPE_BY_SUFFIX = {
    ".onnx": "onnx",
    ".pkl": "sklearn",
    ".pickle": "sklearn",
    ".joblib": "sklearn",
}


def detect_model_type(model_path: str | Path) -> str | None:
    """Return a short type name for a model file based on its extension.

    Supported types:
    - "onnx" for `.onnx` files
    - "sklearn" for `.pkl` / `.joblib` files
    """
    path = Path(model_path)
    if path.is_dir():
        return None

    ext = path.suffix.lower()
    if ext in _MODEL_TYPE_BY_SUFFIX:
        return _MODEL_TYPE_BY_SUFFIX[ext]
    return None


def load_model(model_path: str | Path, **kwargs: Any) -> BaseModel:
    """Load a model artifact and return a framework adapter instance.

    The loader selects an adapter implementation by file extension and
    delegates loading to the adapter's `from_path` classmethod.
    """
    model_type = detect_model_type(model_path)
    if model_type is None:
        raise ModelLoadError(f"Unsupported model artifact: {model_path}")

    loaders = {
        "onnx": "runtime.adapters.onnx_adapters:ONNXModel",
        "sklearn": "runtime.adapters.scikitlearn_adapters:SklearnModel",
    }
    module_name, class_name = loaders[model_type].split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    return adapter_cls.from_path(model_path, **kwargs)


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
