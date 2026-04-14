"""Model management and inference."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.models.model import (
    ContainerStatus,
    ModelMetadata,
    PredictionRequest,
    PredictionResult,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONTAINER_TEMPLATE_DIR = REPO_ROOT / "model_container"
ALLOWED_UPLOAD_SUFFIXES = {".onnx", ".pkl", ".pickle", ".joblib"}


def upload_root() -> Path:
    """Get configured upload root or default."""
    configured = os.environ.get("MODEL_UPLOAD_ROOT")
    if configured:
        return Path(configured)
    return REPO_ROOT / "model_store" / "uploads"


def save_uploaded_model(upload: UploadFile, model_dir: Path) -> Path:
    """Save uploaded file to model directory."""
    filename = Path(upload.filename or "model.bin").name
    target_path = model_dir / filename
    with target_path.open("wb") as target_file:
        shutil.copyfileobj(upload.file, target_file)
    return target_path


def prepare_model_build_context(model_file_path: Path, model_dir: Path) -> Path:
    """Copy template files and generate Dockerfile for model."""
    for template_file in ("runtime.py", "requirements.txt", "entrypoint.sh"):
        source_path = MODEL_CONTAINER_TEMPLATE_DIR / template_file
        destination_path = model_dir / template_file
        if not source_path.exists():
            raise RuntimeError(f"Missing container template file: {source_path}")
        shutil.copy2(source_path, destination_path)

    dockerfile_path = model_dir / "Dockerfile"
    dockerfile_contents = (
        "FROM python:3.13-slim\n"
        "WORKDIR /app\n"
        "COPY requirements.txt /app/requirements.txt\n"
        "RUN pip install --no-cache-dir -r /app/requirements.txt\n"
        "COPY runtime.py /app/runtime.py\n"
        "COPY entrypoint.sh /app/entrypoint.sh\n"
        f"COPY {model_file_path.name} /models/{model_file_path.name}\n"
        "RUN chmod +x /app/entrypoint.sh\n"
        f"ENV MODEL_PATH=/models/{model_file_path.name}\n"
        "ENV PORT=8000\n"
        "EXPOSE 8000\n"
        'ENTRYPOINT ["/app/entrypoint.sh"]\n'
    )
    dockerfile_path.write_text(dockerfile_contents, encoding="utf-8")
    return dockerfile_path


def validate_upload_extension(file_name: str) -> None:
    """Validate that uploaded file has supported extension."""
    from fastapi import HTTPException
    from starlette import status

    suffix = Path(file_name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model format: {suffix}",
        )


async def ingest_upload_and_build(file: UploadFile) -> dict[str, Any]:
    """Handle model upload and Docker image build."""
    import asyncio

    from app.core import build_model_image

    model_id = uuid.uuid4().hex[:12]
    model_dir = upload_root() / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_path = save_uploaded_model(file, model_dir)
        dockerfile_path = prepare_model_build_context(model_path, model_dir)
        image_tag, build_output = await asyncio.to_thread(
            build_model_image,
            model_id,
            model_dir,
        )
    except RuntimeError as exc:
        from fastapi import HTTPException
        from starlette import status

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    finally:
        await file.close()

    return {
        "model_id": model_id,
        "image_tag": image_tag,
        "model_path": str(model_path),
        "dockerfile_path": str(dockerfile_path),
        "build_context": str(model_dir),
        "build_output": build_output,
    }


__all__ = [
    "ModelMetadata",
    "ContainerStatus",
    "PredictionRequest",
    "PredictionResult",
    "upload_root",
    "save_uploaded_model",
    "prepare_model_build_context",
    "validate_upload_extension",
    "ingest_upload_and_build",
]
