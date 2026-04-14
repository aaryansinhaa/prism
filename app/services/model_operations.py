"""Shared helpers for model upload, build, and run operations."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

DOCKER_DAEMON_HINT = (
    "Docker is not available or the daemon is not running. "
    "Start Docker Desktop / the Docker service and make sure the socket exists at /var/run/docker.sock."
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONTAINER_TEMPLATE_DIR = REPO_ROOT / "model_container"
ALLOWED_UPLOAD_SUFFIXES = {".onnx", ".pkl", ".pickle", ".joblib"}
DOCKER_BUILD_TIMEOUT_SECONDS = int(os.environ.get("DOCKER_BUILD_TIMEOUT_SECONDS", "300"))
DOCKER_RUN_TIMEOUT_SECONDS = int(os.environ.get("DOCKER_RUN_TIMEOUT_SECONDS", "60"))


def _normalize_docker_error(action: str, exc: subprocess.CalledProcessError) -> RuntimeError:
    stderr = (exc.stderr or "").strip()
    stdout = (exc.stdout or "").strip()
    details = stderr or stdout or f"docker {action} failed"
    lower_details = details.lower()

    if "failed to connect to the docker api" in lower_details or "docker.sock" in lower_details or "cannot connect to the docker daemon" in lower_details:
        details = f"{details}\n\n{DOCKER_DAEMON_HINT}"

    return RuntimeError(details)


def upload_root() -> Path:
    configured = os.environ.get("MODEL_UPLOAD_ROOT")
    if configured:
        return Path(configured)
    return REPO_ROOT / "model_store" / "uploads"


def save_uploaded_model(upload: UploadFile, model_dir: Path) -> Path:
    filename = Path(upload.filename or "model.bin").name
    target_path = model_dir / filename
    with target_path.open("wb") as target_file:
        shutil.copyfileobj(upload.file, target_file)
    return target_path


def prepare_model_build_context(model_file_path: Path, model_dir: Path) -> Path:
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
        "ENTRYPOINT [\"/app/entrypoint.sh\"]\n"
    )
    dockerfile_path.write_text(dockerfile_contents, encoding="utf-8")
    return dockerfile_path


def build_model_image(model_id: str, model_dir: Path) -> tuple[str, str]:
    image_tag = f"prism_model_{model_id}"
    command = ["docker", "build", "-t", image_tag, "."]
    try:
        result = subprocess.run(
            command,
            cwd=str(model_dir),
            check=True,
            capture_output=True,
            text=True,
            timeout=DOCKER_BUILD_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Docker CLI not found. Please install Docker.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"docker build timed out after {DOCKER_BUILD_TIMEOUT_SECONDS}s"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise _normalize_docker_error("build", exc) from exc

    return image_tag, (result.stdout or "").strip()


def allocate_host_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_model_container(
    model_id: str,
    image_tag: str,
    host_port: int | None = None,
) -> tuple[str, int, str]:
    resolved_port = host_port or allocate_host_port()
    container_name = f"prism_model_{model_id}"
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        f"{resolved_port}:8000",
        image_tag,
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=DOCKER_RUN_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Docker CLI not found. Please install Docker.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"docker run timed out after {DOCKER_RUN_TIMEOUT_SECONDS}s"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise _normalize_docker_error("run", exc) from exc

    container_id = (result.stdout or "").strip()
    return container_name, resolved_port, container_id


def validate_upload_extension(file_name: str) -> None:
    from fastapi import HTTPException
    from starlette import status

    suffix = Path(file_name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model format: {suffix}",
        )


async def ingest_upload_and_build(file: UploadFile) -> dict[str, Any]:
    import asyncio

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
