"""Core Docker and container operations."""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

DOCKER_BUILD_TIMEOUT_SECONDS = int(os.environ.get("DOCKER_BUILD_TIMEOUT_SECONDS", "300"))
DOCKER_RUN_TIMEOUT_SECONDS = int(os.environ.get("DOCKER_RUN_TIMEOUT_SECONDS", "60"))


def allocate_host_port() -> int:
    """Allocate a free host port using OS socket binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_model_image(model_id: str, model_dir: Path) -> tuple[str, str]:
    """Build Docker image from model build context."""
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
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or "docker build failed"
        raise RuntimeError(details) from exc

    return image_tag, (result.stdout or "").strip()


def run_model_container(
    model_id: str,
    image_tag: str,
    host_port: int | None = None,
) -> tuple[str, int, str]:
    """Run Docker container from built image with dynamic port mapping."""
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
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or "docker run failed"
        raise RuntimeError(details) from exc

    container_id = (result.stdout or "").strip()
    return container_name, resolved_port, container_id
