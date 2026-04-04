"""Utility functions for Docker and common operations."""

from __future__ import annotations

import subprocess
from typing import Optional

from app.models.model import ContainerStatus


def get_container_status(container_id: str) -> ContainerStatus:
    """Check if Docker container is running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
            capture_output=True,
            text=True,
            timeout=5,
        )
        is_running = result.stdout.strip() == "true"
        status_text = "Running ✓" if is_running else "Stopped ✗"
        return ContainerStatus(is_running=is_running, status_text=status_text)
    except Exception:
        return ContainerStatus(is_running=False, status_text="Unknown")


def get_container_logs(container_id: str, lines: int = 20) -> str:
    """Get last N lines of container logs."""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), container_id],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout if result.stdout else result.stderr or "No logs available"
    except Exception as e:
        return f"Error fetching logs: {str(e)}"


def delete_container(container_id: str) -> tuple[bool, str]:
    """Stop and remove a Docker container. Returns (success, message)."""
    try:
        # Stop the container first
        subprocess.run(
            ["docker", "stop", container_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Remove the container
        result = subprocess.run(
            ["docker", "rm", container_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, "Container deleted successfully"
        else:
            return False, f"Error: {result.stderr}"
    except Exception as e:
        return False, f"Error deleting container: {str(e)}"


def restart_container(container_id: str) -> tuple[bool, str]:
    """Restart a Docker container. Returns (success, message)."""
    try:
        subprocess.run(
            ["docker", "start", container_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Verify status
        status = get_container_status(container_id)
        if status.is_running:
            return True, "Container restarted successfully"
        else:
            return False, "Container restart initiated but status unclear. Please check Docker logs."
    except Exception as e:
        return False, f"Error restarting container: {str(e)}"


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")


def build_prediction_url(model_id: str, base_url: str = "http://127.0.0.1:8000") -> str:
    """Build prediction UI URL."""
    return f"{base_url}/predict?model_id={model_id}"


def build_api_url(model_id: str, base_url: str = "http://127.0.0.1:8000") -> str:
    """Build prediction API endpoint URL."""
    return f"{base_url}/models/{model_id}/predict"
