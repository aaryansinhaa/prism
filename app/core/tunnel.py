"""Reverse tunnel management for exposing model containers via public URLs.

The ngrok connection itself is started in a detached worker process so app
reloads do not tear down the tunnel.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = Path(tempfile.gettempdir()) / "prism" / "tunnels"
START_TIMEOUT_SECONDS = int(os.environ.get("PRISM_TUNNEL_START_TIMEOUT", "30"))


def _state_dir() -> Path:
    configured = os.environ.get("PRISM_TUNNEL_STATE_DIR")
    if configured:
        return Path(configured)
    return DEFAULT_STATE_DIR


def _safe_model_id(model_id: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in model_id
    )


def _state_path(model_id: str) -> Path:
    return _state_dir() / f"{_safe_model_id(model_id)}.json"


def _log_path(model_id: str) -> Path:
    return _state_dir() / f"{_safe_model_id(model_id)}.log"


def _read_state_file(model_id: str) -> Dict[str, Any] | None:
    path = _state_path(model_id)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return None

    return data if isinstance(data, dict) else None


def _write_state_file(model_id: str, payload: Dict[str, Any]) -> None:
    path = _state_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    temp_path.replace(path)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _cleanup_state(model_id: str) -> None:
    state_path = _state_path(model_id)
    try:
        state_path.unlink()
    except FileNotFoundError:
        pass


async def start_tunnel(local_port: int, model_id: str) -> tuple[str, None]:
    """Start an ngrok reverse tunnel in a detached worker process."""
    existing = _read_state_file(model_id)
    if existing:
        existing_pid = int(existing.get("pid", 0) or 0)
        existing_port = int(existing.get("local_port", 0) or 0)
        existing_url = existing.get("public_url")
        if (
            existing_pid
            and _process_alive(existing_pid)
            and existing_port == local_port
            and isinstance(existing_url, str)
        ):
            print(
                f"[Tunnel] Reusing existing detached tunnel for {model_id}: {existing_url}"
            )
            return existing_url, None

        _cleanup_state(model_id)

    _state_dir().mkdir(parents=True, exist_ok=True)
    log_path = _log_path(model_id)
    command = [
        sys.executable,
        "-m",
        "app.core.tunnel_worker",
        "--local-port",
        str(local_port),
        "--model-id",
        model_id,
    ]

    print(f"[Tunnel] Starting detached worker for http://127.0.0.1:{local_port}...")
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(  # noqa: S603,S607
            command,
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

    deadline = time.monotonic() + START_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        state = _read_state_file(model_id)
        if state:
            public_url = state.get("public_url")
            if isinstance(public_url, str) and public_url:
                print(f"[Tunnel] Detached worker reported public_url: {public_url}")
                print(
                    f"✓ [Tunnel] Successfully created tunnel: http://127.0.0.1:{local_port} -> {public_url}"
                )
                return public_url, None

        if process.poll() is not None:
            break

        await asyncio.sleep(0.5)

    exit_code = process.poll()
    log_excerpt = ""
    try:
        with log_path.open("r", encoding="utf-8") as file:
            log_excerpt = file.read().strip()
    except Exception:
        log_excerpt = ""

    if exit_code is None:
        try:
            process.terminate()
        except Exception:
            pass
        raise RuntimeError(
            f"Timed out waiting for detached tunnel worker to report a public URL after {START_TIMEOUT_SECONDS}s"
        )

    raise RuntimeError(
        "Detached tunnel worker exited before reporting a public URL. "
        f"exit_code={exit_code}. Logs: {log_excerpt or 'no worker logs available'}"
    )


async def stop_tunnel(tunnel_url: str) -> bool:
    """Stop a running detached ngrok tunnel by public URL."""
    state_dir = _state_dir()
    if not state_dir.exists():
        return False

    for state_file in state_dir.glob("*.json"):
        try:
            with state_file.open("r", encoding="utf-8") as file:
                record = json.load(file)
        except Exception:
            continue

        if not isinstance(record, dict) or record.get("public_url") != tunnel_url:
            continue

        pid = int(record.get("pid", 0) or 0)
        model_id = str(record.get("model_id", ""))
        if pid and _process_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

            for _ in range(20):
                if not _process_alive(pid):
                    break
                await asyncio.sleep(0.25)

        if model_id:
            _cleanup_state(model_id)
        else:
            try:
                state_file.unlink()
            except FileNotFoundError:
                pass

        return True

    return False


def get_tunnel_status(model_id: str) -> Dict[str, Any]:
    """Get the status of a detached tunnel worker."""
    record = _read_state_file(model_id)
    if not record:
        return {"status": "not_found", "url": None}

    pid = int(record.get("pid", 0) or 0)
    public_url = record.get("public_url")
    if pid and _process_alive(pid) and isinstance(public_url, str):
        return {"status": "running", "url": public_url}

    _cleanup_state(model_id)
    return {"status": "stopped", "url": None}
