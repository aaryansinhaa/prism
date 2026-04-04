from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "app" / "registry" / "containers.json"


def registry_path() -> Path:
    configured = os.environ.get("MODEL_CONTAINER_REGISTRY_PATH")
    if configured:
        return Path(configured)
    return DEFAULT_REGISTRY_PATH


def _read_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"models": {}}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        return {"models": {}}

    models = data.get("models")
    if not isinstance(models, dict):
        data["models"] = {}
    return data


def register_container(
    model_id: str,
    container_id: str,
    port: int,
    tunnel_url: str | None = None,
) -> Dict[str, Any]:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = _read_registry(path)
    record = {
        "model_id": model_id,
        "container_id": container_id,
        "port": int(port),
    }
    if tunnel_url:
        record["tunnel_url"] = tunnel_url
    
    data["models"][model_id] = record

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)

    return record
