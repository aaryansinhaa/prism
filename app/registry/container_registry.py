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


def _write_registry(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def load_registry() -> Dict[str, Any]:
    """Load the registry file from disk."""
    return _read_registry(registry_path())


def save_registry(data: Dict[str, Any]) -> None:
    """Persist registry data to disk."""
    _write_registry(registry_path(), data)


def _legacy_version_name(model_entry: Dict[str, Any]) -> str:
    version = model_entry.get("version")
    if isinstance(version, str) and version.strip():
        return version
    return "v1"


def _normalize_version_record(
    model_id: str,
    version: str,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = dict(record)
    normalized["model_id"] = model_id
    normalized["version"] = version
    return normalized


def resolve_model_version_entry(
    model_id: str,
    version: str | None = None,
    registry: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], str, str | None]:
    """Resolve a model entry and version from registry data.

    Returns the version-specific record, the resolved version, and the active
    version recorded on the model entry (when available).
    """
    data = registry if registry is not None else load_registry()
    models = data.get("models", {}) if isinstance(data, dict) else {}
    if not isinstance(models, dict) or model_id not in models:
        raise KeyError(model_id)

    model_entry = models[model_id]
    if not isinstance(model_entry, dict):
        raise KeyError(model_id)

    versions = model_entry.get("versions")
    if isinstance(versions, dict) and versions:
        active_version = model_entry.get("active_version")
        resolved_version = version or (
            active_version
            if isinstance(active_version, str) and active_version
            else None
        )
        if resolved_version is None:
            resolved_version = next(iter(versions.keys()))

        version_entry = versions.get(resolved_version)
        if not isinstance(version_entry, dict):
            raise KeyError(f"{model_id}:{resolved_version}")
        return (
            version_entry,
            resolved_version,
            active_version if isinstance(active_version, str) else None,
        )

    resolved_version = _legacy_version_name(model_entry)
    if version is not None and version != resolved_version:
        raise KeyError(f"{model_id}:{version}")

    return model_entry, resolved_version, resolved_version


def _ensure_versioned_model_entry(
    existing: Dict[str, Any],
    model_id: str,
    version: str,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    versions = existing.get("versions")
    if isinstance(versions, dict):
        updated = dict(existing)
        updated_versions = dict(versions)
        updated_versions[version] = record
        updated["versions"] = updated_versions
        updated["active_version"] = version
        updated["model_id"] = model_id
        return updated

    legacy_version = _legacy_version_name(existing)
    if version == legacy_version:
        return record

    return {
        "model_id": model_id,
        "active_version": version,
        "versions": {
            legacy_version: _normalize_version_record(
                model_id,
                legacy_version,
                existing,
            ),
            version: record,
        },
    }


def register_container(
    model_id: str,
    container_id: str,
    port: int,
    version: str | None = None,
    name: str | None = None,
    description: str | None = None,
    expected_input_json: str | None = None,
    tunnel_url: str | None = None,
) -> Dict[str, Any]:
    data = load_registry()
    model_version = version or "v1"
    record = {
        "model_id": model_id,
        "container_id": container_id,
        "port": int(port),
        "version": model_version,
    }
    if name:
        record["name"] = name
    if description:
        record["description"] = description
    if expected_input_json:
        record["expected_input_json"] = expected_input_json
    if tunnel_url:
        record["tunnel_url"] = tunnel_url

    models = data.setdefault("models", {})
    existing = models.get(model_id)
    if version is None and existing is None:
        models[model_id] = record
    elif isinstance(existing, dict):
        models[model_id] = _ensure_versioned_model_entry(
            existing,
            model_id,
            model_version,
            record,
        )
    else:
        models[model_id] = record

    save_registry(data)

    return record


def remove_model_version(model_id: str, version: str | None = None) -> bool:
    """Remove a whole model or a specific version from the registry."""
    data = load_registry()
    models = data.get("models", {})
    if not isinstance(models, dict) or model_id not in models:
        return False

    model_entry = models[model_id]
    if not isinstance(model_entry, dict):
        return False

    if version is None:
        del models[model_id]
        save_registry(data)
        return True

    versions = model_entry.get("versions")
    if isinstance(versions, dict):
        if version not in versions:
            return False
        del versions[version]
        if not versions:
            del models[model_id]
        else:
            if model_entry.get("active_version") == version:
                model_entry["active_version"] = next(iter(versions.keys()))
        save_registry(data)
        return True

    legacy_version = _legacy_version_name(model_entry)
    if version != legacy_version:
        return False

    del models[model_id]
    save_registry(data)
    return True
