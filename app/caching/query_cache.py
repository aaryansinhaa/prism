"""In-memory frequent-query cache for model inference responses."""

from __future__ import annotations

import copy
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict

from app.caching.interface import QueryCacheBackend


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _CacheEntry:
    response: Dict[str, Any]
    expires_at: datetime


class InMemoryFrequentQueryCache(QueryCacheBackend):
    """Caches only frequent requests separately for each model key."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_frequency: int = 3,
        ttl_seconds: int = 120,
        max_entries_per_model: int = 256,
    ) -> None:
        self.enabled = enabled
        self.min_frequency = max(1, int(min_frequency))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_entries_per_model = max(1, int(max_entries_per_model))
        self._lock = Lock()
        self._frequencies: dict[str, dict[str, int]] = {}
        self._entries: dict[str, OrderedDict[str, _CacheEntry]] = {}

    @classmethod
    def from_env(cls) -> "InMemoryFrequentQueryCache":
        return cls(
            enabled=_env_bool("PRISM_QUERY_CACHE_ENABLED", True),
            min_frequency=_env_int("PRISM_QUERY_CACHE_MIN_FREQUENCY", 3, 1),
            ttl_seconds=_env_int("PRISM_QUERY_CACHE_TTL_SECONDS", 120, 1),
            max_entries_per_model=_env_int(
                "PRISM_QUERY_CACHE_MAX_ENTRIES_PER_MODEL", 256, 1
            ),
        )

    @staticmethod
    def make_model_key(model_id: str, version: str | None) -> str:
        return f"{model_id}::{version or 'latest'}"

    @staticmethod
    def _payload_key(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def clear(self) -> None:
        with self._lock:
            self._frequencies.clear()
            self._entries.clear()

    def lookup(self, model_key: str, payload: Dict[str, Any]) -> Dict[str, Any] | None:
        if not self.enabled:
            return None

        key = self._payload_key(payload)
        now = _now_utc()
        with self._lock:
            model_freq = self._frequencies.setdefault(model_key, {})
            model_freq[key] = model_freq.get(key, 0) + 1

            model_entries = self._entries.get(model_key)
            if not model_entries:
                return None

            entry = model_entries.get(key)
            if entry is None:
                return None

            if entry.expires_at <= now:
                del model_entries[key]
                return None

            model_entries.move_to_end(key)
            return copy.deepcopy(entry.response)

    def maybe_store(
        self,
        model_key: str,
        payload: Dict[str, Any],
        response: Dict[str, Any],
    ) -> None:
        if not self.enabled:
            return

        key = self._payload_key(payload)
        now = _now_utc()
        with self._lock:
            model_freq = self._frequencies.setdefault(model_key, {})
            frequency = model_freq.get(key, 0)
            if frequency < self.min_frequency:
                return

            model_entries = self._entries.setdefault(model_key, OrderedDict())
            model_entries[key] = _CacheEntry(
                response=copy.deepcopy(response),
                expires_at=now + timedelta(seconds=self.ttl_seconds),
            )
            model_entries.move_to_end(key)

            while len(model_entries) > self.max_entries_per_model:
                model_entries.popitem(last=False)


FrequentQueryCache = InMemoryFrequentQueryCache
