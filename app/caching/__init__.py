"""Caching utilities for Prism."""

from __future__ import annotations

import logging
import os

from app.caching.interface import QueryCacheBackend
from app.caching.query_cache import FrequentQueryCache, InMemoryFrequentQueryCache

logger = logging.getLogger(__name__)


def _build_query_cache_backend() -> QueryCacheBackend:
    backend = os.environ.get("PRISM_QUERY_CACHE_BACKEND", "memory").strip().lower()
    if backend in {"memory", "in-memory", "in_memory"}:
        return InMemoryFrequentQueryCache.from_env()

    logger.warning(
        "Unknown PRISM_QUERY_CACHE_BACKEND=%s, falling back to in-memory cache",
        backend,
    )
    return InMemoryFrequentQueryCache.from_env()


query_cache: QueryCacheBackend = _build_query_cache_backend()
frequent_query_cache = query_cache

__all__ = [
    "QueryCacheBackend",
    "FrequentQueryCache",
    "InMemoryFrequentQueryCache",
    "query_cache",
    "frequent_query_cache",
]
