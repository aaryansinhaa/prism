"""Public model access control utilities.

Provides API-key validation, lightweight in-memory rate limiting,
and structured access logging for public model inference.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
import time
from collections import deque
from typing import Deque, Dict, Tuple

from fastapi import HTTPException, Request
from starlette import status

logger = logging.getLogger("prism.access")


_rate_limit_lock = threading.Lock()
_rate_limit_buckets: Dict[str, Deque[float]] = {}


def _configured_api_keys() -> list[str]:
    raw = os.environ.get("PRISM_API_KEYS", "")
    return [key.strip() for key in raw.split(",") if key.strip()]


def _extract_api_key(request: Request) -> str | None:
    header_key = request.headers.get("x-api-key")
    if header_key:
        return header_key.strip()

    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            return token

    return None


def _safe_api_key_fingerprint(api_key: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return digest[:12]


def validate_api_key(request: Request) -> str:
    """Validate API key from request and return principal identifier.

    Behavior:
    - If `PRISM_API_KEYS` is configured, request must include a valid key.
    - If not configured, access is allowed in open mode and principal is client IP.
    """
    configured_keys = _configured_api_keys()
    provided_key = _extract_api_key(request)

    if not configured_keys:
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key or Authorization: Bearer <key>",
        )

    is_valid = any(hmac.compare_digest(provided_key, key) for key in configured_keys)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return f"key:{_safe_api_key_fingerprint(provided_key)}"


def _rate_limit_config() -> Tuple[int, int]:
    max_requests = int(os.environ.get("PRISM_RATE_LIMIT_REQUESTS", "120"))
    window_seconds = int(os.environ.get("PRISM_RATE_LIMIT_WINDOW_SECONDS", "60"))
    return max_requests, window_seconds


def enforce_rate_limit(principal: str) -> None:
    """Enforce in-memory sliding window rate limit for a principal.

    Raises HTTPException(429) if limit is exceeded.
    """
    max_requests, window_seconds = _rate_limit_config()
    if max_requests <= 0 or window_seconds <= 0:
        return

    now = time.time()
    cutoff = now - window_seconds

    with _rate_limit_lock:
        bucket = _rate_limit_buckets.setdefault(principal, deque())

        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= max_requests:
            retry_after = int(max(1, window_seconds - (now - bucket[0])))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)


def log_access(
    *,
    model_id: str,
    principal: str,
    client_ip: str,
    status_code: int,
    latency_ms: float,
) -> None:
    """Emit structured access log for inference requests."""
    logger.info(
        "public_predict model_id=%s principal=%s client_ip=%s status=%s latency_ms=%.2f",
        model_id,
        principal,
        client_ip,
        status_code,
        latency_ms,
    )
