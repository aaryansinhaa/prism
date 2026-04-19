"""Request batching for model inference forwarding."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict

import httpx


@dataclass
class _QueuedRequest:
    payload: Dict[str, Any]
    future: asyncio.Future[Dict[str, Any]]


class RequestBatcher:
    """Collects compatible requests for a short window and forwards one batch."""

    def __init__(
        self,
        batch_window_ms: int = 50,
        forwarder: (
            Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]] | None
        ) = None,
    ) -> None:
        self._batch_window_seconds = max(0.0, batch_window_ms / 1000.0)
        self._forwarder = forwarder or self._default_forwarder
        self._state_lock = asyncio.Lock()
        self._pending: dict[str, list[_QueuedRequest]] = {}
        self._flush_tasks: dict[str, asyncio.Task[None]] = {}

    @staticmethod
    async def _default_forwarder(
        container_url: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(container_url, json=payload)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _extract_rows(payload: Dict[str, Any]) -> tuple[list[list[float | int]], int]:
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")

        input_value = payload.get("input")
        if not isinstance(input_value, list) or not input_value:
            raise ValueError("payload.input must be a non-empty array")

        first_item = input_value[0]
        if isinstance(first_item, list):
            width = len(first_item)
            if width == 0:
                raise ValueError("payload.input rows must be non-empty")

            rows: list[list[float | int]] = []
            for row in input_value:
                if not isinstance(row, list) or len(row) != width:
                    raise ValueError("payload.input must have consistent row shape")
                for element in row:
                    if not isinstance(element, (int, float)):
                        raise ValueError("payload.input values must be numeric")
                rows.append(row)
            return rows, width

        rows = []
        for element in input_value:
            if not isinstance(element, (int, float)):
                raise ValueError("payload.input values must be numeric")
            rows.append([element])
        return rows, 1

    async def _submit_for_batch(
        self,
        key: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Dict[str, Any]] = loop.create_future()

        async with self._state_lock:
            queue = self._pending.setdefault(key, [])
            queue.append(_QueuedRequest(payload=payload, future=future))

            task = self._flush_tasks.get(key)
            if task is None or task.done():
                self._flush_tasks[key] = asyncio.create_task(
                    self._flush_after_window(key)
                )

        return await future

    async def _flush_after_window(self, key: str) -> None:
        await asyncio.sleep(self._batch_window_seconds)

        async with self._state_lock:
            batch = self._pending.pop(key, [])
            self._flush_tasks.pop(key, None)

        if not batch:
            return

        await self._dispatch_batch(key, batch)

    async def _dispatch_batch(
        self, container_url: str, batch: list[_QueuedRequest]
    ) -> None:
        try:
            batched_rows: list[list[float | int]] = []
            row_counts: list[int] = []
            expected_width: int | None = None

            for item in batch:
                rows, width = self._extract_rows(item.payload)
                if expected_width is None:
                    expected_width = width
                elif expected_width != width:
                    raise ValueError(
                        "Cannot batch requests with mismatched input width"
                    )

                batched_rows.extend(rows)
                row_counts.append(len(rows))

            response_json = await self._forwarder(
                container_url, {"input": batched_rows}
            )
            if not isinstance(response_json, dict):
                raise ValueError("Container response must be a JSON object")

            predictions = response_json.get("predictions")
            if not isinstance(predictions, list):
                raise ValueError("Container response missing list 'predictions'")

            if len(predictions) != len(batched_rows):
                raise ValueError(
                    "Container response prediction count does not match batched input size"
                )

            index = 0
            for item, count in zip(batch, row_counts):
                split_response = dict(response_json)
                split_response["predictions"] = predictions[index : index + count]
                index += count
                if not item.future.done():
                    item.future.set_result(split_response)
        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)

    async def forward(
        self, container_url: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Forward a prediction payload, batching when payloads are compatible."""
        try:
            self._extract_rows(payload)
        except ValueError:
            return await self._forwarder(container_url, payload)

        return await self._submit_for_batch(container_url, payload)


def _batch_window_from_env(default_ms: int = 50) -> int:
    raw = os.environ.get("PRISM_BATCH_WINDOW_MS")
    if raw is None:
        return default_ms

    try:
        value = int(raw)
    except ValueError:
        return default_ms

    return max(0, value)


request_batcher = RequestBatcher(batch_window_ms=_batch_window_from_env())
