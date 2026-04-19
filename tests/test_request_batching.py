from __future__ import annotations

import asyncio

from app.batching.request_batcher import RequestBatcher


def test_request_batcher_collects_and_splits_predictions() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    async def fake_forwarder(container_url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((container_url, payload))
        rows = payload["input"]
        return {
            "predictions": [float(index) for index, _ in enumerate(rows, start=1)],
            "model": "batched",
        }

    batcher = RequestBatcher(batch_window_ms=50, forwarder=fake_forwarder)

    async def _run() -> tuple[dict[str, object], dict[str, object]]:
        first = asyncio.create_task(
            batcher.forward("http://127.0.0.1:9999/predict", {"input": [[1.0, 2.0]]})
        )
        await asyncio.sleep(0.01)
        second = asyncio.create_task(
            batcher.forward(
                "http://127.0.0.1:9999/predict",
                {"input": [[3.0, 4.0], [5.0, 6.0]]},
            )
        )
        return await asyncio.gather(first, second)

    first_response, second_response = asyncio.run(_run())

    assert len(calls) == 1
    assert calls[0][0] == "http://127.0.0.1:9999/predict"
    assert calls[0][1] == {"input": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}

    assert first_response["predictions"] == [1.0]
    assert second_response["predictions"] == [2.0, 3.0]
    assert first_response["model"] == "batched"
    assert second_response["model"] == "batched"


def test_request_batcher_passthrough_for_non_batchable_payload() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    async def fake_forwarder(container_url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((container_url, payload))
        return {"ok": True}

    batcher = RequestBatcher(batch_window_ms=50, forwarder=fake_forwarder)

    async def _run() -> tuple[dict[str, object], dict[str, object]]:
        first = asyncio.create_task(
            batcher.forward("http://127.0.0.1:9999/predict", {"x": [1.0]})
        )
        second = asyncio.create_task(
            batcher.forward("http://127.0.0.1:9999/predict", {"x": [2.0]})
        )
        return await asyncio.gather(first, second)

    first_response, second_response = asyncio.run(_run())

    assert first_response == {"ok": True}
    assert second_response == {"ok": True}
    assert len(calls) == 2
    assert calls[0][1] == {"x": [1.0]}
    assert calls[1][1] == {"x": [2.0]}
