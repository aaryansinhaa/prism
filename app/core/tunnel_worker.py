"""Detached worker process that owns the ngrok tunnel lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
from threading import Event

import ngrok

from app.core.tunnel import _state_path, _write_state_file


def _ensure_ngrok_initialized() -> None:
    token = os.environ.get("NGROK_AUTHTOKEN") or os.environ.get("PYNGROK_AUTHTOKEN")
    if not token:
        raise RuntimeError(
            "Tunnel requested but NGROK_AUTHTOKEN is not set. "
            "Set NGROK_AUTHTOKEN in your environment to enable public tunnel URLs."
        )

    ngrok.set_auth_token(token)  # type: ignore[attr-defined]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRISM ngrok tunnel worker")
    parser.add_argument("--local-port", type=int, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    return parser.parse_args()


def _is_duplicate_endpoint_error(exc: Exception) -> bool:
    return "ERR_NGROK_334" in str(exc)


def _extract_conflicting_endpoint(exc: Exception) -> str | None:
    message = str(exc)
    match = re.search(r"https://[^'\"\s]+", message)
    if not match:
        return None
    return match.group(0)


def _disconnect_endpoint(url: str) -> bool:
    try:
        ngrok.disconnect(url)  # type: ignore[attr-defined]
        print(f"[TunnelWorker] Disconnected conflicting endpoint: {url}", flush=True)
        return True
    except Exception as disconnect_exc:
        print(
            f"[TunnelWorker] Could not disconnect conflicting endpoint {url}: {disconnect_exc}",
            flush=True,
        )
        return False


def _disconnect_existing_listeners() -> int:
    disconnected = 0
    try:
        listeners = ngrok.get_listeners()  # type: ignore[attr-defined]
    except Exception as exc:
        print(f"[TunnelWorker] Failed to list listeners for cleanup: {exc}", flush=True)
        return disconnected

    for existing in listeners:
        try:
            url = existing.url()
            ngrok.disconnect(url)  # type: ignore[attr-defined]
            disconnected += 1
            print(f"[TunnelWorker] Disconnected existing listener: {url}", flush=True)
        except Exception as exc:
            print(f"[TunnelWorker] Failed to disconnect listener {existing}: {exc}", flush=True)

    return disconnected


def _connect_listener(addr: str, custom_domain: str | None):
    if custom_domain:
        print(f"[TunnelWorker] Using custom domain: {custom_domain}", flush=True)
        return ngrok.connect(addr, domain=custom_domain)  # type: ignore[attr-defined]

    print("[TunnelWorker] No custom domain configured, ngrok will auto-generate", flush=True)
    return ngrok.connect(addr)  # type: ignore[attr-defined]


def main() -> int:
    args = _parse_args()
    stop_event = Event()
    listener = None

    def _handle_signal(signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        _ensure_ngrok_initialized()
        addr = f"http://127.0.0.1:{args.local_port}"
        custom_domain = os.environ.get("NGROK_CUSTOM_DOMAIN")

        try:
            listener = _connect_listener(addr, custom_domain)
        except Exception as exc:
            if not _is_duplicate_endpoint_error(exc):
                raise

            conflicting_endpoint = _extract_conflicting_endpoint(exc)
            disconnected_conflict = False
            if conflicting_endpoint:
                disconnected_conflict = _disconnect_endpoint(conflicting_endpoint)

            disconnected = _disconnect_existing_listeners()
            print(
                "[TunnelWorker] Retrying ngrok connect after ERR_NGROK_334 cleanup "
                f"(disconnected_conflict={disconnected_conflict}, disconnected={disconnected})",
                flush=True,
            )
            listener = _connect_listener(addr, custom_domain)

        public_url = listener.url()
        record = {
            "model_id": args.model_id,
            "local_port": args.local_port,
            "pid": os.getpid(),
            "public_url": public_url,
            "created_at": time.time(),
        }
        _write_state_file(args.model_id, record)
        print(json.dumps({"status": "ready", **record}), flush=True)

        while not stop_event.wait(1.0):
            pass

        return 0
    except Exception as exc:
        print(f"[TunnelWorker] ERROR: {exc}", flush=True)
        raise
    finally:
        if listener is not None:
            try:
                ngrok.disconnect(listener.url())  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            _state_path(args.model_id).unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())