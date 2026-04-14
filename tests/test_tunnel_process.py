from __future__ import annotations

import asyncio
import json
from pathlib import Path

import app.core.tunnel as tunnel_module


def test_start_tunnel_spawns_detached_worker_and_reads_state(monkeypatch, tmp_path):
    monkeypatch.setenv("PRISM_TUNNEL_STATE_DIR", str(tmp_path))

    created = {}

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self._poll = None

        def poll(self):
            return self._poll

    def fake_popen(command, **kwargs):
        created["command"] = command
        created["kwargs"] = kwargs
        state_file = Path(tmp_path) / "model_123.json"
        state_file.write_text(
            json.dumps(
                {
                    "model_id": "model_123",
                    "local_port": 50737,
                    "pid": 4321,
                    "public_url": "https://example.ngrok-free.dev",
                }
            ),
            encoding="utf-8",
        )
        return FakeProcess(pid=4321)

    monkeypatch.setattr(tunnel_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tunnel_module, "START_TIMEOUT_SECONDS", 1)

    public_url, _ = asyncio.run(tunnel_module.start_tunnel(50737, "model_123"))

    assert public_url == "https://example.ngrok-free.dev"
    assert created["command"][0] == tunnel_module.sys.executable
    assert created["command"][1:3] == ["-m", "app.core.tunnel_worker"]


def test_get_tunnel_status_uses_worker_state(monkeypatch, tmp_path):
    monkeypatch.setenv("PRISM_TUNNEL_STATE_DIR", str(tmp_path))
    state_file = Path(tmp_path) / "model_abc.json"
    state_file.write_text(
        json.dumps(
            {
                "model_id": "model_abc",
                "local_port": 4000,
                "pid": 9999,
                "public_url": "https://abc.ngrok-free.dev",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(tunnel_module, "_process_alive", lambda pid: True)

    status = tunnel_module.get_tunnel_status("model_abc")

    assert status == {"status": "running", "url": "https://abc.ngrok-free.dev"}