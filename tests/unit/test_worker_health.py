"""Worker liveness endpoint: the minimal ASGI app Railway healthchecks hit."""

from __future__ import annotations

from typing import Any

import pytest

from services.worker import _health_app, _start_health_server


async def _call(path: str) -> tuple[int, bytes]:
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def receive() -> dict[str, Any]:  # pragma: no cover - unused by app
        return {"type": "http.request"}

    await _health_app({"type": "http", "path": path}, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = next(m["body"] for m in sent if m["type"] == "http.response.body")
    return status, body


@pytest.mark.asyncio
async def test_healthz_returns_alive() -> None:
    status, body = await _call("/healthz")
    assert status == 200
    assert b"alive" in body


@pytest.mark.asyncio
async def test_root_is_also_ok_and_unknown_is_404() -> None:
    assert (await _call("/"))[0] == 200
    assert (await _call("/nope"))[0] == 404


@pytest.mark.asyncio
async def test_non_http_scope_is_ignored() -> None:
    # Lifespan/websocket scopes must be a silent no-op, not an error.
    await _health_app({"type": "lifespan"}, None, None)


@pytest.mark.asyncio
async def test_health_server_noop_without_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PORT", raising=False)
    assert _start_health_server() is None


@pytest.mark.asyncio
async def test_health_server_starts_and_serves_when_port_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import json
    import socket
    import urllib.request

    # Grab a free port, then let the server bind it.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("PORT", str(port))

    def _fetch() -> tuple[int, bytes]:
        # Runs in a thread: a blocking urlopen on the event-loop thread would
        # starve the uvicorn server coroutine and the request could never be served.
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as r:
            return r.status, r.read()

    task = _start_health_server()
    assert task is not None
    try:
        # Wait for the server to accept connections.
        for _ in range(50):
            await asyncio.sleep(0.1)
            try:
                status, body = await asyncio.to_thread(_fetch)
            except Exception:
                continue
            assert status == 200
            assert json.loads(body)["status"] == "alive"
            break
        else:
            pytest.fail("health server never became reachable")
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
