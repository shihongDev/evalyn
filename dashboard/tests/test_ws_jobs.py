"""Tests for ``/ws/jobs/{job_id}`` WebSocket (Lane B1.5).

Uses ``TestClient.websocket_connect`` and the client's anyio portal to
keep all subprocess transport on the same event loop the WebSocket route
will run on. ``asyncio.run`` for spawning would tear down the loop the
subprocess handle was bound to and any subsequent WS read would hang.
"""

from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from evalyn_dashboard.server import build_app


def _spawn(client: TestClient, app, cmd) -> str:
    return client.portal.call(app.state.job_manager.spawn, cmd)


def _wait(client: TestClient, app, job_id: str, timeout: float = 5.0) -> None:
    client.portal.call(app.state.job_manager.wait, job_id, timeout)


def _drain_until_exit(ws) -> list[dict]:
    """Collect frames from ``ws`` until an ``exit`` event arrives."""
    received: list[dict] = []
    while True:
        text = ws.receive_text()
        evt = json.loads(text)
        received.append(evt)
        if evt["type"] == "exit":
            break
    return received


def test_ws_streams_events():
    app = build_app()
    with TestClient(app) as client:
        job_id = _spawn(client, app, [sys.executable, "-c", "print('hello')"])

        with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
            events = _drain_until_exit(ws)

    types = [e["type"] for e in events]
    assert "stdout" in types
    assert events[-1]["type"] == "exit"
    assert events[-1]["code"] == 0
    assert any("hello" in e.get("line", "") for e in events if e["type"] == "stdout")


def test_ws_unknown_job_closes():
    app = build_app()
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/jobs/nope") as ws:
                ws.receive_text()


def test_ws_replays_history_for_completed_job():
    """Subscribing after a job finishes still receives the full event log."""
    app = build_app()
    with TestClient(app) as client:
        job_id = _spawn(client, app, [sys.executable, "-c", "print('replay')"])
        _wait(client, app, job_id)

        with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
            events = _drain_until_exit(ws)

    assert any("replay" in e.get("line", "") for e in events if e["type"] == "stdout")
    assert events[-1]["type"] == "exit"


def test_ws_since_replays_only_new_events():
    """``?since=N`` skips events with ``event_id <= N``."""
    app = build_app()
    with TestClient(app) as client:
        job_id = _spawn(
            client, app, [sys.executable, "-c", "print('one'); print('two')"]
        )
        _wait(client, app, job_id)

        # First subscribe to discover the event_id of the 'one' line.
        with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
            full = _drain_until_exit(ws)

        one_evt = next(
            e for e in full if e["type"] == "stdout" and "one" in e.get("line", "")
        )
        cursor = one_evt["event_id"]

        with client.websocket_connect(f"/ws/jobs/{job_id}?since={cursor}") as ws:
            replay = _drain_until_exit(ws)

    # Every replayed event has id strictly greater than cursor.
    assert all(e["event_id"] > cursor for e in replay)
    # 'one' is excluded; 'two' is present.
    assert not any("one" in e.get("line", "") for e in replay)
    assert any("two" in e.get("line", "") for e in replay)


def test_ws_since_with_invalid_value_falls_back_to_full_replay():
    """A non-integer ``since`` is silently ignored (no replay clipping)."""
    app = build_app()
    with TestClient(app) as client:
        job_id = _spawn(client, app, [sys.executable, "-c", "print('full')"])
        _wait(client, app, job_id)

        with client.websocket_connect(f"/ws/jobs/{job_id}?since=not-a-number") as ws:
            events = _drain_until_exit(ws)

    assert any("full" in e.get("line", "") for e in events if e["type"] == "stdout")
