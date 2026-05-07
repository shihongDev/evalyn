"""Tests for ``/ws/v2/events`` (v2 cache-bust event stream).

Mirrors the conventions in ``test_ws_jobs.py``: builds the app inside
``TestClient`` so startup hooks run and the watcher task is scheduled
on the same loop the WS handler uses. Broadcasts are driven through
the client's anyio portal so they target that same loop.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from evalyn_dashboard.api.v2 import _shared as _v2_shared
from evalyn_dashboard.api.v2 import v2_ws as _v2_ws
from evalyn_dashboard.server import build_app

from ._v2_helpers import make_empty_workspace


def test_ws_sends_hello_on_connect(tmp_path, monkeypatch):
    """First frame is the hello envelope so the FE can verify the upgrade."""
    make_empty_workspace(tmp_path, monkeypatch)
    app = build_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/v2/events") as ws:
            evt = json.loads(ws.receive_text())
    assert evt == {"type": "hello", "v": 1}


def test_ws_receives_broadcast(tmp_path, monkeypatch):
    """A broadcast scheduled on the loop reaches every subscriber."""
    make_empty_workspace(tmp_path, monkeypatch)
    app = build_app()
    payload = {"type": "cache_invalidate", "keys": ["home", "experiments"]}

    with TestClient(app) as client:
        with client.websocket_connect("/ws/v2/events") as ws:
            # Drain hello first so the next read is the broadcast.
            hello = json.loads(ws.receive_text())
            assert hello["type"] == "hello"

            # Drive broadcast on the same loop the WS handler runs on.
            client.portal.call(_v2_ws.broadcast, payload)

            evt = json.loads(ws.receive_text())

    assert evt == payload


def test_ws_pong_on_ping(tmp_path, monkeypatch):
    """Client ``ping`` text frame round-trips as ``pong``."""
    make_empty_workspace(tmp_path, monkeypatch)
    app = build_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/v2/events") as ws:
            assert json.loads(ws.receive_text())["type"] == "hello"
            ws.send_text("ping")
            evt = json.loads(ws.receive_text())
    assert evt == {"type": "pong"}


def test_ws_server_heartbeat_fires_on_idle(tmp_path, monkeypatch):
    """v2 events WS sends server-side ping frames every
    WS_PING_INTERVAL_S so idle connections don't get reaped by
    NATs (corporate proxies are 60s; cloud LBs 60-120s). Mirror
    the jobs_ws + agent_ws idle-survival behavior. Patch the
    interval to milliseconds and assert at least one ping frame
    arrives within the test timeout."""
    import evalyn_dashboard.api._ws_heartbeat as _hb

    make_empty_workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(_hb, "WS_PING_INTERVAL_S", 0.05)

    app = build_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/v2/events") as ws:
            assert json.loads(ws.receive_text())["type"] == "hello"
            # Wait for a ping. The first frame after hello should
            # be a server heartbeat ping (no client activity, no
            # broadcasts firing).
            pings = 0
            for _ in range(10):
                evt = json.loads(ws.receive_text())
                if evt.get("type") == "ping":
                    pings += 1
                    # Wire spec contract: ping carries a "ts"
                    # wall-clock field for client-side latency
                    # observability (matches jobs_ws / agent_ws
                    # behavior; pinned across all three handlers).
                    assert isinstance(evt.get("ts"), (int, float))
                    break
            assert pings >= 1, "expected at least one ping frame on idle v2 WS"


def test_ws_disconnect_cleans_subscriber_set(tmp_path, monkeypatch):
    """Closing a WS removes it from the broadcast set."""
    make_empty_workspace(tmp_path, monkeypatch)
    app = build_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/v2/events") as ws:
            assert json.loads(ws.receive_text())["type"] == "hello"
            assert _v2_ws.subscriber_count() >= 1
        # Force the server-side handler to notice the close by issuing a
        # broadcast; the failed send drops the dead socket.
        client.portal.call(
            _v2_ws.broadcast, {"type": "cache_invalidate", "keys": []}
        )
    assert _v2_ws.subscriber_count() == 0


def test_ws_broadcast_with_no_subscribers_is_noop(tmp_path, monkeypatch):
    """``broadcast`` is safe to call when nobody is connected."""
    make_empty_workspace(tmp_path, monkeypatch)
    app = build_app()
    with TestClient(app) as client:
        # No WS open here.
        client.portal.call(
            _v2_ws.broadcast, {"type": "cache_invalidate", "keys": ["home"]}
        )
    # Reaching here without raising is the assertion.


def test_watcher_fires_on_root_signature_change(tmp_path, monkeypatch):
    """When ``roots_signature`` changes the watcher emits cache_invalidate.

    Drives the watcher synchronously by stubbing ``roots_signature`` and
    invoking ``_watcher_loop`` for one iteration via a tight interval.
    """
    make_empty_workspace(tmp_path, monkeypatch)
    # Speed up the watcher so the test doesn't sit on the default 3s.
    monkeypatch.setattr(_v2_shared, "_WATCHER_INTERVAL_S", 0.05)

    sigs = iter([
        (("/tmp/root", 1.0),),
        (("/tmp/root", 2.0),),  # change -> should fire
        (("/tmp/root", 2.0),),  # stable
    ])

    def fake_sig():
        try:
            return next(sigs)
        except StopIteration:
            return (("/tmp/root", 2.0),)

    monkeypatch.setattr(_v2_shared, "roots_signature", fake_sig)

    app = build_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/v2/events") as ws:
            assert json.loads(ws.receive_text())["type"] == "hello"
            # Wait for the watcher to tick at least twice (seed + change).
            evt = json.loads(ws.receive_text())
    assert evt["type"] == "cache_invalidate"
    assert "home" in evt["keys"]
    assert "experiments" in evt["keys"]


def test_unknown_v2_ws_path_404s(tmp_path, monkeypatch):
    """Sanity: a sibling path under /ws/v2/* without a route refuses upgrade."""
    make_empty_workspace(tmp_path, monkeypatch)
    app = build_app()
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/v2/does-not-exist") as ws:
                ws.receive_text()
