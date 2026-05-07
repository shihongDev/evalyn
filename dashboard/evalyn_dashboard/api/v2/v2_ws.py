"""``/ws/v2/events`` - cache-bust event stream for the v2 dashboard.

The dashboard's ``/api/v2/*`` routers cache responses keyed on dataset
root mtimes (see :mod:`._shared`). When ``evalyn run-eval`` lands a
fresh run on disk the FE has no way to know the cache went stale - so
we push a tiny ``cache_invalidate`` event over a single shared WS
whenever a watcher loop notices the on-disk signature change.

Wire-up
-------
:func:`register_v2_ws_routes` mounts the route on ``app``; it is called
from :func:`evalyn_dashboard.server.build_app` alongside the jobs WS so
the route lives at ``/ws/v2/events`` (not ``/api/...``).

Watcher lifecycle
-----------------
The polling loop is started on FastAPI's ``startup`` event so it runs
inside the same event loop as the WS handler (required for
``broadcast`` to actually reach subscribers). It is cancelled on
``shutdown``. A single-task guard prevents double-start when ``build_app``
is called more than once in a process (tests).

Event protocol (forward-compatible)
-----------------------------------
* ``{"type": "hello", "v": 1}`` - sent on accept so the client can verify
  it actually opened a working WS.
* ``{"type": "cache_invalidate", "keys": [...]}`` - one or more cache
  key prefixes the FE should mark stale.
* ``{"type": "pong"}`` - reply to a client ``ping`` text frame (kept
  so reconnect-quality probes from the FE don't fail silently).
* ``{"type": "ping", "ts": <unix>}`` - server-pushed heartbeat every
  WS_PING_INTERVAL_S seconds (currently 25s). Defeats NAT idle
  reaping on corporate proxies and cloud LBs. The FE accepts the
  frame in its V2Event union but doesn't react - the frame's job
  is to keep TCP/proxy state warm. Matches the heartbeat used by
  ``api/jobs_ws.py`` and ``api/agent_ws.py`` via the shared
  ``api/_ws_heartbeat.py`` helper.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .._ws_heartbeat import spawn_heartbeat

logger = logging.getLogger(__name__)

# Process-wide set of currently subscribed sockets. Membership is
# managed exclusively by the WS handler below: add on accept, discard
# on disconnect / failed send. We never iterate while mutating.
_subscribers: set[WebSocket] = set()


async def broadcast(event: dict[str, Any]) -> None:
    """Send ``event`` to every subscribed WS, dropping closed sockets.

    Best-effort: a send that raises (peer closed, write failed) is
    logged and the socket is removed from the subscriber set so we
    don't leak handles. The event is serialised once to avoid
    ``json.dumps`` per subscriber.
    """
    if not _subscribers:
        return
    payload = json.dumps(event)
    dead: list[WebSocket] = []
    # Snapshot the set before iterating so a concurrent disconnect that
    # mutates ``_subscribers`` from another task doesn't trip us up.
    for ws in list(_subscribers):
        try:
            await ws.send_text(payload)
        except Exception as exc:  # noqa: BLE001 - all sends are best-effort
            logger.warning("v2 ws broadcast failed: %s", exc)
            dead.append(ws)
    for d in dead:
        _subscribers.discard(d)


def subscriber_count() -> int:
    """Return the current subscriber count (test seam)."""
    return len(_subscribers)


def register_v2_ws_routes(app: FastAPI) -> None:
    """Mount the ``/ws/v2/events`` WebSocket on ``app``."""

    @app.websocket("/ws/v2/events")
    async def v2_events(ws: WebSocket) -> None:
        await ws.accept()
        _subscribers.add(ws)
        # Send-lock matters for the heartbeat: the broadcast() free
        # function and the heartbeat task could otherwise interleave
        # send_text calls and corrupt a frame. The lock here is
        # local to this handler; broadcast() doesn't share it (it's
        # a different code path that snapshots subscribers and
        # accepts torn writes via its except-and-discard pattern).
        # For pings, we use the lock to serialize against the pong
        # response below, the only place we directly send_text from
        # this handler.
        send_lock = asyncio.Lock()
        disconnected = asyncio.Event()
        # Server-side heartbeat. Without this, idle connections
        # (no cache_invalidate events for >60s) get reaped by NATs
        # and corporate proxies. The FE's onclose handler triggers
        # reconnect, but the user pays a 60-120s detection window.
        # 25s heartbeat keeps the link warm. Same shared helper as
        # jobs_ws and agent_ws.
        heartbeat_task = spawn_heartbeat(ws, send_lock, disconnected)
        try:
            # Hello frame: lets the FE confirm the connection actually
            # came up (some proxies drop WS upgrades silently).
            async with send_lock:
                await ws.send_text(json.dumps({"type": "hello", "v": 1}))
            # Drain inbound frames so we notice client disconnects and
            # can answer ``ping`` probes. We don't otherwise interpret
            # what the client sends.
            while True:
                msg = await ws.receive_text()
                if msg == "ping":
                    try:
                        async with send_lock:
                            await ws.send_text(json.dumps({"type": "pong"}))
                    except Exception:  # noqa: BLE001
                        break
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 - log + close cleanly
            logger.warning("v2 ws handler error: %s", exc)
        finally:
            disconnected.set()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            _subscribers.discard(ws)
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass


__all__ = [
    "register_v2_ws_routes",
    "broadcast",
    "subscriber_count",
]
