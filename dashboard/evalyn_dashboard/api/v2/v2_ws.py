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
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

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
        try:
            # Hello frame: lets the FE confirm the connection actually
            # came up (some proxies drop WS upgrades silently).
            await ws.send_text(json.dumps({"type": "hello", "v": 1}))
            # Drain inbound frames so we notice client disconnects and
            # can answer ``ping`` probes. We don't otherwise interpret
            # what the client sends.
            while True:
                msg = await ws.receive_text()
                if msg == "ping":
                    try:
                        await ws.send_text(json.dumps({"type": "pong"}))
                    except Exception:  # noqa: BLE001
                        break
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 - log + close cleanly
            logger.warning("v2 ws handler error: %s", exc)
        finally:
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
