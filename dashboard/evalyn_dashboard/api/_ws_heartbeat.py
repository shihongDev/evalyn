"""Shared application-layer ping heartbeat for the dashboard's WebSocket
endpoints.

Long-idle WS connections get reaped by NATs / proxies (60-120s on most
cloud LBs, as low as 60s on corporate proxies). Sending a periodic
``{"type":"ping"}`` text frame keeps the link warm without requiring
the FE to support protocol-level pings (which the browser ``WebSocket``
API doesn't expose anyway).

Used by ``api/jobs_ws.py``, ``api/agent_ws.py``, and ``api/v2/v2_ws.py``
so all three streaming surfaces have the same idle-survival behavior.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# How often to send the ping. 25s comfortably under the typical 60s
# proxy idle timeout without being noisy on otherwise-quiet links.
# Exposed at module scope so tests can patch it.
WS_PING_INTERVAL_S = 25.0


async def heartbeat_loop(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    disconnected: asyncio.Event,
) -> None:
    """Send a ping frame every :data:`WS_PING_INTERVAL_S` seconds until
    the peer disconnects.

    ``send_lock`` MUST be the same lock the event-fanout loop uses so
    a ping never interleaves into a partial event frame. ``disconnected``
    is the shared "client gone" signal - we check it both before sleep
    and after wake to exit promptly when the reader notices a closed
    peer.

    Frame shape: ``{"type":"ping","ts":<unix>}``. The ``ts`` field is
    wall-clock seconds; clients can use it for one-way latency
    observability (we don't expect pong, but a future tick could).

    Suppresses ``CancelledError`` so the standard cancel-and-await
    pattern in the WS handler's ``finally`` block doesn't surface a
    spurious traceback.
    """
    try:
        while not disconnected.is_set():
            await asyncio.sleep(WS_PING_INTERVAL_S)
            if disconnected.is_set():
                return
            async with send_lock:
                try:
                    await websocket.send_text(
                        json.dumps({"type": "ping", "ts": time.time()})
                    )
                except Exception:  # noqa: BLE001 - peer gone
                    disconnected.set()
                    return
    except asyncio.CancelledError:
        return


def spawn_heartbeat(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    disconnected: asyncio.Event,
) -> asyncio.Task[None]:
    """Convenience: ``asyncio.create_task(heartbeat_loop(...))``.

    Returns the task so callers can cancel + await it in the WS
    handler's ``finally`` block alongside other cleanup tasks.
    """
    return asyncio.create_task(heartbeat_loop(websocket, send_lock, disconnected))


# Re-export for tests that want to patch the interval.
__all__ = ["WS_PING_INTERVAL_S", "heartbeat_loop", "spawn_heartbeat"]


# Type alias used by the WS handlers when they want to be explicit
# about the contract they're signing for. Imported but unused at
# runtime is fine.
_HeartbeatFactory = Callable[
    [WebSocket, asyncio.Lock, asyncio.Event],
    Awaitable[None],
]
