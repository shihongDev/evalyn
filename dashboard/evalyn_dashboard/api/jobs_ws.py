"""WebSocket route ``/ws/jobs/{job_id}`` (Lane B1.5).

Subscribes to a job's :class:`JobManager` event stream and pushes events
as JSON text frames. Supports reconnect-with-replay via the
``?since=<event_id>`` query parameter (forwarded to
:meth:`JobManager.subscribe` ``since=`` so older events are replayed
before live fanout begins).

Wire-up: :func:`register_ws_routes` is called from ``server.build_app`` so
the route lives at ``/ws/jobs/{job_id}`` (not ``/api/...``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

# How often to send an application-layer ping. Chosen to be safely
# under the typical NAT / proxy idle timeout (60-120s on most cloud
# providers; AWS NLB defaults to 350s, but corporate proxies can be
# as aggressive as 60s). A 25s cadence keeps the connection warm
# without flooding the wire on otherwise-quiet jobs (e.g. an eval that
# spends a minute in a single LLM call before printing again).
WS_PING_INTERVAL_S = 25.0


def register_ws_routes(app: FastAPI) -> None:
    """Mount the ``/ws/jobs/{job_id}`` WebSocket on ``app``."""

    @app.websocket("/ws/jobs/{job_id}")
    async def jobs_ws(websocket: WebSocket, job_id: str) -> None:
        # Parse ``since`` from query string. Invalid values (non-int) are
        # treated as "no since" rather than rejected so a buggy client
        # always gets at least the live tail.
        since_raw: Optional[str] = websocket.query_params.get("since")
        since: Optional[int] = None
        if since_raw is not None:
            try:
                since = int(since_raw)
            except ValueError:
                since = None

        jm = websocket.app.state.job_manager
        if jm.get(job_id) is None:
            # Accept then close with a 1008 policy violation so the client
            # can read the close reason. Closing pre-accept emits 403 on
            # some browsers without a reason payload.
            await websocket.accept()
            await websocket.close(code=1008, reason="unknown job")
            return

        await websocket.accept()
        send_lock = asyncio.Lock()
        client_disconnected = asyncio.Event()

        async def reader() -> None:
            """Drain inbound frames so we notice client disconnects.

            We don't process inbound messages, but reading them is the
            standard way to observe a closed peer with FastAPI.
            """
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                client_disconnected.set()
            except Exception:
                client_disconnected.set()

        reader_task = asyncio.create_task(reader())

        async def heartbeat() -> None:
            """Send a periodic application-layer ping so idle WS
            connections aren't reaped by NATs / proxies.

            A long-running eval can sit in a single LLM call for
            minutes without producing output; without these pings,
            many proxies (default-60s in some corporate setups)
            silently close the underlying TCP connection and the
            client only notices when the next event would have
            arrived. The ping keeps the link warm.

            Frame shape ``{"type":"ping","ts":<unix>}`` is parsed by
            the FE's JSON path and discarded by the if/else chain
            (no matching branch). No new wire field; no FE change
            required to ship safely.

            send_lock shared with the event-fanout loop so we never
            interleave a ping into the middle of a large event frame.
            """
            try:
                while not client_disconnected.is_set():
                    await asyncio.sleep(WS_PING_INTERVAL_S)
                    if client_disconnected.is_set():
                        return
                    async with send_lock:
                        try:
                            await websocket.send_text(
                                json.dumps({"type": "ping", "ts": time.time()})
                            )
                        except Exception:  # noqa: BLE001 - peer gone
                            client_disconnected.set()
                            return
            except asyncio.CancelledError:
                return

        heartbeat_task = asyncio.create_task(heartbeat())

        try:
            async with jm.subscribe(job_id, since=since) as stream:
                async for event in stream:
                    if client_disconnected.is_set():
                        break
                    async with send_lock:
                        try:
                            await websocket.send_text(json.dumps(event))
                        except Exception:
                            client_disconnected.set()
                            break
                    if event["type"] == "exit":
                        break
        except KeyError:
            # Job vanished mid-stream (history pruned). Close cleanly.
            pass
        except Exception as exc:  # noqa: BLE001 - log + close
            logger.warning("ws/jobs/%s stream error: %s", job_id, exc)
        finally:
            reader_task.cancel()
            heartbeat_task.cancel()
            for t in (reader_task, heartbeat_task):
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001
                pass


__all__ = ["register_ws_routes"]
