"""WebSocket route ``/ws/agent/{thread_id}`` (Lane C1.6).

Subscribes to an :class:`AgentRuntime` thread's event stream and pushes
events as JSON text frames. Reconnect-with-replay via the
``?since=<event_id>`` query parameter mirrors :mod:`api.jobs_ws`.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ._ws_heartbeat import spawn_heartbeat

logger = logging.getLogger(__name__)


def register_agent_ws_routes(app: FastAPI) -> None:
    @app.websocket("/ws/agent/{thread_id}")
    async def agent_ws(websocket: WebSocket, thread_id: str) -> None:
        since_raw: str | None = websocket.query_params.get("since")
        since: int | None = None
        if since_raw is not None:
            try:
                since = int(since_raw)
            except ValueError:
                since = None

        runtime = getattr(websocket.app.state, "agent_runtime", None)
        if runtime is None:
            await websocket.accept()
            await websocket.close(code=1011, reason="agent runtime not configured")
            return
        if not runtime.has_thread(thread_id):
            await websocket.accept()
            await websocket.close(code=1008, reason="unknown thread")
            return

        await websocket.accept()
        send_lock = asyncio.Lock()
        client_disconnected = asyncio.Event()

        async def reader() -> None:
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                client_disconnected.set()
            except Exception:  # noqa: BLE001
                client_disconnected.set()

        reader_task = asyncio.create_task(reader())
        heartbeat_task = spawn_heartbeat(
            websocket, send_lock, client_disconnected,
        )

        try:
            async with runtime.subscribe(thread_id, since=since) as stream:
                async for event in stream:
                    if client_disconnected.is_set():
                        break
                    async with send_lock:
                        try:
                            await websocket.send_text(json.dumps(event))
                        except Exception:  # noqa: BLE001
                            client_disconnected.set()
                            break
                    if event.get("type") == "final":
                        break
        except KeyError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("ws/agent/%s stream error: %s", thread_id, exc)
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


__all__ = ["register_agent_ws_routes"]
