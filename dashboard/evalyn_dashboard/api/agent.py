"""``/api/agent`` router: chat + confirmation routes (Lane C1.5, C1.7).

POST ``/api/agent/chat`` accepts ``{message, thread_id?}``. When
``thread_id`` is omitted (or unknown) a fresh thread is created. The body
is appended to the thread and the agent loop is scheduled as a fire-and-
forget asyncio task. Returns ``{thread_id}`` immediately.

POST ``/api/agent/chat/{thread_id}/confirm`` accepts ``{approve}`` and
flips the per-thread :class:`asyncio.Event` so a paused tool call can
proceed (or report ``user did not confirm``).

The companion WebSocket route ``/ws/agent/{thread_id}`` lives in
``api/agent_ws.py`` and is registered directly on the FastAPI app.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/chat")
async def chat(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a json object")

    message = body.get("message")
    if not isinstance(message, str) or not message:
        raise HTTPException(status_code=400, detail="message must be a non-empty string")

    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="agent runtime not configured")

    thread_id = body.get("thread_id")
    if not isinstance(thread_id, str) or not runtime.has_thread(thread_id):
        thread_id = runtime.create_thread()

    runtime.schedule_turn(thread_id, message)
    return JSONResponse({"thread_id": thread_id})


@router.post("/chat/{thread_id}/confirm")
async def confirm(request: Request, thread_id: str) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a json object")
    approve = body.get("approve")
    if not isinstance(approve, bool):
        raise HTTPException(status_code=400, detail="approve must be a boolean")

    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="agent runtime not configured")
    if not runtime.confirm(thread_id, approve):
        raise HTTPException(status_code=404, detail=f"unknown thread: {thread_id}")
    return JSONResponse({"ok": True, "approved": approve})


__all__ = ["router"]
