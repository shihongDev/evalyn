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


async def _json_object_body(request: Request) -> dict:
    """Parse the request body as a JSON object or raise HTTP 400."""
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a json object")
    return body


@router.post("/threads/purge-old")
async def purge_old_threads(
    request: Request, max_age_s: int = 86400
) -> JSONResponse:
    """Drop closed threads whose newest event is older than
    ``max_age_s`` (default 24h).

    Returns ``{removed: N}``. Only closed threads are eligible - an
    open thread (mid-turn or awaiting confirmation) is never purged.
    Useful as a periodic cleanup on long-running daemons; can be
    wired to a cron-like task or invoked manually from an admin tool.
    """
    if max_age_s < 0:
        raise HTTPException(
            status_code=400, detail="max_age_s must be >= 0"
        )
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="agent runtime not configured")
    removed = runtime.purge_old_threads(max_age_s)
    return JSONResponse({"removed": removed})


@router.get("/threads/{thread_id}")
async def get_thread(request: Request, thread_id: str) -> JSONResponse:
    """Return one thread's metadata. Same shape as a single entry in
    GET /api/agent/threads. 404 if the thread is unknown.

    Useful when a client only knows the id (e.g. a deep link) and
    wants to confirm the thread still exists before opening a WS
    subscription.
    """
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="agent runtime not configured")
    meta = runtime.thread_metadata(thread_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"unknown thread: {thread_id}")
    return JSONResponse(meta)


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(request: Request, thread_id: str) -> JSONResponse:
    """Return the user + assistant message bodies for a thread.

    Useful when a client wants the conversation text without setting
    up a WebSocket - e.g. an agent-runner test harness checking what
    the assistant answered, a CLI tool scripting around the
    dashboard, or a scheduled report capturing thread snapshots.

    Filters out the system prompt (clients rarely care about it and
    including it would leak prompt text on a public-ish read
    endpoint). Tool-call structure inside assistant messages is
    preserved as-is.

    Response shape:
      {
        "id": "<thread_id>",
        "messages": [
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."},
          ...
        ]
      }

    404 when the thread is unknown.
    """
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=503, detail="agent runtime not configured"
        )
    messages = runtime.thread_messages(thread_id)
    if messages is None:
        raise HTTPException(
            status_code=404, detail=f"unknown thread: {thread_id}"
        )
    return JSONResponse({"id": thread_id, "messages": messages})


@router.delete("/threads/{thread_id}")
async def delete_thread(request: Request, thread_id: str) -> JSONResponse:
    """Remove an agent thread from the runtime.

    Closes any active /ws/agent subscribers (their async iterators
    unblock with StopAsyncIteration) and wakes any paused
    confirmation gate with reject=False. Idempotent on a 404 - a
    second DELETE for the same id returns 404 not 500.

    Mirrors POST /api/jobs/{id}/cancel + DELETE /api/jobs/{id}: this
    is admin cleanup for "I'm done with this thread", not a way to
    cancel a turn mid-flight (the turn loop continues running but
    its emits are no-ops once the thread object is gone).

    200 on success, 404 if unknown.
    """
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="agent runtime not configured")
    if not runtime.remove_thread(thread_id):
        raise HTTPException(status_code=404, detail=f"unknown thread: {thread_id}")
    return JSONResponse({"ok": True, "id": thread_id})


@router.get("/threads")
async def list_threads(request: Request) -> JSONResponse:
    """Return a snapshot of every active agent thread.

    Each entry: ``{id, message_count, event_count, closed,
    has_pending_confirmation}``. Useful for an admin/debug surface
    answering "what threads has this runtime accumulated?", or for
    polling whether a thread is gated on user confirmation (a stuck
    chat).

    The bodies of messages and events are NOT included - they can
    grow large and a list endpoint should stay lightweight. Fetch the
    full event stream via ``/ws/agent/{thread_id}``.
    """
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="agent runtime not configured")
    return JSONResponse(runtime.list_threads())


@router.post("/chat")
async def chat(request: Request) -> JSONResponse:
    body = await _json_object_body(request)

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
    body = await _json_object_body(request)
    approve = body.get("approve")
    if not isinstance(approve, bool):
        raise HTTPException(status_code=400, detail="approve must be a boolean")
    # Optional in the wire protocol so older clients keep working, but the
    # current dashboard always sends it. When supplied, AgentRuntime.confirm
    # validates against the per-thread pending gate so a stale card cannot
    # approve whatever happens to be pending now (KNOWN_ISSUES #4 + #5).
    tool_call_id = body.get("tool_call_id")
    if tool_call_id is not None and not isinstance(tool_call_id, str):
        raise HTTPException(status_code=400, detail="tool_call_id must be a string")
    # Optional inline-edit + per-session whitelist (P1 spec §5.5). The
    # frontend's redesigned confirmation card may send either or both when
    # the user clicks "Approve once" / "Approve - session" with edited args.
    args_override = body.get("args_override")
    if args_override is not None and not isinstance(args_override, dict):
        raise HTTPException(
            status_code=400, detail="args_override must be a json object"
        )
    auto_approve_session = body.get("auto_approve_session", False)
    if not isinstance(auto_approve_session, bool):
        raise HTTPException(
            status_code=400, detail="auto_approve_session must be a boolean"
        )

    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="agent runtime not configured")
    if not runtime.has_thread(thread_id):
        raise HTTPException(status_code=404, detail=f"unknown thread: {thread_id}")
    accepted = runtime.confirm(
        thread_id,
        approve,
        tool_call_id=tool_call_id,
        args_override=args_override,
        auto_approve_session=auto_approve_session,
    )
    if not accepted:
        # The thread exists but the tool_call_id didn't match the pending
        # gate. Surface this clearly so the client can recover (e.g. resync
        # by reading the next confirmation_required event) rather than
        # silently approving the wrong call.
        raise HTTPException(
            status_code=409,
            detail=(
                "tool_call_id does not match the currently pending confirmation"
            ),
        )
    return JSONResponse({"ok": True, "approved": approve})


__all__ = ["router"]
