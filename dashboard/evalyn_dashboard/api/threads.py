"""``/api/threads`` router — persist chat threads to disk as markdown.

POST ``/api/threads/save`` accepts a serialised thread (id, title, message
log) and writes it to ``<cwd>/.evalyn/threads/<thread-id>.md``. The
project-local ``.evalyn`` directory is preferred over ``~/.evalyn`` so
saved threads travel with the repo (matches how runs/datasets are stored).
If ``.evalyn`` cannot be created in the cwd we fall back to
``~/.evalyn/threads/`` so a user without write access to the project root
can still archive a conversation.

Response shape: ``{"path": "<absolute-path>"}`` on success.

Markdown format:

    # <title>

    ## <role> · <timestamp>
    <content>

    ### Tool: <tool>
    args:
      <key>: <value>
    output:
        <truncated to 4KB>

Tool output is truncated to ``MAX_TOOL_OUTPUT`` chars so a megabyte stdout
dump cannot blow up the disk write.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Per-message tool output truncation. Mirrors the localStorage cap on the
# frontend (4000 chars). Keeps the markdown file readable.
MAX_TOOL_OUTPUT = 4096

# Per-message text cap for safety; full conversations rarely exceed this
# but a malicious or buggy client could push megabytes through.
MAX_TEXT = 65_536

# Hard cap on the number of messages we'll write — a runaway loop should
# not be able to fill the disk via this endpoint.
MAX_MESSAGES = 5000


class SaveThreadRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    title: str = Field(..., max_length=512)
    messages: list[dict[str, Any]] = Field(default_factory=list)


def _safe_thread_id(thread_id: str) -> str:
    """Strip path-traversal characters; only allow ``[A-Za-z0-9._-]``.

    Anything else collapses to ``_``. An empty result raises 400 so we
    never write to ``.md`` (which would be reachable via the directory).
    """

    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", thread_id)
    if not cleaned or cleaned in {".", ".."}:
        raise HTTPException(status_code=400, detail="invalid thread id")
    return cleaned


def _resolve_threads_dir() -> Path:
    """Prefer ``<cwd>/.evalyn/threads/`` falling back to ``~/.evalyn/threads/``.

    ``mkdir(parents=True, exist_ok=True)`` is attempted on the project
    location first; on PermissionError or OSError we fall back.
    """

    candidates: list[Path] = [
        Path.cwd() / ".evalyn" / "threads",
        Path.home() / ".evalyn" / "threads",
    ]
    last_err: Optional[Exception] = None
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except (PermissionError, OSError) as exc:
            last_err = exc
            continue
    # Both paths failed — surface a 500 with the underlying message.
    raise HTTPException(
        status_code=500,
        detail=f"could not create threads directory: {last_err}",
    )


def _format_value(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)) or v is None:
        return str(v)
    try:
        return json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(v)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit]
    return f"{head}\n... [truncated, original {len(text)} chars]"


def _render_message(msg: dict[str, Any]) -> str:
    raw_role = msg.get("role")
    role = raw_role if isinstance(raw_role, str) and raw_role else "unknown"
    raw_ts = msg.get("ts")
    ts = raw_ts if isinstance(raw_ts, str) and raw_ts else ""
    header = f"## {role}" + (f" · {ts}" if ts else "")
    parts: list[str] = [header]

    text = msg.get("text")
    if isinstance(text, str) and text:
        parts.append(_truncate(text, MAX_TEXT))

    tool_call = msg.get("toolCall")
    if isinstance(tool_call, dict):
        tool_name = tool_call.get("tool", "<unknown>")
        if not isinstance(tool_name, str):
            tool_name = str(tool_name)
        parts.append(f"### Tool: {tool_name}")

        status = tool_call.get("status")
        if isinstance(status, str):
            parts.append(f"_status: {status}_")

        args = tool_call.get("args")
        if isinstance(args, dict) and args:
            parts.append("**args:**\n")
            arg_lines: list[str] = []
            for k, v in args.items():
                arg_lines.append(f"- `{k}`: `{_format_value(v)}`")
            parts.append("\n".join(arg_lines))

        output = tool_call.get("output")
        if isinstance(output, str) and output:
            truncated = _truncate(output, MAX_TOOL_OUTPUT)
            parts.append("**output:**\n")
            parts.append("```\n" + truncated + "\n```")

        error = tool_call.get("error")
        if isinstance(error, str) and error:
            parts.append("**error:**\n")
            parts.append("```\n" + _truncate(error, MAX_TOOL_OUTPUT) + "\n```")

    return "\n\n".join(parts)


def _render_thread(title: str, messages: list[dict[str, Any]]) -> str:
    safe_title = title.strip() or "Untitled thread"
    rendered_messages = [
        _render_message(m) for m in messages[:MAX_MESSAGES] if isinstance(m, dict)
    ]
    saved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header = f"# {safe_title}\n\n_saved {saved_at}_"
    body = "\n\n".join(rendered_messages)
    return f"{header}\n\n{body}\n" if body else f"{header}\n"


@router.post("/save")
async def save_thread(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a json object")

    try:
        payload = SaveThreadRequest(**body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if len(payload.messages) > MAX_MESSAGES:
        raise HTTPException(
            status_code=400,
            detail=f"too many messages (max {MAX_MESSAGES})",
        )

    safe_id = _safe_thread_id(payload.id)
    threads_dir = _resolve_threads_dir()
    file_path = threads_dir / f"{safe_id}.md"

    rendered = _render_thread(payload.title, payload.messages)
    try:
        file_path.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"failed to write thread: {exc}"
        ) from exc

    # Audit log: thread save writes a markdown file under the
    # workspace's threads dir. Lightweight metadata only - log the
    # safe id (already sanitized via _safe_thread_id) and the
    # message count, never message bodies (they can contain
    # API keys / PII the user pasted into the chat).
    logger.info(
        "thread saved: id=%s messages=%d",
        safe_id,
        len(payload.messages),
    )

    return JSONResponse({"path": str(file_path.resolve())})
