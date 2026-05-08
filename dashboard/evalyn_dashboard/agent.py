"""Agent runtime for the evalyn dashboard (Phase 3 Lane C1).

Exposes a multi-provider streaming chat runtime that uses the dashboard's
introspected CLI catalog as a tool surface. Read-only commands run
automatically; write/destructive commands require explicit user
confirmation through an :class:`asyncio.Event` gate.

Public surface:

- :class:`BaseProvider` - abstract async streaming provider interface.
- :class:`OpenAIProvider`, :class:`AnthropicProvider`,
  :class:`OllamaProvider` - concrete implementations.
- :class:`AgentRuntime` - thread state + tool loop + per-thread event
  buffer for WebSocket fanout.
- :func:`catalog_to_tools` - canonical tool serialization for any
  provider.
- :data:`READ_ONLY_ALLOWLIST` - hard-coded set of CLI ids that may
  auto-run without user confirmation.

Event shapes pushed through :meth:`AgentRuntime.subscribe`:

    {"type": "text_delta",          "thread_id": str, "text": str, "ts": float, "event_id": int}
    {"type": "tool_call_proposal",  "thread_id": str, "tool_call_id": str, "tool": str,
                                     "args": dict, "ts": float, "event_id": int}
    {"type": "tool_call_running",   "thread_id": str, "tool_call_id": str, "tool": str,
                                     "ts": float, "event_id": int}
    {"type": "tool_call_complete",  "thread_id": str, "tool_call_id": str, "tool": str,
                                     "ok": bool, "stdout": str, "exit_code": int, "ts": float,
                                     "event_id": int}
    {"type": "confirmation_required","thread_id": str, "tool_call_id": str, "tool": str,
                                     "args": dict, "preview_cmd": str, "ts": float,
                                     "event_id": int}
    {"type": "final",               "thread_id": str, "reason": str, "ts": float,
                                     "event_id": int}
    {"type": "error",               "thread_id": str, "message": str, "ts": float,
                                     "event_id": int}
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Hard-coded read-only allowlist (spec 7 / 10). Any CLI not in this set
# requires explicit user confirmation before execution.
READ_ONLY_ALLOWLIST: frozenset[str] = frozenset(
    {
        "list-calls",
        "list-runs",
        "list-metrics",
        "list-calibrations",
        "show-call",
        "show-trace",
        "show-span",
        "show-projects",
        "analyze",
        "compare",
        "trend",
        "annotation-stats",
        "validate",
        "status",
        "workflow",
        "cluster-failures",
        "cluster-misalignments",
        "insights",
        "select-metrics",
    }
)

# Per-turn caps. The tool-call ceiling guards against runaway loops.
DEFAULT_TOOL_BUDGET = 8
DEFAULT_CONFIRM_TIMEOUT = 300.0  # 5 minutes
DEFAULT_TOOL_OUTPUT_CAP = 50 * 1024  # 50 KB
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


# ---------------------------------------------------------------------------
# System prompt + frontend-only tools.
# ---------------------------------------------------------------------------
#
# The agent is the dashboard co-pilot. It reads/writes the project via the CLI
# catalog AND can drive the dashboard UI itself via a small set of "frontend"
# tools that the backend never executes - only emits a tool_call_proposal for.
# The frontend (useCoPilotThread) intercepts these and acts on them in JS land
# (e.g. start_tour -> Zustand setTour). The agent loop sees a synthetic success
# response so it can continue its turn coherently.

SYSTEM_PROMPT = (
    "You are the co-pilot for evalyn, an LLM evaluation toolkit.\n"
    "\n"
    "You help the user understand their evaluation runs, debug failures, and "
    "operate the evalyn CLI. You have access to evalyn CLI commands as tools. "
    "Use them to list runs, inspect rubrics, fetch dataset items, and answer "
    "the user's questions with concrete data instead of guessing.\n"
    "\n"
    "You also have a special `start_tour` tool that triggers a guided UI "
    "walk-through in the dashboard. Use start_tour ONLY when the user "
    "explicitly asks to be SHOWN how to do something (phrases like \"show me "
    "how to...\", \"walk me through...\", \"how do I... [UI action]\"). Do not "
    "call start_tour proactively, do not call it as a substitute for "
    "answering a content question, and call at most one tour per turn.\n"
    "\n"
    "Available tour ids:\n"
    "  - firstRun.home.v1: orientation walk-through of the Home dashboard.\n"
    "  - datasetUpload.v1: how to add a dataset on the Datasets page.\n"
    "  - runEval.v1: how to start a new evaluation on the Experiments page.\n"
    "  - reviewFailures.v1: how to triage failures in the Review queue.\n"
    "  - readMetrics.v1: how to read the per-rubric metrics page.\n"
    "\n"
    "Be concise. When you do not know the answer, say so and offer to help "
    "the user find it via the available tools."
)


_FRONTEND_TOOLS: list[dict[str, Any]] = [
    {
        "name": "start_tour",
        "description": (
            "Start a guided UI walk-through of the dashboard. Use ONLY when "
            "the user explicitly asks to be shown how to do something. The "
            "frontend handles the tour visually; this tool returns "
            "immediately on the backend with no side effects."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tour_id": {
                    "type": "string",
                    "description": "Identifier of the tour to start.",
                    "enum": [
                        "firstRun.home.v1",
                        "datasetUpload.v1",
                        "runEval.v1",
                        "reviewFailures.v1",
                        "readMetrics.v1",
                    ],
                },
            },
            "required": ["tour_id"],
        },
    },
]

FRONTEND_ONLY_TOOL_NAMES: frozenset[str] = frozenset({"start_tour"})


def _frontend_tool_result(name: str, args: dict[str, Any]) -> str:
    """Synthetic stdout returned to the agent loop after a frontend-only
    tool has been dispatched to the UI. The text is what the model sees as
    the tool result, so it should describe what happened in human terms."""
    if name == "start_tour":
        tour_id = args.get("tour_id", "?")
        return f"Tour '{tour_id}' has been displayed to the user."
    return f"{name} dispatched to the frontend."


# ---------------------------------------------------------------------------
# Side-effects copy for the confirmation card.
# ---------------------------------------------------------------------------
#
# The chat confirmation card surfaces a "THIS WILL" bullet list so the user
# can see what they are about to approve in plain English (P1 spec §5.5).
# Keys are evalyn CLI ids that are NOT in :data:`READ_ONLY_ALLOWLIST` (i.e.
# write or potentially destructive commands). Anything missing falls back to
# :data:`DEFAULT_SIDE_EFFECTS` so an unknown write tool still surfaces a
# helpful warning.

DEFAULT_SIDE_EFFECTS: list[str] = [
    "This is a write command and may modify your project on disk.",
]

SIDE_EFFECTS: dict[str, list[str]] = {
    "run-eval": [
        "Spawn an evalyn run-eval subprocess",
        "Write a new eval run under .evalyn/data/datasets/<dataset>/eval_runs/<new-id>/",
        "Stream stdout to a job tab",
    ],
    "calibrate": [
        "Spawn an evalyn calibrate subprocess",
        "Write calibration record under .evalyn/data/datasets/<dataset>/calibrations/",
        "May make many LLM calls (cost-relevant)",
    ],
    "delete-traces": [
        "Permanently delete trace records from .evalyn/data/traces.sqlite",
        "This action cannot be undone",
    ],
    "build-dataset": [
        "Read traces from .evalyn/data/traces.sqlite",
        "Write a new dataset directory under .evalyn/data/datasets/<dataset_name>/",
    ],
    "annotate": [
        "Open an annotation session",
        "Write annotations to disk",
    ],
    "import-annotations": [
        "Read annotations from a JSONL file",
        "Write/merge into .evalyn/data/datasets/<dataset>/annotations/",
    ],
    "simulate": [
        "Generate synthetic traces",
        "Write to .evalyn/data/traces.sqlite",
    ],
    "one-click": [
        "Run the full pipeline (dataset -> eval -> judge)",
        "Multiple write operations across .evalyn/",
    ],
    "export": [
        "Read run results",
        "Write export bundle to disk",
    ],
    "export-for-annotation": [
        "Read traces",
        "Write annotation queue file to disk",
    ],
    "init": [
        "Create .evalyn/ directory structure if missing",
        "Write evalyn.yaml",
    ],
    "quickstart": [
        "Detect agent code",
        "Add @eval decorator",
        "Write evalyn.yaml",
    ],
    "report": [
        "Generate static HTML report",
        "Write to specified output path",
    ],
    "dashboard": [
        "Generate static HTML report",
        "Write to specified output path",
    ],
    "suggest-metrics": [
        "May make LLM calls if --use-llm is set",
    ],
}


def _side_effects_for(tool: str) -> list[str]:
    """Return the bullet list for a tool name, falling back to a default."""
    return list(SIDE_EFFECTS.get(tool, DEFAULT_SIDE_EFFECTS))


# ---------------------------------------------------------------------------
# Canonical tool format
# ---------------------------------------------------------------------------


def _kind_to_json_schema(kind: str, options: list[str] | None) -> dict[str, Any]:
    """Map a ``ParamKind`` to a minimal JSON Schema fragment."""
    if kind == "bool":
        return {"type": "boolean"}
    if kind == "number":
        return {"type": "number"}
    if kind == "select":
        schema: dict[str, Any] = {"type": "string"}
        if options:
            schema["enum"] = list(options)
        return schema
    if kind == "multiselect":
        item: dict[str, Any] = {"type": "string"}
        if options:
            item["enum"] = list(options)
        return {"type": "array", "items": item}
    # path / long-text / string all serialize as plain strings; provider
    # tool schemas don't carry a richer notion of "path" or "textarea".
    return {"type": "string"}


def catalog_to_tools(catalog: list[Any]) -> list[dict[str, Any]]:
    """Convert a list of :class:`CliSchema` into a canonical tool list.

    Returned shape (provider-neutral):

        {"name": "<cli-id>", "description": "<blurb>", "parameters":
         {"type": "object", "properties": {...}, "required": [...]}}

    Provider-specific serialization is handled by the providers (e.g.
    OpenAI wraps each entry with ``{"type": "function", "function": ...}``;
    Anthropic uses ``input_schema`` instead of ``parameters``).
    """
    tools: list[dict[str, Any]] = []
    for schema in catalog:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in schema.params:
            properties[param.name] = _kind_to_json_schema(param.kind, param.options)
            if getattr(param, "help", None):
                properties[param.name]["description"] = param.help
            if getattr(param, "default", None) is not None:
                # Surface defaults so the model can avoid restating them.
                properties[param.name]["default"] = param.default
            if getattr(param, "required", False):
                required.append(param.name)
        params_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            params_schema["required"] = required
        tools.append(
            {
                "name": schema.id,
                "description": (schema.blurb or schema.name)[:1024],
                "parameters": params_schema,
            }
        )
    return tools


def _to_openai_tools(canonical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in canonical
    ]


def _to_anthropic_tools(canonical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in canonical
    ]


def _to_ollama_tools(canonical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Ollama follows the OpenAI tool shape.
    return _to_openai_tools(canonical)


# ---------------------------------------------------------------------------
# Provider events
# ---------------------------------------------------------------------------


@dataclass
class ProviderToolCall:
    """A single tool call surfaced by the provider's stream."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderEvent:
    """One event from :meth:`BaseProvider.stream_chat`.

    ``kind`` is one of:

    - ``"text_delta"`` - ``text`` carries an incremental string.
    - ``"tool_call"`` - ``tool_call`` carries the parsed call.
    - ``"finish"`` - the provider has emitted everything for this turn.
    - ``"error"`` - ``message`` describes what went wrong.
    """

    kind: str
    text: str | None = None
    tool_call: ProviderToolCall | None = None
    message: str | None = None


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class BaseProvider(ABC):
    """Abstract streaming chat provider."""

    name: str = ""

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[ProviderEvent]:  # pragma: no cover - interface
        ...


class OpenAIProvider(BaseProvider):
    """Streaming OpenAI chat completion provider."""

    name = "openai"

    def __init__(self, *, api_key: str, model: str, client: Any | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self._client = client  # test seam

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        import openai  # type: ignore[import-not-found]

        self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[ProviderEvent]:
        client = self._ensure_client()
        provider_tools = _to_openai_tools(tools) if tools else None

        def _make_stream() -> Any:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": True,
            }
            if provider_tools:
                kwargs["tools"] = provider_tools
            return client.chat.completions.create(**kwargs)

        try:
            stream = await asyncio.to_thread(_make_stream)
        except Exception as exc:  # noqa: BLE001
            yield ProviderEvent(kind="error", message=str(exc))
            return

        # Aggregate tool-call fragments across deltas (OpenAI streams them
        # incrementally by index).
        tool_buffers: dict[int, dict[str, str]] = {}

        def _next_chunk(it: Any) -> Any:
            try:
                return next(it)
            except StopIteration:
                return None

        iterator = iter(stream)
        while True:
            chunk = await asyncio.to_thread(_next_chunk, iterator)
            if chunk is None:
                break
            try:
                delta = chunk.choices[0].delta  # type: ignore[index]
            except Exception:  # noqa: BLE001
                continue
            text = getattr(delta, "content", None)
            if text:
                yield ProviderEvent(kind="text_delta", text=text)
            tool_calls = getattr(delta, "tool_calls", None) or []
            for tc in tool_calls:
                idx = getattr(tc, "index", 0) or 0
                buf = tool_buffers.setdefault(idx, {"id": "", "name": "", "args": ""})
                if getattr(tc, "id", None):
                    buf["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        buf["name"] = fn.name
                    frag = getattr(fn, "arguments", None)
                    if frag:
                        buf["args"] += frag

        for buf in tool_buffers.values():
            if not buf.get("name"):
                continue
            try:
                args = json.loads(buf["args"]) if buf["args"] else {}
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            yield ProviderEvent(
                kind="tool_call",
                tool_call=ProviderToolCall(
                    id=buf["id"] or uuid.uuid4().hex,
                    name=buf["name"],
                    arguments=args,
                ),
            )

        yield ProviderEvent(kind="finish")


class AnthropicProvider(BaseProvider):
    """Streaming Anthropic Messages provider."""

    name = "anthropic"

    def __init__(self, *, api_key: str, model: str, client: Any | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        import anthropic  # type: ignore[import-not-found]

        self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[ProviderEvent]:
        client = self._ensure_client()
        anthro_tools = _to_anthropic_tools(tools) if tools else None

        # Anthropic separates the system prompt from messages.
        system: str | None = None
        msgs: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content") or ""
            else:
                msgs.append(m)

        def _make_stream() -> Any:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": msgs,
            }
            if system:
                kwargs["system"] = system
            if anthro_tools:
                kwargs["tools"] = anthro_tools
            return client.messages.stream(**kwargs)

        try:
            stream_ctx = await asyncio.to_thread(_make_stream)
        except Exception as exc:  # noqa: BLE001
            yield ProviderEvent(kind="error", message=str(exc))
            return

        tool_buffers: dict[int, dict[str, Any]] = {}

        def _enter(ctx: Any) -> Any:
            return ctx.__enter__()

        def _exit(ctx: Any) -> None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

        def _next_chunk(it: Any) -> Any:
            try:
                return next(it)
            except StopIteration:
                return None

        try:
            stream = await asyncio.to_thread(_enter, stream_ctx)
            iterator = iter(stream)
            while True:
                event = await asyncio.to_thread(_next_chunk, iterator)
                if event is None:
                    break
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    block = getattr(event, "content_block", None)
                    idx = getattr(event, "index", 0) or 0
                    if block is not None and getattr(block, "type", None) == "tool_use":
                        tool_buffers[idx] = {
                            "id": getattr(block, "id", "") or uuid.uuid4().hex,
                            "name": getattr(block, "name", "") or "",
                            "args": "",
                        }
                elif etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    idx = getattr(event, "index", 0) or 0
                    if delta is None:
                        continue
                    dtype = getattr(delta, "type", None)
                    if dtype == "text_delta":
                        text = getattr(delta, "text", "") or ""
                        if text:
                            yield ProviderEvent(kind="text_delta", text=text)
                    elif dtype == "input_json_delta":
                        partial = getattr(delta, "partial_json", "") or ""
                        if idx in tool_buffers:
                            tool_buffers[idx]["args"] += partial
                elif etype == "message_stop":
                    break
        finally:
            await asyncio.to_thread(_exit, stream_ctx)

        for buf in tool_buffers.values():
            if not buf.get("name"):
                continue
            try:
                args = json.loads(buf["args"]) if buf["args"] else {}
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            yield ProviderEvent(
                kind="tool_call",
                tool_call=ProviderToolCall(
                    id=buf["id"], name=buf["name"], arguments=args
                ),
            )

        yield ProviderEvent(kind="finish")


class OllamaProvider(BaseProvider):
    """Streaming Ollama chat provider via httpx."""

    name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = client  # injected httpx.AsyncClient for tests

    @asynccontextmanager
    async def _httpx_client(self) -> AsyncIterator[Any]:
        if self._client is not None:
            yield self._client
            return
        import httpx

        async with httpx.AsyncClient(timeout=None) as client:
            yield client

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[ProviderEvent]:
        # O3 note: runtime tool-support detection is intentionally NOT
        # implemented here. Many Ollama models silently ignore the `tools`
        # payload and reply with free text, which would look indistinguishable
        # from a normal text-only turn. The gate for tool-capability is the
        # test-connection probe in `credentials.CredentialStore._test_ollama`,
        # which sends a 1-token /api/chat probe with a trivial tool and
        # rejects the credential if the model does not return a `tool_calls`
        # field. Adding a runtime check here would either break legitimate
        # text-only turns from a tool-capable model or duplicate the probe
        # cost on every chat call without a reliable signal.
        ollama_tools = _to_ollama_tools(tools) if tools else None
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if ollama_tools:
            payload["tools"] = ollama_tools

        url = f"{self.base_url}/api/chat"
        try:
            async with self._httpx_client() as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    async for raw_line in response.aiter_lines():
                        if not raw_line:
                            continue
                        try:
                            chunk = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue
                        message = chunk.get("message") or {}
                        content = message.get("content") or ""
                        if content:
                            yield ProviderEvent(kind="text_delta", text=content)
                        for tc in message.get("tool_calls") or []:
                            fn = tc.get("function") or {}
                            args = fn.get("arguments")
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {}
                            if not isinstance(args, dict):
                                args = {}
                            yield ProviderEvent(
                                kind="tool_call",
                                tool_call=ProviderToolCall(
                                    id=tc.get("id") or uuid.uuid4().hex,
                                    name=fn.get("name") or "",
                                    arguments=args,
                                ),
                            )
                        if chunk.get("done"):
                            break
        except Exception as exc:  # noqa: BLE001
            yield ProviderEvent(kind="error", message=str(exc))
            return

        yield ProviderEvent(kind="finish")


# ---------------------------------------------------------------------------
# Tool execution helper
# ---------------------------------------------------------------------------


def _argv_for_tool(tool: str, args: dict[str, Any]) -> list[str]:
    """Render ``tool`` + ``args`` into an argv list for ``evalyn ...``.

    This is intentionally permissive: providers occasionally hallucinate
    flag names, and we'd rather pass them through to evalyn (and let it
    return a friendly error in the tool result) than fail hard inside the
    runtime. Validation against the catalog is a future enhancement.
    """
    argv: list[str] = ["evalyn", tool]
    for name, value in args.items():
        flag = "--" + str(name).replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if value is None or value == "":
            continue
        if isinstance(value, list):
            if not value:
                continue
            argv.append(flag)
            argv.extend(str(v) for v in value)
            continue
        argv.extend([flag, str(value)])
    return argv


def _preview_command(tool: str, args: dict[str, Any]) -> str:
    return " ".join(_argv_for_tool(tool, args))


# ---------------------------------------------------------------------------
# AgentRuntime
# ---------------------------------------------------------------------------


@dataclass
class _Thread:
    id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: set["_AgentEventStream"] = field(default_factory=set)
    confirm_event: asyncio.Event = field(default_factory=asyncio.Event)
    confirm_decision: bool = False
    # Tool-call id currently waiting on user confirmation. Set when
    # `_execute_tool_call` arms the gate; cleared once the gate fires.
    # Used by `AgentRuntime.confirm` to reject stale confirmations from
    # the UI (KNOWN_ISSUES #4 + #5).
    pending_tool_call_id: str | None = None
    # Live ProviderToolCall awaiting confirmation. Held so an `args_override`
    # from the UI can mutate it in place before `_execute_tool_call` resumes.
    # Underscore-prefixed because the WS event payload already exposes the id;
    # this is the in-memory object tests reach into.
    _pending_tool_call_obj: Optional["ProviderToolCall"] = None
    # Per-session whitelist of tool names the user has chosen to auto-approve
    # for the rest of this thread (P1 spec §5.5: "Approve - don't ask again
    # this session for <tool>"). Populated by `AgentRuntime.confirm` when the
    # caller passes `auto_approve_session=True`. Not persisted across thread
    # resets; cleared whenever the thread is recreated.
    session_auto_approved: set[str] = field(default_factory=set)
    _next_event_id: int = 1
    _turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False


class _AgentEventStream:
    """Per-subscriber queue, async-iterable until thread closes.

    Replay phase (closes KNOWN_ISSUES.md #3): ``AgentRuntime.subscribe``
    flips the stream into replay mode before registering it for live
    fanout. While replaying, live events arriving via ``put_nowait``
    divert into ``_live_buffer``; replay events use ``_replay_put`` to
    bypass the buffer. ``_end_replay`` flushes the buffer in emit order
    and switches the stream to direct fanout. This keeps order correct
    even if a future change adds awaits in the subscribe body.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._closed = False
        self._replaying = False
        self._live_buffer: list[dict[str, Any]] = []

    def put_nowait(self, event: dict[str, Any]) -> None:
        if self._replaying:
            self._live_buffer.append(event)
        else:
            self._queue.put_nowait(event)

    def _replay_put(self, event: dict[str, Any]) -> None:
        """Replay path: bypass the live buffer when enqueuing history."""
        self._queue.put_nowait(event)

    def _begin_replay(self) -> None:
        self._replaying = True

    def _end_replay(self) -> None:
        for event in self._live_buffer:
            self._queue.put_nowait(event)
        self._live_buffer.clear()
        self._replaying = False

    def close(self) -> None:
        self._closed = True
        # Wake any pending __anext__ so subscribers can exit.
        try:
            self._queue.put_nowait({"type": "__stream_close__"})
        except Exception:  # noqa: BLE001
            pass

    def __aiter__(self) -> "_AgentEventStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._closed and self._queue.empty():
            raise StopAsyncIteration
        event = await self._queue.get()
        if event.get("type") == "__stream_close__":
            raise StopAsyncIteration
        return event


# Type alias for tool runners. Returns ``(stdout, exit_code)``.
ToolRunner = Callable[[list[str]], Awaitable[tuple[str, int]]]


class AgentRuntime:
    """Manages chat threads, tool loops, and event fanout.

    Parameters mirror the spec: a provider factory (so the active
    provider can change between turns), the catalog (for tool surfaces),
    and the :class:`JobManager` used to execute approved tools. The
    ``tool_runner`` parameter is a test seam that bypasses
    :class:`JobManager` entirely.
    """

    def __init__(
        self,
        *,
        provider_factory: Callable[[], BaseProvider | None],
        catalog: list[Any] | None = None,
        job_manager: Any | None = None,
        tool_runner: ToolRunner | None = None,
        tool_budget: int = DEFAULT_TOOL_BUDGET,
        confirm_timeout: float = DEFAULT_CONFIRM_TIMEOUT,
        tool_output_cap: int = DEFAULT_TOOL_OUTPUT_CAP,
        allowlist: frozenset[str] = READ_ONLY_ALLOWLIST,
    ) -> None:
        self._provider_factory = provider_factory
        self._catalog = catalog or []
        self._job_manager = job_manager
        self._tool_runner = tool_runner
        self.tool_budget = tool_budget
        self.confirm_timeout = confirm_timeout
        self.tool_output_cap = tool_output_cap
        self.allowlist = allowlist
        self._threads: dict[str, _Thread] = {}

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def create_thread(self) -> str:
        thread_id = uuid.uuid4().hex
        thread = _Thread(id=thread_id)
        # Inject the co-pilot system prompt as the first message so providers
        # see the tour-policy instructions on every turn. Anthropic separates
        # `system` from `messages` (handled in AnthropicProvider.stream_chat);
        # OpenAI and Ollama accept role:"system" entries directly.
        thread.messages.append({"role": "system", "content": SYSTEM_PROMPT})
        self._threads[thread_id] = thread
        return thread_id

    def has_thread(self, thread_id: str) -> bool:
        return thread_id in self._threads

    def messages(self, thread_id: str) -> list[dict[str, Any]]:
        thread = self._threads.get(thread_id)
        return list(thread.messages) if thread else []

    def remove_thread(self, thread_id: str) -> bool:
        """Drop a thread from the runtime. Returns True if the thread
        existed (and is now gone), False on miss.

        Closes any active stream subscribers so their async iterators
        unblock with StopAsyncIteration. The pending confirmation gate
        (if any) is fired with reject=False so a paused tool-call
        coroutine returns to the caller cleanly. Idempotent on the
        already-removed case.

        Note: a turn that is mid-execution will continue running its
        async loop, but the runtime no longer holds a reference to the
        thread object so the loop's eventual ``_emit`` calls are
        no-ops (their target is the orphan ``thread.subscribers`` set,
        already closed). The HTTP caller should not rely on the loop
        actually stopping immediately - this is admin cleanup, not
        cancellation.
        """
        thread = self._threads.get(thread_id)
        if thread is None:
            return False
        # Close subscribers before deleting so any /ws/agent/{id} stream
        # iterator unblocks with StopAsyncIteration. Best-effort - don't
        # let one buggy stream object stop us from removing.
        for sub in list(thread.subscribers):
            try:
                sub.close()
            except Exception:  # noqa: BLE001
                pass
        # Wake any pending confirmation gate so a paused tool-call
        # coroutine returns. Confirm decision is False (treated as user
        # did not confirm).
        try:
            thread.confirm_decision = False
            thread.confirm_event.set()
        except Exception:  # noqa: BLE001
            pass
        del self._threads[thread_id]
        return True

    def _thread_to_dict(self, thread: "_Thread") -> dict[str, Any]:
        """Project a ``_Thread`` to its JSON-friendly metadata shape.

        Shared by ``list_threads`` and ``thread_metadata`` so the two
        surfaces never drift. Bodies of messages and events are NOT
        included - the listing should stay lightweight; full event
        stream is reachable via ``/ws/agent/{thread_id}``.

        ``last_event_at`` is the wall-clock timestamp (unix epoch
        seconds) of the most recent event, or None when the thread
        has none yet. Useful for "is this thread stuck?" detection:
        a thread with ``has_pending_confirmation=True`` AND a stale
        ``last_event_at`` is awaiting user action; one with a fresh
        timestamp is mid-turn. Without this field, a client polling
        the listing has no way to tell live activity from idle.
        """
        last_event_at: float | None = None
        if thread.events:
            ts = thread.events[-1].get("ts")
            if isinstance(ts, (int, float)):
                last_event_at = float(ts)
        return {
            "id": thread.id,
            "message_count": len(thread.messages),
            "event_count": len(thread.events),
            "closed": thread.closed,
            "has_pending_confirmation": (
                thread.pending_tool_call_id is not None
            ),
            "last_event_at": last_event_at,
        }

    def list_threads(self) -> list[dict[str, Any]]:
        """Return a JSON-friendly snapshot of every active thread.

        Each entry: ``{id, message_count, event_count, closed,
        has_pending_confirmation}``. Useful for admin/debug surfaces
        answering "what threads has this runtime accumulated?". The
        message and event counts are cheap (len() on the lists);
        ``closed`` reflects whether ``final`` has fired; the
        confirmation flag tells the caller a turn is gated on user
        approval (a stuck thread).
        """
        return [self._thread_to_dict(t) for t in self._threads.values()]

    def thread_messages(self, thread_id: str) -> list[dict[str, Any]] | None:
        """Return a copy of the thread's user+assistant messages, or
        None if the thread is unknown.

        Filters out the system prompt (clients of the HTTP endpoint
        rarely care about it; including it would also leak the prompt
        text on a public-ish read endpoint). Returns a deep-enough
        copy so callers can mutate freely without affecting the
        runtime's in-memory list.

        Used by ``GET /api/agent/threads/{id}/messages`` for clients
        that want the conversation text without setting up a WS
        subscription (e.g. an agent-runner test harness, a CLI tool
        scripting around the dashboard).
        """
        thread = self._threads.get(thread_id)
        if thread is None:
            return None
        return [
            dict(m)
            for m in thread.messages
            if m.get("role") != "system"
        ]

    def thread_counts(self) -> dict[str, int]:
        """Return ``{total, open}`` thread counts.

        Cheap aggregate for the healthcheck and any future capacity
        surfaces. ``open`` excludes threads whose ``final`` has fired
        (which would otherwise distort "are agents stuck?" alerting).
        Implemented as a public helper so callers don't have to reach
        into ``_threads`` directly - that would couple server.py to
        a private attribute and break under future restructuring.
        """
        total = len(self._threads)
        open_count = sum(1 for t in self._threads.values() if not t.closed)
        return {"total": total, "open": open_count}

    def thread_metadata(self, thread_id: str) -> dict[str, Any] | None:
        """Return the metadata dict for a single thread, or None on miss.
        Same shape as one entry in ``list_threads``."""
        thread = self._threads.get(thread_id)
        return self._thread_to_dict(thread) if thread is not None else None

    def purge_old_threads(
        self,
        max_age_seconds: float,
        now: float | None = None,
    ) -> int:
        """Drop closed threads whose newest event is older than the
        cutoff. Returns the number of threads removed.

        Without this helper, ``self._threads`` accumulates indefinitely
        on a long-running daemon - every chat thread sticks around in
        memory after its ``final`` event, even though no consumer can
        meaningfully reattach (the WS subscriber would just get the
        replay-then-close cycle on every reconnect).

        Only ``closed`` threads are eligible: an open thread (mid-turn
        or awaiting confirmation) must not be purged out from under
        its own loop. Pre-final threads with no events also stay -
        they may have been just-created and not yet had ``start_turn``
        called.

        ``now`` is overridable for tests. Reads each thread's newest
        event ts via ``max(events)`` so a long-running closed thread
        with continuing emits would also stay (defensive; today
        ``closed`` is set only by ``final`` and emits stop after).
        """
        import time as _time

        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must be >= 0")
        cutoff = (_time.time() if now is None else now) - max_age_seconds
        to_remove: list[str] = []
        for tid, thread in self._threads.items():
            if not thread.closed:
                continue
            if not thread.events:
                # Closed but no events should not happen given current
                # _emit semantics, but skip defensively rather than
                # purging on a missing-data signal.
                continue
            last_ts = max(e.get("ts", 0.0) for e in thread.events)
            if last_ts < cutoff:
                to_remove.append(tid)
        for tid in to_remove:
            self.remove_thread(tid)
        return len(to_remove)

    def confirm(
        self,
        thread_id: str,
        approve: bool,
        tool_call_id: str | None = None,
        args_override: dict[str, Any] | None = None,
        auto_approve_session: bool = False,
    ) -> bool:
        """Resolve a pending confirmation gate.

        ``tool_call_id`` (when supplied) MUST match the thread's currently
        pending tool-call id. This guards against the UI race documented in
        KNOWN_ISSUES #4 + #5: with multiple stale confirmation cards on
        screen, a click on the wrong card would otherwise approve whatever
        is now pending. Mismatches return ``False`` and leave the gate
        untouched. ``tool_call_id`` of None preserves backward compatibility
        with callers that don't track the id (legacy tests).

        ``args_override`` (when provided alongside ``approve=True``) replaces
        the pending tool call's arguments before the runtime resumes. The
        in-flight ``_execute_tool_call`` reads ``tc.arguments`` after the
        gate fires, so we mutate the live ``ProviderToolCall`` it's holding
        via the per-thread ``_pending_tool_call_obj`` reference (set when
        the gate was armed).

        ``auto_approve_session=True`` adds the pending tool's *name* to the
        per-thread :attr:`_Thread.session_auto_approved` whitelist so the
        next call to the same tool in this thread skips the gate entirely.
        Rejection (``approve=False``) ignores both override flags - a
        rejection is a rejection.
        """
        thread = self._threads.get(thread_id)
        if thread is None:
            return False
        if tool_call_id is not None:
            pending = thread.pending_tool_call_id
            if pending is None or pending != tool_call_id:
                logger.warning(
                    "agent.confirm: stale tool_call_id %s (pending %s) "
                    "on thread %s; refusing",
                    tool_call_id,
                    pending,
                    thread_id,
                )
                return False
        if approve:
            pending_tc = thread._pending_tool_call_obj
            if args_override is not None and pending_tc is not None:
                # Restrict overrides to the SET OF KEYS the model already
                # proposed. Without this, a user could approve a benign
                # `analyze --run X` confirmation while injecting an
                # unrelated `--unsafe-flag` (or even pivot to writing a
                # different output path the agent never asked about).
                # The CSRF gate prevents cross-origin abuse, but a
                # confused-deputy attack via a hijacked browser tab still
                # warrants this defence-in-depth check.
                proposed_keys = set(pending_tc.arguments.keys())
                override_keys = set(args_override.keys())
                extra = override_keys - proposed_keys
                if extra:
                    logger.warning(
                        "agent.confirm: args_override adds keys the model "
                        "did not propose: %s on thread %s; refusing",
                        sorted(extra),
                        thread_id,
                    )
                    return False
                pending_tc.arguments = {
                    **pending_tc.arguments,
                    **args_override,
                }
            if auto_approve_session and pending_tc is not None and pending_tc.name:
                thread.session_auto_approved.add(pending_tc.name)
        thread.confirm_decision = approve
        thread.confirm_event.set()
        return True

    @asynccontextmanager
    async def subscribe(
        self, thread_id: str, since: int | None = None
    ) -> AsyncIterator[_AgentEventStream]:
        """Subscribe to a thread's event stream.

        Replays buffered events whose ``event_id > since`` (or all events
        when ``since`` is None) before live fanout begins. The stream
        closes automatically when the thread is marked closed.

        Closes KNOWN_ISSUES.md #3 (subscribe race). Flow:
          1. ``_begin_replay()`` flips the stream into replay mode so any
             concurrent ``_emit()`` arriving via ``put_nowait`` is buffered.
          2. Register the stream in ``thread.subscribers`` BEFORE replay
             so live fanout is not lost between replay-end and registration.
          3. Snapshot ``_next_event_id`` so the replay set is bounded;
             events with ``event_id > snapshot_id`` are guaranteed to
             arrive via the now-registered live fanout.
          4. Replay via ``_replay_put`` (bypasses the live buffer).
          5. ``_end_replay()`` flushes any buffered live events in order.
        """
        thread = self._threads.get(thread_id)
        if thread is None:
            raise KeyError(thread_id)
        cursor = since if since is not None else 0
        stream = _AgentEventStream()

        stream._begin_replay()
        thread.subscribers.add(stream)
        snapshot_id = thread._next_event_id - 1
        replay_events = [
            evt for evt in thread.events
            if cursor < evt.get("event_id", 0) <= snapshot_id
        ]
        for evt in replay_events:
            stream._replay_put(evt)
        stream._end_replay()

        try:
            yield stream
        finally:
            thread.subscribers.discard(stream)
            stream.close()

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    def schedule_turn(self, thread_id: str, user_message: str) -> asyncio.Task:
        """Append ``user_message`` and schedule a non-blocking turn.

        The returned :class:`asyncio.Task` is mostly useful for tests; the
        HTTP route layer fires-and-forgets.
        """
        if thread_id not in self._threads:
            raise KeyError(thread_id)
        return asyncio.create_task(self.start_turn(thread_id, user_message))

    async def start_turn(self, thread_id: str, user_message: str) -> None:
        thread = self._threads.get(thread_id)
        if thread is None:
            raise KeyError(thread_id)
        async with thread._turn_lock:
            thread.messages.append({"role": "user", "content": user_message})
            try:
                await self._run_loop(thread)
            except Exception as exc:  # noqa: BLE001
                logger.exception("agent loop crashed")
                self._emit(
                    thread,
                    {"type": "error", "message": f"agent crashed: {exc}"},
                )
                self._emit(thread, {"type": "final", "reason": "error"})

    async def _run_loop(self, thread: _Thread) -> None:
        provider = self._provider_factory()
        if provider is None:
            self._emit(
                thread,
                {
                    "type": "error",
                    "message": "no active provider configured",
                },
            )
            self._emit(thread, {"type": "final", "reason": "no_provider"})
            return

        # O2 fix: Ollama models can reject conversations that contain
        # tool_call ids they did not themselves emit (we synthesize a UUID
        # at agent.py:618 when Ollama omits `id`). For Ollama, omit the
        # `tool_calls` array from the re-injected assistant message and
        # inject tool results as `user` turns. The synthetic UUID is still
        # used for internal correlation (frontend event matching) but
        # never sent back to the model. OpenAI and Anthropic require the
        # canonical assistant.tool_calls + role:"tool" shape.
        provider_name = getattr(provider, "name", "") or ""
        is_ollama = provider_name == "ollama"

        # Catalog tools (CLI introspection) plus the frontend-only tools.
        # Frontend tools are dispatched to the UI by the backend (proposal +
        # synthetic success) without ever shelling out, so the LLM can drive
        # the dashboard surface in addition to invoking real commands.
        canonical_tools = catalog_to_tools(self._catalog) if self._catalog else []
        canonical_tools = canonical_tools + _FRONTEND_TOOLS
        tool_calls_used = 0

        for _ in range(self.tool_budget + 1):
            current_text = []
            pending_tool_calls: list[ProviderToolCall] = []
            # Per-turn message id so the frontend can stitch text_deltas into one bubble.
            assistant_msg_id = uuid.uuid4().hex

            async for evt in provider.stream_chat(thread.messages, canonical_tools):
                if evt.kind == "text_delta" and evt.text:
                    current_text.append(evt.text)
                    self._emit(
                        thread,
                        {"type": "text_delta", "text": evt.text, "message_id": assistant_msg_id},
                    )
                elif evt.kind == "tool_call" and evt.tool_call is not None:
                    pending_tool_calls.append(evt.tool_call)
                elif evt.kind == "error":
                    # O1 fix: any provider error MUST emit a `final` so WS
                    # subscribers always get a terminal event and the UI can
                    # release its "thinking" state. This path covers httpx
                    # exceptions in OllamaProvider, SDK errors in OpenAI/
                    # Anthropic, and any future provider that yields a
                    # ProviderEvent(kind="error"). Verified: all providers
                    # in this file route their failures through this branch.
                    self._emit(
                        thread,
                        {"type": "error", "message": evt.message or "provider error"},
                    )
                    self._emit(thread, {"type": "final", "reason": "provider_error"})
                    return

            full_text = "".join(current_text)
            if full_text or pending_tool_calls:
                # Record the assistant message with tool calls (if any) so
                # the next iteration can carry the tool results back.
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": full_text,
                }
                # O2 fix: skip tool_calls re-injection for Ollama (see
                # _run_loop preamble). Other providers need the canonical
                # OpenAI-style `tool_calls` block to correlate tool results
                # in the next turn.
                if pending_tool_calls and not is_ollama:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in pending_tool_calls
                    ]
                thread.messages.append(assistant_msg)

            if not pending_tool_calls:
                self._emit(thread, {"type": "final", "reason": "complete"})
                return

            # Execute every tool call requested in this turn before looping
            # back to the provider. Hard ceiling on cumulative calls (check
            # before increment so a batch return cannot overshoot the budget).
            for tc in pending_tool_calls:
                if tool_calls_used + 1 > self.tool_budget:
                    self._emit(
                        thread,
                        {
                            "type": "final",
                            "reason": "tool budget exceeded",
                        },
                    )
                    return
                tool_calls_used += 1

                ok, stdout, exit_code = await self._execute_tool_call(thread, tc)
                # O2 fix: Ollama gets the tool result as a `user` turn
                # WITHOUT the synthetic tool_call_id (see _run_loop
                # preamble). OpenAI and Anthropic require the canonical
                # role:"tool" + tool_call_id form to correlate the result.
                if is_ollama:
                    thread.messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Tool `{tc.name}` result:\n{stdout}"
                            ),
                        }
                    )
                else:
                    thread.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": stdout,
                        }
                    )
                # B1 fix: emit both `output` and `stdout` keys with the same
                # value. The frontend prefers `output` but falls back to
                # `stdout`; emitting both is the cleanest backward-compatible
                # contract for any subscriber still on the old shape.
                self._emit(
                    thread,
                    {
                        "type": "tool_call_complete",
                        "tool_call_id": tc.id,
                        "tool": tc.name,
                        "ok": ok,
                        "output": stdout,
                        "stdout": stdout,
                        "exit_code": exit_code,
                    },
                )

        # Loop fell through without natural completion.
        self._emit(thread, {"type": "final", "reason": "tool budget exceeded"})

    async def _execute_tool_call(
        self, thread: _Thread, tc: ProviderToolCall
    ) -> tuple[bool, str, int]:
        """Run one tool call, gating on the allowlist + confirmation event.

        Returns ``(ok, stdout, exit_code)``. ``ok`` is False if the user
        rejected, the confirmation timed out, or the subprocess exited
        non-zero.
        """
        preview = _preview_command(tc.name, tc.arguments)
        self._emit(
            thread,
            {
                "type": "tool_call_proposal",
                "tool_call_id": tc.id,
                "tool": tc.name,
                "args": tc.arguments,
                "preview_cmd": preview,
            },
        )

        # Frontend-only tools (start_tour, etc.): the proposal event above
        # is the dispatch signal to the UI. The backend has no work to do
        # beyond emitting `running` and a synthetic successful completion
        # so the agent loop sees a coherent tool result and can continue.
        if tc.name in FRONTEND_ONLY_TOOL_NAMES:
            self._emit(
                thread,
                {
                    "type": "tool_call_running",
                    "tool_call_id": tc.id,
                    "tool": tc.name,
                },
            )
            return True, _frontend_tool_result(tc.name, tc.arguments), 0

        if (
            tc.name not in self.allowlist
            and tc.name not in thread.session_auto_approved
        ):
            # Reset the confirm gate for this turn and wait. Track the
            # tool_call_id so `AgentRuntime.confirm` can reject stale
            # confirmations from the UI (KNOWN_ISSUES #4 + #5). Also stash
            # the live `tc` so an `args_override` from the UI can mutate it
            # in place before we resume.
            thread.confirm_event = asyncio.Event()
            thread.confirm_decision = False
            thread.pending_tool_call_id = tc.id
            thread._pending_tool_call_obj = tc
            self._emit(
                thread,
                {
                    "type": "confirmation_required",
                    "tool_call_id": tc.id,
                    "tool": tc.name,
                    "args": tc.arguments,
                    "preview_cmd": preview,
                    "side_effects": _side_effects_for(tc.name),
                },
            )
            try:
                await asyncio.wait_for(
                    thread.confirm_event.wait(), timeout=self.confirm_timeout
                )
            except asyncio.TimeoutError:
                return False, "user did not confirm (timeout)", -1
            finally:
                # Clear regardless of outcome so a later stale confirm
                # cannot match this id again.
                if thread.pending_tool_call_id == tc.id:
                    thread.pending_tool_call_id = None
                if thread._pending_tool_call_obj is tc:
                    thread._pending_tool_call_obj = None
            if not thread.confirm_decision:
                return False, "user did not confirm", -1

        self._emit(
            thread,
            {
                "type": "tool_call_running",
                "tool_call_id": tc.id,
                "tool": tc.name,
            },
        )
        argv = _argv_for_tool(tc.name, tc.arguments)
        try:
            stdout, exit_code = await self._run_argv(argv)
        except Exception as exc:  # noqa: BLE001
            return False, f"tool execution failed: {exc}", -1
        if len(stdout) > self.tool_output_cap:
            head = stdout[: self.tool_output_cap]
            stdout = (
                head
                + f"\n[truncated; original length {len(stdout)} bytes,"
                f" cap {self.tool_output_cap}]"
            )
        return exit_code == 0, stdout, exit_code

    async def _run_argv(self, argv: list[str]) -> tuple[str, int]:
        """Run an argv list and return ``(combined_stdout, exit_code)``."""
        if self._tool_runner is not None:
            return await self._tool_runner(argv)
        if self._job_manager is None:
            raise RuntimeError(
                "AgentRuntime needs either tool_runner or job_manager to run tools"
            )
        job_id = await self._job_manager.spawn(argv)
        job = await self._job_manager.wait(job_id)
        # Combine stdout + stderr lines in order. The dashboard's terminal
        # makes the same UX choice for visual run logs.
        history = self._job_manager.history(job_id)
        text = "\n".join(line for _kind, line, _ts in history)
        return text, int(job.exit_code if job.exit_code is not None else -1)

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit(self, thread: _Thread, payload: dict[str, Any]) -> None:
        event = dict(payload)
        event.setdefault("ts", time.time())
        event["thread_id"] = thread.id
        event["event_id"] = thread._next_event_id
        thread._next_event_id += 1
        thread.events.append(event)
        for sub in list(thread.subscribers):
            with contextlib.suppress(Exception):
                sub.put_nowait(event)
        if event.get("type") == "final":
            thread.closed = True
            for sub in list(thread.subscribers):
                with contextlib.suppress(Exception):
                    sub.close()


# ---------------------------------------------------------------------------
# Provider construction helpers
# ---------------------------------------------------------------------------


def build_provider(
    name: str, record: dict[str, Any] | None
) -> BaseProvider | None:
    """Construct a provider from a credential record.

    Returns ``None`` when the record is missing required fields. The
    runtime treats this as "no active provider configured".
    """
    if record is None:
        return None
    model = record.get("model")
    if name == "openai":
        api_key = record.get("api_key")
        if not api_key or not model:
            return None
        return OpenAIProvider(api_key=api_key, model=model)
    if name == "anthropic":
        api_key = record.get("api_key")
        if not api_key or not model:
            return None
        return AnthropicProvider(api_key=api_key, model=model)
    if name == "ollama":
        if not model:
            return None
        base_url = record.get("base_url") or DEFAULT_OLLAMA_BASE_URL
        return OllamaProvider(model=model, base_url=base_url)
    return None


def make_provider_factory(
    credential_store: Any,
) -> Callable[[], Optional[BaseProvider]]:
    """Return a zero-arg callable that builds the active provider on demand."""

    def _factory() -> BaseProvider | None:
        public = credential_store.public_view()
        active = public.get("active")
        if not active:
            return None
        record = credential_store.get_provider(active)
        return build_provider(active, record)

    return _factory


__all__ = [
    "READ_ONLY_ALLOWLIST",
    "SIDE_EFFECTS",
    "DEFAULT_SIDE_EFFECTS",
    "DEFAULT_TOOL_BUDGET",
    "DEFAULT_CONFIRM_TIMEOUT",
    "DEFAULT_TOOL_OUTPUT_CAP",
    "DEFAULT_OLLAMA_BASE_URL",
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "ProviderEvent",
    "ProviderToolCall",
    "AgentRuntime",
    "catalog_to_tools",
    "build_provider",
    "make_provider_factory",
]
