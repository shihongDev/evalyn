"""Unit tests for ``evalyn_dashboard.agent`` (Phase 3 Lane C1).

Covers:
- :func:`catalog_to_tools` canonical shape + provider-specific serializers.
- :class:`AgentRuntime` happy path: text deltas, allowlisted tool calls,
  multi-turn loop, tool budget cap, confirmation gate (approve / reject /
  timeout), provider-error path.
- :class:`OpenAIProvider`, :class:`AnthropicProvider`, :class:`OllamaProvider`
  via mocked SDK objects (no real network).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from evalyn_dashboard.agent import (
    DEFAULT_CONFIRM_TIMEOUT,
    READ_ONLY_ALLOWLIST,
    AgentRuntime,
    AnthropicProvider,
    BaseProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderEvent,
    ProviderToolCall,
    _AgentEventStream,
    _argv_for_tool,
    _to_anthropic_tools,
    _to_openai_tools,
    catalog_to_tools,
)
from evalyn_dashboard.introspect import CliSchema, ParamSchema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_catalog() -> list[CliSchema]:
    return [
        CliSchema(
            id="list-runs",
            name="list-runs",
            group="Eval",
            blurb="List recent runs",
            params=[
                ParamSchema(
                    name="limit", kind="number", default=10, help="max rows"
                ),
                ParamSchema(
                    name="format",
                    kind="select",
                    default="table",
                    options=["table", "json"],
                ),
            ],
        ),
        CliSchema(
            id="run-eval",
            name="run-eval",
            group="Eval",
            blurb="Run evaluation",
            params=[
                ParamSchema(
                    name="dataset", kind="path", required=True, help="dataset path"
                ),
                ParamSchema(
                    name="dry_run", kind="bool", default=False
                ),
                ParamSchema(
                    name="tags",
                    kind="multiselect",
                    options=["fast", "slow"],
                    default=[],
                ),
            ],
        ),
    ]


class MockProvider(BaseProvider):
    """Yields a canned, scripted sequence per turn.

    Construction takes a list of "turns"; each turn is a list of
    :class:`ProviderEvent`. The provider replays one turn per call to
    :meth:`stream_chat` and StopIteration raises if exhausted.
    """

    name = "mock"

    def __init__(self, turns: list[list[ProviderEvent]]) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    async def stream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ProviderEvent]:
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        if not self._turns:
            yield ProviderEvent(kind="finish")
            return
        turn = self._turns.pop(0)
        for evt in turn:
            yield evt


@dataclass
class _FakeJob:
    exit_code: int = 0


class _FakeJobManager:
    """In-process tool runner that records argv and returns canned output."""

    def __init__(self) -> None:
        self.spawned: list[list[str]] = []
        self._next_id = 0
        self._stdout: dict[str, str] = {}
        self._exit: dict[str, int] = {}

    def queue(self, *, stdout: str = "ok", exit_code: int = 0) -> None:
        # Pre-program the next spawn's response.
        self._next_stdout = stdout
        self._next_exit = exit_code

    async def spawn(self, cmd: list[str]) -> str:
        self.spawned.append(list(cmd))
        self._next_id += 1
        jid = f"job-{self._next_id}"
        self._stdout[jid] = getattr(self, "_next_stdout", "ok")
        self._exit[jid] = getattr(self, "_next_exit", 0)
        return jid

    async def wait(self, job_id: str, timeout: float | None = None) -> _FakeJob:
        return _FakeJob(exit_code=self._exit[job_id])

    def history(self, job_id: str) -> list[tuple[str, str, float]]:
        return [("stdout", self._stdout[job_id], 0.0)]


# ---------------------------------------------------------------------------
# catalog_to_tools
# ---------------------------------------------------------------------------


def test_catalog_to_tools_canonical_shape() -> None:
    catalog = _sample_catalog()
    tools = catalog_to_tools(catalog)
    assert {t["name"] for t in tools} == {"list-runs", "run-eval"}

    list_runs = next(t for t in tools if t["name"] == "list-runs")
    assert list_runs["description"] == "List recent runs"
    params = list_runs["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["limit"]["type"] == "number"
    assert params["properties"]["format"]["type"] == "string"
    assert params["properties"]["format"]["enum"] == ["table", "json"]


def test_catalog_to_tools_required_propagates() -> None:
    tools = catalog_to_tools(_sample_catalog())
    run = next(t for t in tools if t["name"] == "run-eval")
    assert run["parameters"]["required"] == ["dataset"]
    assert run["parameters"]["properties"]["dry_run"]["type"] == "boolean"
    assert run["parameters"]["properties"]["tags"]["type"] == "array"
    assert run["parameters"]["properties"]["tags"]["items"]["enum"] == [
        "fast",
        "slow",
    ]


def test_to_openai_tools_wraps_function() -> None:
    canonical = catalog_to_tools(_sample_catalog())
    out = _to_openai_tools(canonical)
    assert all(t["type"] == "function" for t in out)
    assert all("parameters" in t["function"] for t in out)


def test_to_anthropic_tools_uses_input_schema() -> None:
    canonical = catalog_to_tools(_sample_catalog())
    out = _to_anthropic_tools(canonical)
    assert all("input_schema" in t for t in out)
    assert all("parameters" not in t for t in out)


# ---------------------------------------------------------------------------
# _argv_for_tool
# ---------------------------------------------------------------------------


def test_argv_for_tool_renders_flags_correctly() -> None:
    argv = _argv_for_tool(
        "run-eval",
        {"dataset": "ds.json", "dry_run": True, "tags": ["fast"], "limit": 5},
    )
    assert argv[:2] == ["evalyn", "run-eval"]
    # Order of dict keys is preserved in Python 3.7+, but we accept any order
    # by checking presence + adjacency for list args.
    assert "--dataset" in argv
    di = argv.index("--dataset")
    assert argv[di + 1] == "ds.json"
    assert "--dry-run" in argv
    assert "--tags" in argv
    ti = argv.index("--tags")
    assert argv[ti + 1] == "fast"


def test_argv_for_tool_skips_false_bools_and_empty() -> None:
    argv = _argv_for_tool(
        "x",
        {"flag": False, "name": "", "tags": [], "value": None},
    )
    assert argv == ["evalyn", "x"]


# ---------------------------------------------------------------------------
# AgentRuntime: simple turn (text-only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_text_turn_emits_text_then_final() -> None:
    provider = MockProvider(
        [
            [
                ProviderEvent(kind="text_delta", text="Hello "),
                ProviderEvent(kind="text_delta", text="world."),
                ProviderEvent(kind="finish"),
            ]
        ]
    )
    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    thread_id = runtime.create_thread()

    received: list[dict[str, Any]] = []

    async def collect() -> None:
        async with runtime.subscribe(thread_id) as stream:
            async for evt in stream:
                received.append(evt)
                if evt["type"] == "final":
                    return

    consumer = asyncio.create_task(collect())
    await runtime.start_turn(thread_id, "say hi")
    await asyncio.wait_for(consumer, timeout=2.0)

    assert [e["type"] for e in received] == ["text_delta", "text_delta", "final"]
    assert received[0]["text"] == "Hello "
    assert received[2]["reason"] == "complete"


# ---------------------------------------------------------------------------
# AgentRuntime: read-only tool auto-runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_only_tool_auto_runs() -> None:
    tc = ProviderToolCall(id="tc-1", name="list-runs", arguments={"limit": 5})
    provider = MockProvider(
        [
            [
                ProviderEvent(kind="text_delta", text="Looking up runs."),
                ProviderEvent(kind="tool_call", tool_call=tc),
                ProviderEvent(kind="finish"),
            ],
            [
                ProviderEvent(kind="text_delta", text="Found 5."),
                ProviderEvent(kind="finish"),
            ],
        ]
    )
    spawned: list[list[str]] = []

    async def runner(argv: list[str]) -> tuple[str, int]:
        spawned.append(argv)
        return "id\trun-a\nid\trun-b\n", 0

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=runner,
    )
    thread_id = runtime.create_thread()
    await runtime.start_turn(thread_id, "list runs")

    types = [e["type"] for e in runtime._threads[thread_id].events]
    assert "tool_call_proposal" in types
    assert "tool_call_running" in types
    assert "tool_call_complete" in types
    assert types[-1] == "final"

    assert spawned == [["evalyn", "list-runs", "--limit", "5"]]
    # The provider should have been called twice (initial turn + after tool).
    assert len(provider.calls) == 2


# ---------------------------------------------------------------------------
# AgentRuntime: confirmation gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_destructive_tool_requires_confirmation_approve() -> None:
    tc = ProviderToolCall(
        id="tc-1", name="run-eval", arguments={"dataset": "ds.json"}
    )
    provider = MockProvider(
        [
            [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="text_delta", text="done"), ProviderEvent(kind="finish")],
        ]
    )

    async def runner(argv: list[str]) -> tuple[str, int]:
        return "ran", 0

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=runner,
        confirm_timeout=2.0,
    )
    thread_id = runtime.create_thread()

    task = asyncio.create_task(runtime.start_turn(thread_id, "run it"))
    # Wait for the confirmation_required event then approve.
    for _ in range(50):
        if any(
            e["type"] == "confirmation_required"
            for e in runtime._threads[thread_id].events
        ):
            break
        await asyncio.sleep(0.02)
    assert runtime.confirm(thread_id, approve=True)
    await asyncio.wait_for(task, timeout=3.0)

    types = [e["type"] for e in runtime._threads[thread_id].events]
    assert "confirmation_required" in types
    assert "tool_call_running" in types
    assert "tool_call_complete" in types
    final = next(e for e in runtime._threads[thread_id].events if e["type"] == "final")
    assert final["reason"] == "complete"


@pytest.mark.asyncio
async def test_destructive_tool_rejected_returns_user_did_not_confirm() -> None:
    tc = ProviderToolCall(
        id="tc-1", name="run-eval", arguments={"dataset": "ds.json"}
    )
    provider = MockProvider(
        [
            [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="text_delta", text="ok"), ProviderEvent(kind="finish")],
        ]
    )

    async def runner(argv: list[str]) -> tuple[str, int]:  # pragma: no cover - never invoked
        raise AssertionError("rejected tools must not run")

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=runner,
        confirm_timeout=2.0,
    )
    thread_id = runtime.create_thread()

    task = asyncio.create_task(runtime.start_turn(thread_id, "run it"))
    for _ in range(50):
        if any(
            e["type"] == "confirmation_required"
            for e in runtime._threads[thread_id].events
        ):
            break
        await asyncio.sleep(0.02)
    runtime.confirm(thread_id, approve=False)
    await asyncio.wait_for(task, timeout=3.0)

    complete = next(
        e for e in runtime._threads[thread_id].events if e["type"] == "tool_call_complete"
    )
    assert complete["ok"] is False
    assert "user did not confirm" in complete["stdout"]


@pytest.mark.asyncio
async def test_confirmation_timeout_returns_did_not_confirm() -> None:
    tc = ProviderToolCall(
        id="tc-1", name="run-eval", arguments={"dataset": "ds.json"}
    )
    provider = MockProvider(
        [
            [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="text_delta", text="ok"), ProviderEvent(kind="finish")],
        ]
    )
    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
        confirm_timeout=0.1,
    )
    thread_id = runtime.create_thread()
    await runtime.start_turn(thread_id, "run it")

    complete = next(
        e for e in runtime._threads[thread_id].events if e["type"] == "tool_call_complete"
    )
    assert complete["ok"] is False
    assert "timeout" in complete["stdout"].lower()


# ---------------------------------------------------------------------------
# AgentRuntime: tool budget cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_budget_exceeded_emits_final() -> None:
    """The provider keeps proposing read-only tool calls forever; the
    runtime should bail after the configured budget."""
    tc = ProviderToolCall(id="tc", name="list-runs", arguments={})

    def _make_turn() -> list[ProviderEvent]:
        return [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")]

    provider = MockProvider([_make_turn() for _ in range(20)])

    async def runner(argv: list[str]) -> tuple[str, int]:
        return "ok", 0

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=runner,
        tool_budget=3,
    )
    thread_id = runtime.create_thread()
    await runtime.start_turn(thread_id, "loop")

    finals = [e for e in runtime._threads[thread_id].events if e["type"] == "final"]
    assert finals
    assert "tool budget" in finals[-1]["reason"]


# ---------------------------------------------------------------------------
# AgentRuntime: provider error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_error_emits_error_then_final() -> None:
    provider = MockProvider(
        [[ProviderEvent(kind="error", message="rate limited")]]
    )
    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    thread_id = runtime.create_thread()
    await runtime.start_turn(thread_id, "hi")

    types = [e["type"] for e in runtime._threads[thread_id].events]
    assert "error" in types
    assert types[-1] == "final"


@pytest.mark.asyncio
async def test_no_provider_emits_error_final() -> None:
    runtime = AgentRuntime(
        provider_factory=lambda: None,
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    thread_id = runtime.create_thread()
    await runtime.start_turn(thread_id, "hi")
    events = runtime._threads[thread_id].events
    assert events[0]["type"] == "error"
    assert events[-1]["type"] == "final"


# ---------------------------------------------------------------------------
# AgentRuntime: tool output truncation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_output_truncated_at_cap() -> None:
    big = "x" * 1000
    tc = ProviderToolCall(id="tc", name="list-runs", arguments={})
    provider = MockProvider(
        [
            [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="finish")],
        ]
    )

    async def runner(argv: list[str]) -> tuple[str, int]:
        return big, 0

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=runner,
        tool_output_cap=100,
    )
    thread_id = runtime.create_thread()
    await runtime.start_turn(thread_id, "go")

    complete = next(
        e for e in runtime._threads[thread_id].events if e["type"] == "tool_call_complete"
    )
    assert "[truncated" in complete["stdout"]
    assert complete["stdout"].startswith("x" * 100)


# ---------------------------------------------------------------------------
# AgentRuntime: subscribe replay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_replays_buffered_events() -> None:
    provider = MockProvider(
        [
            [
                ProviderEvent(kind="text_delta", text="hi"),
                ProviderEvent(kind="finish"),
            ]
        ]
    )
    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    thread_id = runtime.create_thread()
    await runtime.start_turn(thread_id, "hi")

    received: list[dict[str, Any]] = []
    async with runtime.subscribe(thread_id) as stream:
        async for evt in stream:
            received.append(evt)
            if evt["type"] == "final":
                break
    assert [e["type"] for e in received] == ["text_delta", "final"]


# ---------------------------------------------------------------------------
# Regression: KNOWN_ISSUES.md #3 (AgentRuntime.subscribe race)
# ---------------------------------------------------------------------------


def test_agent_event_stream_replay_buffer_orders_live_after_flush() -> None:
    """Direct unit test of _AgentEventStream.{_begin,_end}_replay buffering.

    During replay, put_nowait events buffer and the queue stays empty.
    _end_replay flushes them to the queue in original emit order.
    After end_replay, put_nowait goes straight to the queue.
    """
    s = _AgentEventStream()
    s._begin_replay()
    s.put_nowait({"type": "text_delta", "text": "live1", "event_id": 10})
    s.put_nowait({"type": "text_delta", "text": "live2", "event_id": 11})
    assert s._queue.empty(), "live events must buffer during replay"

    s._end_replay()
    drained = []
    while not s._queue.empty():
        drained.append(s._queue.get_nowait())
    assert [d["text"] for d in drained] == ["live1", "live2"]

    s.put_nowait({"type": "final", "event_id": 12})
    assert s._queue.get_nowait()["type"] == "final"


@pytest.mark.asyncio
async def test_subscribe_registers_before_replay() -> None:
    """KNOWN_ISSUES.md #3 regression. Verify the new invariant: the
    subscriber stream is registered in ``thread.subscribers`` BEFORE
    any replay event is enqueued. Previously, registration happened
    after replay so live ``_emit`` calls between the last replay and
    registration would be silently dropped on the new subscriber.
    """
    provider = MockProvider(
        [
            [
                ProviderEvent(kind="text_delta", text="hello"),
                ProviderEvent(kind="finish"),
            ]
        ]
    )
    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    thread_id = runtime.create_thread()
    await runtime.start_turn(thread_id, "hi")

    thread = runtime._threads[thread_id]
    pre_count = len(thread.subscribers)

    received: list[dict[str, Any]] = []
    async with runtime.subscribe(thread_id) as stream:
        # By the time __aenter__ has yielded, registration must be done.
        # The old code did registration AFTER replay; the fix moves it
        # BEFORE. Closes the gap KNOWN_ISSUES.md #3 documents.
        assert stream in thread.subscribers, (
            "subscribe() must register the stream before yielding"
        )
        # Sanity: replay still delivered the buffered events.
        async for evt in stream:
            received.append(evt)
            if evt["type"] == "final":
                break

    assert any(e["type"] == "text_delta" for e in received)
    assert received[-1]["type"] == "final"
    # Cleanup happened on context exit.
    assert len(thread.subscribers) == pre_count


# ---------------------------------------------------------------------------
# Read-only allowlist sanity
# ---------------------------------------------------------------------------


def test_allowlist_contains_expected_commands() -> None:
    expected = {
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
    assert expected == set(READ_ONLY_ALLOWLIST)


def test_destructive_commands_not_in_allowlist() -> None:
    for cmd in [
        "run-eval",
        "calibrate",
        "delete-traces",
        "build-dataset",
        "annotate",
        "import-annotations",
        "simulate",
        "one-click",
        "export",
        "export-for-annotation",
        "init",
        "quickstart",
        "report",
        "dashboard",
        "suggest-metrics",
    ]:
        assert cmd not in READ_ONLY_ALLOWLIST, f"{cmd} must require confirmation"


# ---------------------------------------------------------------------------
# Provider serialization / parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_provider_yields_text_and_tool_call() -> None:
    """OpenAI streams deltas; the provider aggregates fragments into one tool call."""
    fake_chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="hello ", tool_calls=None))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="world", tool_calls=None))]),
        MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(
                        content=None,
                        tool_calls=[
                            MagicMock(
                                index=0,
                                id="call-1",
                                function=MagicMock(name="list-runs", arguments='{"limit"'),
                            )
                        ],
                    )
                )
            ]
        ),
        MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(
                        content=None,
                        tool_calls=[
                            MagicMock(
                                index=0,
                                id=None,
                                function=MagicMock(name=None, arguments=":5}"),
                            )
                        ],
                    )
                )
            ]
        ),
    ]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter(fake_chunks)
    # MagicMock auto-mocks attribute access; we need ``function.name`` to
    # equal the string "list-runs" not a MagicMock.
    fake_chunks[2].choices[0].delta.tool_calls[0].function.name = "list-runs"
    fake_chunks[2].choices[0].delta.tool_calls[0].function.arguments = '{"limit"'
    fake_chunks[3].choices[0].delta.tool_calls[0].function.name = None
    fake_chunks[3].choices[0].delta.tool_calls[0].function.arguments = ":5}"

    provider = OpenAIProvider(api_key="sk", model="gpt-x", client=fake_client)
    events: list[ProviderEvent] = []
    async for evt in provider.stream_chat([{"role": "user", "content": "hi"}], []):
        events.append(evt)

    text = "".join(e.text or "" for e in events if e.kind == "text_delta")
    assert text == "hello world"
    tcalls = [e for e in events if e.kind == "tool_call"]
    assert len(tcalls) == 1
    assert tcalls[0].tool_call is not None
    assert tcalls[0].tool_call.name == "list-runs"
    assert tcalls[0].tool_call.arguments == {"limit": 5}
    assert events[-1].kind == "finish"


@pytest.mark.asyncio
async def test_ollama_provider_parses_streamed_lines() -> None:
    """Stub httpx.AsyncClient that yields canned JSON lines."""
    payload_lines = [
        json.dumps({"message": {"content": "hi "}, "done": False}),
        json.dumps(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "list-runs",
                                "arguments": {"limit": 3},
                            }
                        }
                    ],
                },
                "done": False,
            }
        ),
        json.dumps({"message": {"content": "done"}, "done": True}),
    ]

    class _StubResponse:
        def __init__(self, lines: list[str]) -> None:
            self._lines = lines

        def raise_for_status(self) -> None:
            pass

        async def aiter_lines(self) -> AsyncIterator[str]:
            for line in self._lines:
                yield line

        async def __aenter__(self) -> "_StubResponse":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            pass

    class _StubClient:
        def __init__(self, lines: list[str]) -> None:
            self._lines = lines
            self.calls: list[dict[str, Any]] = []

        def stream(self, method: str, url: str, **kwargs: Any) -> _StubResponse:
            self.calls.append({"method": method, "url": url, "json": kwargs.get("json")})
            return _StubResponse(self._lines)

    stub = _StubClient(payload_lines)
    provider = OllamaProvider(model="llama3", base_url="http://x:11434", client=stub)

    events: list[ProviderEvent] = []
    async for evt in provider.stream_chat([{"role": "user", "content": "hi"}], []):
        events.append(evt)

    text = "".join(e.text or "" for e in events if e.kind == "text_delta")
    assert text == "hi done"
    tcalls = [e for e in events if e.kind == "tool_call"]
    assert len(tcalls) == 1
    assert tcalls[0].tool_call is not None
    assert tcalls[0].tool_call.name == "list-runs"
    assert tcalls[0].tool_call.arguments == {"limit": 3}
    assert events[-1].kind == "finish"
    assert stub.calls[0]["url"] == "http://x:11434/api/chat"


# ---------------------------------------------------------------------------
# AgentRuntime + JobManager wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_uses_job_manager_when_no_runner_provided() -> None:
    tc = ProviderToolCall(id="tc", name="list-runs", arguments={"limit": 3})
    provider = MockProvider(
        [
            [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="text_delta", text="ok"), ProviderEvent(kind="finish")],
        ]
    )
    jm = _FakeJobManager()
    jm.queue(stdout="row1\nrow2", exit_code=0)
    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        job_manager=jm,
    )
    thread_id = runtime.create_thread()
    await runtime.start_turn(thread_id, "go")

    assert jm.spawned == [["evalyn", "list-runs", "--limit", "3"]]
    complete = next(
        e for e in runtime._threads[thread_id].events if e["type"] == "tool_call_complete"
    )
    assert "row1" in complete["stdout"]


# ---------------------------------------------------------------------------
# AgentRuntime: confirmation upgrades (P1 §5.5 - editable args + session
# auto-approve + side-effects + stale id rejection)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirmation_required_includes_side_effects() -> None:
    """`confirmation_required` events carry the SIDE_EFFECTS bullets so the
    UI can render the "THIS WILL" copy without looking up tool metadata."""
    tc = ProviderToolCall(
        id="tc-1", name="run-eval", arguments={"dataset": "ds.json"}
    )
    provider = MockProvider(
        [
            [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="text_delta", text="ok"), ProviderEvent(kind="finish")],
        ]
    )

    async def runner(argv: list[str]) -> tuple[str, int]:
        return "ran", 0

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=runner,
        confirm_timeout=2.0,
    )
    thread_id = runtime.create_thread()
    task = asyncio.create_task(runtime.start_turn(thread_id, "run it"))
    for _ in range(50):
        if any(
            e["type"] == "confirmation_required"
            for e in runtime._threads[thread_id].events
        ):
            break
        await asyncio.sleep(0.02)
    runtime.confirm(thread_id, approve=True, tool_call_id="tc-1")
    await asyncio.wait_for(task, timeout=3.0)

    confirm_evt = next(
        e
        for e in runtime._threads[thread_id].events
        if e["type"] == "confirmation_required"
    )
    assert "side_effects" in confirm_evt
    bullets = confirm_evt["side_effects"]
    assert isinstance(bullets, list) and len(bullets) >= 1
    # run-eval has a curated entry in the SIDE_EFFECTS dict.
    assert any("run-eval" in b for b in bullets)


@pytest.mark.asyncio
async def test_args_override_changes_executed_argv() -> None:
    """When the user edits args inline before approving, the runtime must
    spawn the *edited* argv, not the agent's original."""
    tc = ProviderToolCall(
        id="tc-1", name="run-eval", arguments={"dataset": "wrong.jsonl"}
    )
    provider = MockProvider(
        [
            [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="text_delta", text="done"), ProviderEvent(kind="finish")],
        ]
    )
    spawn_calls: list[list[str]] = []

    async def runner(argv: list[str]) -> tuple[str, int]:
        spawn_calls.append(argv)
        return "ran", 0

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=runner,
        confirm_timeout=2.0,
    )
    thread_id = runtime.create_thread()
    task = asyncio.create_task(runtime.start_turn(thread_id, "run it"))
    for _ in range(50):
        if any(
            e["type"] == "confirmation_required"
            for e in runtime._threads[thread_id].events
        ):
            break
        await asyncio.sleep(0.02)
    # Override only the EXISTING key the model proposed. Adding new
    # keys via args_override is intentionally rejected — see the
    # `test_args_override_rejects_new_keys` test below.
    accepted = runtime.confirm(
        thread_id,
        approve=True,
        tool_call_id="tc-1",
        args_override={"dataset": "evals/correct.jsonl"},
    )
    assert accepted is True
    await asyncio.wait_for(task, timeout=3.0)

    assert spawn_calls, "runner was never invoked"
    argv = spawn_calls[0]
    # Original arg replaced by edited value.
    assert "wrong.jsonl" not in argv
    assert "evals/correct.jsonl" in argv


@pytest.mark.asyncio
async def test_args_override_rejects_new_keys() -> None:
    """args_override may only edit values for keys the model proposed.

    Adding a new key (e.g. ``--unsafe-flag`` after a benign confirmation
    card) would let a confused-deputy attack escalate privileges past
    the original allow-list check. The runtime must refuse such overrides.
    """
    tc = ProviderToolCall(
        id="tc-1", name="run-eval", arguments={"dataset": "wrong.jsonl"}
    )
    provider = MockProvider(
        [
            [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="text_delta", text="done"), ProviderEvent(kind="finish")],
        ]
    )
    spawn_calls: list[list[str]] = []

    async def runner(argv: list[str]) -> tuple[str, int]:
        spawn_calls.append(argv)
        return "ran", 0

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=runner,
        confirm_timeout=2.0,
    )
    thread_id = runtime.create_thread()
    asyncio.create_task(runtime.start_turn(thread_id, "run it"))
    for _ in range(50):
        if any(
            e["type"] == "confirmation_required"
            for e in runtime._threads[thread_id].events
        ):
            break
        await asyncio.sleep(0.02)
    # Add `dry_run` — a key the model did NOT propose. confirm() must
    # return False without setting the gate, so the agent stays paused.
    accepted = runtime.confirm(
        thread_id,
        approve=True,
        tool_call_id="tc-1",
        args_override={"dataset": "evals/correct.jsonl", "dry_run": True},
    )
    assert accepted is False
    # Subprocess must NOT have run since the confirmation was refused.
    assert spawn_calls == []


@pytest.mark.asyncio
async def test_auto_approve_session_skips_gate_for_same_tool() -> None:
    """After `auto_approve_session=True` for run-eval, a *second* run-eval
    call in the same thread executes without a fresh confirmation gate."""
    tc1 = ProviderToolCall(
        id="tc-1", name="run-eval", arguments={"dataset": "a.jsonl"}
    )
    tc2 = ProviderToolCall(
        id="tc-2", name="run-eval", arguments={"dataset": "b.jsonl"}
    )
    provider = MockProvider(
        [
            [ProviderEvent(kind="tool_call", tool_call=tc1), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="tool_call", tool_call=tc2), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="text_delta", text="done"), ProviderEvent(kind="finish")],
        ]
    )
    spawn_calls: list[list[str]] = []

    async def runner(argv: list[str]) -> tuple[str, int]:
        spawn_calls.append(argv)
        return "ok", 0

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=runner,
        confirm_timeout=2.0,
    )
    thread_id = runtime.create_thread()

    task = asyncio.create_task(runtime.start_turn(thread_id, "run it twice"))
    # First call: wait for the gate, approve with auto_approve_session=True.
    for _ in range(50):
        if any(
            e["type"] == "confirmation_required"
            for e in runtime._threads[thread_id].events
        ):
            break
        await asyncio.sleep(0.02)
    runtime.confirm(
        thread_id,
        approve=True,
        tool_call_id="tc-1",
        auto_approve_session=True,
    )
    await asyncio.wait_for(task, timeout=3.0)

    # The session whitelist now contains run-eval.
    assert "run-eval" in runtime._threads[thread_id].session_auto_approved
    # Both calls were executed; only ONE confirmation_required event should
    # have been emitted (the second call bypassed the gate).
    confirm_events = [
        e
        for e in runtime._threads[thread_id].events
        if e["type"] == "confirmation_required"
    ]
    assert len(confirm_events) == 1
    assert len(spawn_calls) == 2
    assert spawn_calls[0] == ["evalyn", "run-eval", "--dataset", "a.jsonl"]
    assert spawn_calls[1] == ["evalyn", "run-eval", "--dataset", "b.jsonl"]


@pytest.mark.asyncio
async def test_stale_tool_call_id_still_rejected() -> None:
    """Regression test for KNOWN_ISSUES #4: a confirm POST with the wrong
    tool_call_id must NOT release the gate, even with the new override
    parameters in play."""
    tc = ProviderToolCall(
        id="tc-real", name="run-eval", arguments={"dataset": "ds.json"}
    )
    provider = MockProvider(
        [
            [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="text_delta", text="ok"), ProviderEvent(kind="finish")],
        ]
    )

    async def runner(argv: list[str]) -> tuple[str, int]:
        return "ran", 0

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=runner,
        confirm_timeout=0.3,
    )
    thread_id = runtime.create_thread()
    task = asyncio.create_task(runtime.start_turn(thread_id, "run it"))
    for _ in range(50):
        if any(
            e["type"] == "confirmation_required"
            for e in runtime._threads[thread_id].events
        ):
            break
        await asyncio.sleep(0.02)

    # Stale id - even with args_override + auto_approve_session - must be
    # refused and leave the gate untouched.
    accepted = runtime.confirm(
        thread_id,
        approve=True,
        tool_call_id="tc-OLD",
        args_override={"dataset": "x.jsonl"},
        auto_approve_session=True,
    )
    assert accepted is False
    assert "run-eval" not in runtime._threads[thread_id].session_auto_approved
    # Pending tool args still untouched.
    pending_obj = runtime._threads[thread_id]._pending_tool_call_obj  # type: ignore[attr-defined]
    assert pending_obj is not None
    assert pending_obj.arguments == {"dataset": "ds.json"}

    # Let the gate time out so the task completes.
    await asyncio.wait_for(task, timeout=3.0)
    complete = next(
        e for e in runtime._threads[thread_id].events if e["type"] == "tool_call_complete"
    )
    assert complete["ok"] is False
    assert "timeout" in complete["stdout"].lower()


@pytest.mark.asyncio
async def test_args_override_with_rejection_does_nothing() -> None:
    """approve=False short-circuits any override - rejection wins, the tool
    must NOT run regardless of args_override / auto_approve_session."""
    tc = ProviderToolCall(
        id="tc-1", name="run-eval", arguments={"dataset": "ds.json"}
    )
    provider = MockProvider(
        [
            [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="text_delta", text="ok"), ProviderEvent(kind="finish")],
        ]
    )

    async def runner(argv: list[str]) -> tuple[str, int]:  # pragma: no cover
        raise AssertionError("rejected tools must not run")

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=runner,
        confirm_timeout=2.0,
    )
    thread_id = runtime.create_thread()
    task = asyncio.create_task(runtime.start_turn(thread_id, "run it"))
    for _ in range(50):
        if any(
            e["type"] == "confirmation_required"
            for e in runtime._threads[thread_id].events
        ):
            break
        await asyncio.sleep(0.02)
    accepted = runtime.confirm(
        thread_id,
        approve=False,
        tool_call_id="tc-1",
        args_override={"dataset": "ignored.jsonl"},
        auto_approve_session=True,
    )
    assert accepted is True  # confirm itself accepts the message
    await asyncio.wait_for(task, timeout=3.0)

    # Whitelist NOT mutated on rejection.
    assert "run-eval" not in runtime._threads[thread_id].session_auto_approved
    complete = next(
        e for e in runtime._threads[thread_id].events if e["type"] == "tool_call_complete"
    )
    assert complete["ok"] is False
    assert "user did not confirm" in complete["stdout"]


# ---------------------------------------------------------------------------
# Bug-fix regression tests (B1, O1, O2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b1_tool_call_complete_emits_both_output_and_stdout() -> None:
    """B1: tool_call_complete payload must include BOTH `output` and `stdout`
    keys with the same value. The frontend prefers `output` but falls back to
    `stdout`; emitting both keeps old subscribers working while the new
    contract takes effect."""
    tc = ProviderToolCall(id="tc-1", name="list-runs", arguments={"limit": 5})
    provider = MockProvider(
        [
            [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="text_delta", text="done"), ProviderEvent(kind="finish")],
        ]
    )

    async def runner(argv: list[str]) -> tuple[str, int]:
        return "row1\nrow2", 0

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=runner,
    )
    thread_id = runtime.create_thread()
    await runtime.start_turn(thread_id, "list runs")

    complete = next(
        e
        for e in runtime._threads[thread_id].events
        if e["type"] == "tool_call_complete"
    )
    assert "output" in complete, "tool_call_complete must include `output` key"
    assert "stdout" in complete, "tool_call_complete must include `stdout` key"
    assert complete["output"] == complete["stdout"] == "row1\nrow2"


@pytest.mark.asyncio
async def test_o1_ollama_httpx_error_yields_final_event() -> None:
    """O1: when the OllamaProvider raises an httpx exception, the runtime
    must still emit a `final` event so WS subscribers see a terminal event.
    Verifies the `evt.kind == "error"` branch of `_run_loop` correctly
    routes provider errors to error+final."""
    import httpx

    class _RaisingClient:
        def stream(self, method: str, url: str, **kwargs: Any) -> Any:
            raise httpx.ConnectError("connection refused")

    provider = OllamaProvider(
        model="llama3", base_url="http://localhost:11434", client=_RaisingClient()
    )
    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    thread_id = runtime.create_thread()

    received: list[dict[str, Any]] = []

    async def collect() -> None:
        async with runtime.subscribe(thread_id) as stream:
            async for evt in stream:
                received.append(evt)
                if evt["type"] == "final":
                    return

    consumer = asyncio.create_task(collect())
    await runtime.start_turn(thread_id, "hi")
    await asyncio.wait_for(consumer, timeout=3.0)

    types = [e["type"] for e in received]
    assert "error" in types
    assert types[-1] == "final"
    final = next(e for e in received if e["type"] == "final")
    assert final["reason"] == "provider_error"


@pytest.mark.asyncio
async def test_o2_ollama_does_not_reinject_synthetic_tool_call_id() -> None:
    """O2: for the Ollama provider, the next-turn message_history must NOT
    include the synthetic tool_call_id that the runtime fabricated when
    Ollama omitted `tool_calls[].id`. The synthetic UUID is kept for
    internal correlation (frontend matching) but never echoed back to the
    model. Tool results are injected as `user` turns instead of role:tool."""
    synthetic_id = "synthetic-uuid-1234"
    tc = ProviderToolCall(id=synthetic_id, name="list-runs", arguments={"limit": 3})

    class _OllamaShape(MockProvider):
        name = "ollama"

    provider = _OllamaShape(
        [
            [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="text_delta", text="done"), ProviderEvent(kind="finish")],
        ]
    )

    async def runner(argv: list[str]) -> tuple[str, int]:
        return "row1", 0

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=runner,
    )
    thread_id = runtime.create_thread()
    await runtime.start_turn(thread_id, "list runs")

    # Inspect the messages the provider received on its SECOND call (the
    # post-tool turn). The synthetic UUID must not appear anywhere in the
    # serialized message history.
    assert len(provider.calls) >= 2, "provider should be called twice (pre + post tool)"
    second_messages = provider.calls[1]["messages"]
    serialized = json.dumps(second_messages)
    assert synthetic_id not in serialized, (
        f"Ollama re-injection leaked synthetic tool_call_id: {serialized}"
    )

    # Assistant message should NOT carry tool_calls for Ollama.
    assistant_msgs = [m for m in second_messages if m.get("role") == "assistant"]
    assert assistant_msgs, "expected an assistant message in history"
    assert all("tool_calls" not in m for m in assistant_msgs), (
        "Ollama assistant message must not include synthetic tool_calls"
    )

    # Tool result should be injected as a user turn (no role:"tool"), and
    # must NOT include a tool_call_id field.
    user_msgs = [m for m in second_messages if m.get("role") == "user"]
    assert any("row1" in (m.get("content") or "") for m in user_msgs), (
        "tool result should be injected as a user turn"
    )
    tool_role_msgs = [m for m in second_messages if m.get("role") == "tool"]
    assert not tool_role_msgs, (
        "Ollama path must not emit role:'tool' messages"
    )
    for m in second_messages:
        assert "tool_call_id" not in m, (
            f"no message in Ollama history should carry tool_call_id: {m}"
        )


@pytest.mark.asyncio
async def test_o2_openai_path_still_uses_canonical_tool_call_shape() -> None:
    """O2 regression guard: non-Ollama providers MUST still use the
    canonical OpenAI-style tool_calls + role:'tool' shape so the existing
    contract with OpenAI/Anthropic SDKs is preserved."""
    tc = ProviderToolCall(id="tc-real", name="list-runs", arguments={"limit": 3})

    class _OpenAIShape(MockProvider):
        name = "openai"

    provider = _OpenAIShape(
        [
            [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")],
            [ProviderEvent(kind="text_delta", text="done"), ProviderEvent(kind="finish")],
        ]
    )

    async def runner(argv: list[str]) -> tuple[str, int]:
        return "row1", 0

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=_sample_catalog(),
        tool_runner=runner,
    )
    thread_id = runtime.create_thread()
    await runtime.start_turn(thread_id, "list runs")

    second_messages = provider.calls[1]["messages"]
    assistant = next(m for m in second_messages if m.get("role") == "assistant")
    assert "tool_calls" in assistant
    assert assistant["tool_calls"][0]["id"] == "tc-real"
    tool_msg = next(m for m in second_messages if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "tc-real"


# ---------------------------------------------------------------------------
# Frontend-only tools (start_tour)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_tour_short_circuits_without_running_argv() -> None:
    """The agent's start_tour tool is dispatched to the frontend via the
    proposal event; the backend MUST emit running + complete with a
    synthetic success and never invoke the tool runner. Verified by using
    `_unused_runner` (asserts on call) as the seam."""
    tc = ProviderToolCall(
        id="tc-tour-1",
        name="start_tour",
        arguments={"tour_id": "runEval.v1"},
    )
    provider = MockProvider(
        [
            [
                ProviderEvent(kind="text_delta", text="Walking you through it."),
                ProviderEvent(kind="tool_call", tool_call=tc),
                ProviderEvent(kind="finish"),
            ],
            [
                ProviderEvent(kind="text_delta", text="Tour started."),
                ProviderEvent(kind="finish"),
            ],
        ]
    )

    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=[],  # frontend-only tools are appended automatically
        tool_runner=_unused_runner,
    )
    thread_id = runtime.create_thread()
    await runtime.start_turn(thread_id, "show me how to run an eval")

    events = runtime._threads[thread_id].events
    types = [e["type"] for e in events]

    # The full lifecycle still emits, so the frontend sees the proposal
    # (which is its dispatch signal) followed by running + complete that
    # close the bubble cleanly.
    assert "tool_call_proposal" in types
    assert "tool_call_running" in types
    assert "tool_call_complete" in types
    assert types[-1] == "final"

    complete = next(e for e in events if e["type"] == "tool_call_complete")
    assert complete["ok"] is True
    assert complete["exit_code"] == 0
    # The synthetic stdout is what the model sees as the tool result on
    # the next turn - it should describe what the user is now seeing.
    assert "runEval.v1" in complete["stdout"]
    # B1: both output and stdout fields populated for protocol parity.
    assert complete["output"] == complete["stdout"]


@pytest.mark.asyncio
async def test_start_tour_appears_in_canonical_tools() -> None:
    """The start_tour frontend-only tool must be visible to the LLM in
    the canonical_tools list passed to provider.stream_chat, even when
    the user-facing catalog is empty."""
    provider = MockProvider([[ProviderEvent(kind="finish")]])
    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=[],
        tool_runner=_unused_runner,
    )
    thread_id = runtime.create_thread()
    await runtime.start_turn(thread_id, "hello")
    tools = provider.calls[0]["tools"]
    names = [t["name"] for t in tools]
    assert "start_tour" in names


@pytest.mark.asyncio
async def test_create_thread_seeds_system_prompt() -> None:
    """create_thread must inject the co-pilot system message at index 0
    so providers see tour-policy instructions on every turn."""
    provider = MockProvider([[ProviderEvent(kind="finish")]])
    runtime = AgentRuntime(
        provider_factory=lambda: provider,
        catalog=[],
        tool_runner=_unused_runner,
    )
    thread_id = runtime.create_thread()
    messages = runtime._threads[thread_id].messages
    assert len(messages) >= 1
    assert messages[0]["role"] == "system"
    assert "start_tour" in messages[0]["content"]


def test_thread_counts_zero_on_fresh_runtime() -> None:
    """Brand-new AgentRuntime has no threads. Healthcheck reads
    `thread_counts` for the agent_threads / agent_open_threads
    fields, so the empty-runtime baseline must report zeros
    (not some stale or default-initialized non-zero)."""
    runtime = AgentRuntime(
        provider_factory=lambda: MockProvider([]),
        catalog=[],
        tool_runner=_unused_runner,
    )
    counts = runtime.thread_counts()
    assert counts == {"total": 0, "open": 0}


def test_thread_counts_matches_create_then_remove() -> None:
    """Create N threads -> total=N open=N. Remove one -> total=N-1
    open=N-1. The total/open relationship matters because the
    healthcheck surfaces them separately and "open > total"
    would be a clear contract bug."""
    runtime = AgentRuntime(
        provider_factory=lambda: MockProvider([]),
        catalog=[],
        tool_runner=_unused_runner,
    )
    a = runtime.create_thread()
    runtime.create_thread()
    runtime.create_thread()
    counts = runtime.thread_counts()
    assert counts == {"total": 3, "open": 3}

    runtime.remove_thread(a)
    counts2 = runtime.thread_counts()
    assert counts2 == {"total": 2, "open": 2}


def test_thread_counts_open_excludes_closed() -> None:
    """Closing a thread keeps it in `total` but drops it from
    `open`. This is the load-bearing distinction for the
    "are agents stuck?" forensic question - a runtime with many
    closed threads (normal post-conversation state) shouldn't
    look saturated to a healthcheck reading `open`."""
    runtime = AgentRuntime(
        provider_factory=lambda: MockProvider([]),
        catalog=[],
        tool_runner=_unused_runner,
    )
    a = runtime.create_thread()
    runtime.create_thread()

    # Mark `a` as closed via the internal flag (the test mirrors
    # what `final` would do in production).
    runtime._threads[a].closed = True

    counts = runtime.thread_counts()
    assert counts == {"total": 2, "open": 1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _unused_runner(argv: list[str]) -> tuple[str, int]:  # pragma: no cover
    raise AssertionError(f"tool_runner should not be called: {argv}")
