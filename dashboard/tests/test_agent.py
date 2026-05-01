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
# Helpers
# ---------------------------------------------------------------------------


async def _unused_runner(argv: list[str]) -> tuple[str, int]:  # pragma: no cover
    raise AssertionError(f"tool_runner should not be called: {argv}")
