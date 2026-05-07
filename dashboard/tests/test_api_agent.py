"""Tests for ``/api/agent/*`` and ``/ws/agent/{thread_id}`` (Lane C1.5-C1.7).

The runtime is constructed with a ``MockProvider`` so no real LLM call
happens. WebSocket interactions use ``TestClient.websocket_connect``.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from evalyn_dashboard.agent import (
    AgentRuntime,
    BaseProvider,
    ProviderEvent,
    ProviderToolCall,
)
from evalyn_dashboard.credentials import CredentialStore
from evalyn_dashboard.introspect import CliSchema, ParamSchema
from evalyn_dashboard.server import CSRF_HEADER, build_app


class MockProvider(BaseProvider):
    name = "mock"

    def __init__(self, turns: list[list[ProviderEvent]]) -> None:
        self._turns = list(turns)

    async def stream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ProviderEvent]:
        if not self._turns:
            yield ProviderEvent(kind="finish")
            return
        turn = self._turns.pop(0)
        for evt in turn:
            yield evt


def _sample_catalog() -> list[CliSchema]:
    return [
        CliSchema(
            id="list-runs",
            name="list-runs",
            group="Eval",
            blurb="List runs",
            params=[ParamSchema(name="limit", kind="number", default=10)],
        ),
        CliSchema(
            id="run-eval",
            name="run-eval",
            group="Eval",
            blurb="Run an eval",
            params=[ParamSchema(name="dataset", kind="path", required=True)],
        ),
    ]


def _token_from(client: TestClient) -> str:
    html = client.get("/").text
    m = re.search(r'content="([^"]+)"', html)
    assert m
    return m.group(1)


def _make_app(
    tmp_path: Path, runtime: AgentRuntime
) -> Any:
    store = CredentialStore(path=tmp_path / "cred.json")
    app = build_app(credential_store=store, agent_runtime=runtime)
    # Replace the runtime's catalog with our small one so test tools route
    # cleanly. This also makes the test run fast (no full catalog walk).
    app.state.cli_catalog = _sample_catalog()
    return app


def _make_client(
    tmp_path: Path, runtime: AgentRuntime
) -> tuple[TestClient, str]:
    app = _make_app(tmp_path, runtime)
    client = TestClient(app)
    return client, _token_from(client)


# ---------------------------------------------------------------------------
# POST /api/agent/chat
# ---------------------------------------------------------------------------


def test_chat_creates_thread(tmp_path: Path) -> None:
    runner_calls: list[list[str]] = []

    async def runner(argv: list[str]) -> tuple[str, int]:
        runner_calls.append(argv)
        return "row\n", 0

    runtime = AgentRuntime(
        provider_factory=lambda: MockProvider(
            [[ProviderEvent(kind="text_delta", text="hi"), ProviderEvent(kind="finish")]]
        ),
        catalog=_sample_catalog(),
        tool_runner=runner,
    )
    client, token = _make_client(tmp_path, runtime)

    r = client.post(
        "/api/agent/chat",
        json={"message": "hello"},
        headers={CSRF_HEADER: token},
    )
    assert r.status_code == 200
    body = r.json()
    assert "thread_id" in body
    assert isinstance(body["thread_id"], str)
    assert runtime.has_thread(body["thread_id"])


def test_chat_reuses_thread_id(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        provider_factory=lambda: MockProvider(
            [
                [ProviderEvent(kind="text_delta", text="a"), ProviderEvent(kind="finish")],
                [ProviderEvent(kind="text_delta", text="b"), ProviderEvent(kind="finish")],
            ]
        ),
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    client, token = _make_client(tmp_path, runtime)
    first = client.post(
        "/api/agent/chat",
        json={"message": "1"},
        headers={CSRF_HEADER: token},
    ).json()["thread_id"]
    second = client.post(
        "/api/agent/chat",
        json={"message": "2", "thread_id": first},
        headers={CSRF_HEADER: token},
    ).json()["thread_id"]
    assert first == second


def test_chat_rejects_empty_message(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        provider_factory=lambda: None,
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    client, token = _make_client(tmp_path, runtime)
    r = client.post(
        "/api/agent/chat",
        json={"message": ""},
        headers={CSRF_HEADER: token},
    )
    assert r.status_code == 400


def test_chat_requires_csrf(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        provider_factory=lambda: None,
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    client, _ = _make_client(tmp_path, runtime)
    r = client.post("/api/agent/chat", json={"message": "hi"})
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# POST /api/agent/chat/{thread_id}/confirm
# ---------------------------------------------------------------------------


def test_confirm_unknown_thread_404(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        provider_factory=lambda: None,
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    client, token = _make_client(tmp_path, runtime)
    r = client.post(
        "/api/agent/chat/nonexistent/confirm",
        json={"approve": True},
        headers={CSRF_HEADER: token},
    )
    assert r.status_code == 404


def test_confirm_invalid_payload_400(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        provider_factory=lambda: None,
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    client, token = _make_client(tmp_path, runtime)
    thread_id = runtime.create_thread()
    r = client.post(
        f"/api/agent/chat/{thread_id}/confirm",
        json={"approve": "yes"},
        headers={CSRF_HEADER: token},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# WS /ws/agent/{thread_id}
# ---------------------------------------------------------------------------


def test_ws_agent_streams_events(tmp_path: Path) -> None:
    """End-to-end: POST chat, then connect WS, expect text + final."""
    turns = [
        [
            ProviderEvent(kind="text_delta", text="hello"),
            ProviderEvent(kind="finish"),
        ]
    ]
    runtime = AgentRuntime(
        provider_factory=lambda: MockProvider(turns),
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    client, token = _make_client(tmp_path, runtime)

    # Pre-create the thread so the WS opens before the turn schedules.
    thread_id = runtime.create_thread()
    with client.websocket_connect(f"/ws/agent/{thread_id}") as ws:
        # Schedule the turn after WS is up.
        client.post(
            "/api/agent/chat",
            json={"message": "go", "thread_id": thread_id},
            headers={CSRF_HEADER: token},
        )
        events: list[dict[str, Any]] = []
        for _ in range(20):
            data = ws.receive_text()
            events.append(json.loads(data))
            if events[-1]["type"] == "final":
                break
    types = [e["type"] for e in events]
    assert "text_delta" in types
    assert types[-1] == "final"

    # Contract regression guard for the v2 co-pilot frontend
    # (dashboard/frontend/src/v2/copilot/types.ts AgentWsEvent).
    # If these field names change here, update the TS contract together.
    text_deltas = [e for e in events if e["type"] == "text_delta"]
    assert text_deltas, "expected at least one text_delta event"
    for d in text_deltas:
        assert "text" in d, "text_delta uses 'text' (not 'delta')"
        assert "message_id" in d, "text_delta needs message_id for bubble correlation"
    final = events[-1]
    # The 'final' emit sites in agent.py do not include message_id; the
    # frontend handler must fall back to the last streaming bubble.
    assert "message_id" not in final, (
        "agent.py 'final' emits without message_id; frontend "
        "useCoPilotThread.ts handles this by patching the last streaming bubble"
    )


def test_ws_agent_unknown_thread_closes_1008(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        provider_factory=lambda: None,
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    client, _ = _make_client(tmp_path, runtime)
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/agent/does-not-exist") as ws:
            ws.receive_text()


def test_ws_agent_replays_with_since(tmp_path: Path) -> None:
    """After a turn completes, a fresh WS should still get all events."""
    runtime = AgentRuntime(
        provider_factory=lambda: MockProvider(
            [
                [
                    ProviderEvent(kind="text_delta", text="abc"),
                    ProviderEvent(kind="finish"),
                ]
            ]
        ),
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    client, token = _make_client(tmp_path, runtime)
    r = client.post(
        "/api/agent/chat",
        json={"message": "go"},
        headers={CSRF_HEADER: token},
    )
    thread_id = r.json()["thread_id"]

    # Wait for the turn to finish (events are buffered on the thread).
    deadline = time.time() + 5.0
    while time.time() < deadline:
        evts = runtime._threads[thread_id].events
        if any(e["type"] == "final" for e in evts):
            break
        time.sleep(0.02)

    received: list[dict[str, Any]] = []
    with client.websocket_connect(f"/ws/agent/{thread_id}") as ws:
        for _ in range(20):
            try:
                data = ws.receive_text()
            except Exception:
                break
            received.append(json.loads(data))
            if received[-1]["type"] == "final":
                break
    types = [e["type"] for e in received]
    assert types[-1] == "final"
    assert "text_delta" in types


# ---------------------------------------------------------------------------
# Confirmation flow end-to-end
# ---------------------------------------------------------------------------


def test_confirmation_flow_approve_runs_tool(tmp_path: Path) -> None:
    """Drive the confirmation gate through the HTTP API.

    The TestClient serialises every call through the same anyio portal
    so the WS frames and the confirm POST see the same event loop and
    asyncio.Event. We poll the buffered events on the runtime to decide
    when to confirm rather than trying to interleave ws.receive_text()
    with client.post() (which TestClient cannot do safely - one blocks
    the portal until completion).
    """
    tc = ProviderToolCall(id="tc-1", name="run-eval", arguments={"dataset": "ds"})
    turns = [
        [ProviderEvent(kind="tool_call", tool_call=tc), ProviderEvent(kind="finish")],
        [ProviderEvent(kind="text_delta", text="done"), ProviderEvent(kind="finish")],
    ]
    spawn_calls: list[list[str]] = []

    async def runner(argv: list[str]) -> tuple[str, int]:
        spawn_calls.append(argv)
        return "ran", 0

    runtime = AgentRuntime(
        provider_factory=lambda: MockProvider(turns),
        catalog=_sample_catalog(),
        tool_runner=runner,
        confirm_timeout=2.0,
    )
    app = _make_app(tmp_path, runtime)
    with TestClient(app) as client:
        token = _token_from(client)
        r = client.post(
            "/api/agent/chat",
            json={"message": "go"},
            headers={CSRF_HEADER: token},
        )
        thread_id = r.json()["thread_id"]

        # Wait for confirmation_required to land in the buffer.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            evts = runtime._threads[thread_id].events
            if any(e["type"] == "confirmation_required" for e in evts):
                break
            time.sleep(0.02)
        else:
            pytest.fail("confirmation_required never emitted")

        r = client.post(
            f"/api/agent/chat/{thread_id}/confirm",
            json={"approve": True},
            headers={CSRF_HEADER: token},
        )
        assert r.status_code == 200

        # Wait for the loop to finish.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            evts = runtime._threads[thread_id].events
            if any(e["type"] == "final" for e in evts):
                break
            time.sleep(0.02)

        events = list(runtime._threads[thread_id].events)

    types = [e["type"] for e in events]
    assert "confirmation_required" in types
    assert "tool_call_running" in types
    assert "tool_call_complete" in types
    assert types[-1] == "final"
    assert spawn_calls == [["evalyn", "run-eval", "--dataset", "ds"]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _unused_runner(argv: list[str]) -> tuple[str, int]:  # pragma: no cover
    raise AssertionError(f"runner should not be called: {argv}")


# ---------------------------------------------------------------------------
# GET /api/agent/threads
# ---------------------------------------------------------------------------


def test_list_threads_empty_when_runtime_fresh(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        provider_factory=lambda: MockProvider([]),
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    client, _token = _make_client(tmp_path, runtime)
    r = client.get("/api/agent/threads")
    assert r.status_code == 200
    assert r.json() == []


def test_list_threads_returns_metadata_for_each_thread(tmp_path: Path) -> None:
    """Threads created via the runtime show up in the listing with
    cheap metadata (counts + flags). The full message bodies are NOT
    included; the listing should stay lightweight."""
    runtime = AgentRuntime(
        provider_factory=lambda: MockProvider(
            [[ProviderEvent(kind="text_delta", text="hi"), ProviderEvent(kind="finish")]]
        ),
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    client, token = _make_client(tmp_path, runtime)
    # Create two threads via the chat endpoint.
    r1 = client.post(
        "/api/agent/chat",
        json={"message": "first"},
        headers={CSRF_HEADER: token},
    )
    assert r1.status_code == 200
    r2 = client.post(
        "/api/agent/chat",
        json={"message": "second"},
        headers={CSRF_HEADER: token},
    )
    assert r2.status_code == 200
    tid1 = r1.json()["thread_id"]
    tid2 = r2.json()["thread_id"]

    r = client.get("/api/agent/threads")
    assert r.status_code == 200
    body = r.json()
    ids = {t["id"] for t in body}
    assert tid1 in ids
    assert tid2 in ids
    # Each entry has the documented shape; counts are integers.
    for t in body:
        assert isinstance(t["message_count"], int) and t["message_count"] >= 1
        assert isinstance(t["event_count"], int)
        assert isinstance(t["closed"], bool)
        assert isinstance(t["has_pending_confirmation"], bool)


def test_get_single_thread_returns_metadata(tmp_path: Path) -> None:
    """GET /api/agent/threads/{id} returns the same shape as one entry
    in the list endpoint. 404 on unknown."""
    runtime = AgentRuntime(
        provider_factory=lambda: MockProvider(
            [[ProviderEvent(kind="text_delta", text="hi"), ProviderEvent(kind="finish")]]
        ),
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    client, token = _make_client(tmp_path, runtime)

    r1 = client.post(
        "/api/agent/chat",
        json={"message": "hello"},
        headers={CSRF_HEADER: token},
    )
    tid = r1.json()["thread_id"]

    r = client.get(f"/api/agent/threads/{tid}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == tid
    assert isinstance(body["message_count"], int)
    assert isinstance(body["event_count"], int)
    assert isinstance(body["closed"], bool)
    assert isinstance(body["has_pending_confirmation"], bool)

    # Same shape as entries in the list endpoint.
    listed = client.get("/api/agent/threads").json()
    matching = [t for t in listed if t["id"] == tid]
    assert len(matching) == 1
    assert matching[0] == body


def test_get_single_thread_unknown_returns_404(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        provider_factory=lambda: MockProvider([]),
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    client, _token = _make_client(tmp_path, runtime)
    r = client.get("/api/agent/threads/never-existed")
    assert r.status_code == 404


def test_delete_thread_removes_from_runtime(tmp_path: Path) -> None:
    """DELETE /api/agent/threads/{id} drops the thread from the runtime
    so a subsequent GET /threads no longer lists it. 404 on unknown."""
    runtime = AgentRuntime(
        provider_factory=lambda: MockProvider(
            [[ProviderEvent(kind="text_delta", text="hi"), ProviderEvent(kind="finish")]]
        ),
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    client, token = _make_client(tmp_path, runtime)

    # Create a thread, verify it lists.
    r1 = client.post(
        "/api/agent/chat",
        json={"message": "hello"},
        headers={CSRF_HEADER: token},
    )
    tid = r1.json()["thread_id"]
    listed_before = {t["id"] for t in client.get("/api/agent/threads").json()}
    assert tid in listed_before

    # DELETE removes it.
    r = client.delete(
        f"/api/agent/threads/{tid}", headers={CSRF_HEADER: token}
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True, "id": tid}

    # No longer listed.
    listed_after = {t["id"] for t in client.get("/api/agent/threads").json()}
    assert tid not in listed_after

    # Idempotent miss returns 404, not 500.
    r2 = client.delete(
        f"/api/agent/threads/{tid}", headers={CSRF_HEADER: token}
    )
    assert r2.status_code == 404


def test_delete_thread_unknown_returns_404(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        provider_factory=lambda: MockProvider([]),
        catalog=_sample_catalog(),
        tool_runner=_unused_runner,
    )
    client, token = _make_client(tmp_path, runtime)
    r = client.delete(
        "/api/agent/threads/never-existed", headers={CSRF_HEADER: token}
    )
    assert r.status_code == 404


def test_list_threads_503_when_runtime_missing(tmp_path: Path) -> None:
    """If agent_runtime is absent on app.state, the endpoint returns
    503 (matching the chat endpoint's behavior)."""
    from evalyn_dashboard.server import build_app

    app = build_app()
    # build_app installs an agent_runtime by default; clear it to
    # simulate a misconfigured deployment.
    app.state.agent_runtime = None
    client = TestClient(app)
    r = client.get("/api/agent/threads")
    assert r.status_code == 503
