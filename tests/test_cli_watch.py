"""Unit tests for `evalyn watch` CLI command.

Tests cover:
- Initial backlog rendering (most-recent-last)
- Poll loop emits only newly-arrived calls
- Filters: --project, --function, --error-only
- KeyboardInterrupt is caught cleanly
- No real sleeping (interval=0 or stubbed time.sleep)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from evalyn_sdk.cli.commands import watch as watch_mod


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


BASE_TS = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)


def make_call(
    *,
    idx: int,
    function_name: str = "agent_main",
    project: str = "demo",
    error: str | None = None,
    duration_ms: float = 123.0,
    started_at: datetime | None = None,
) -> Any:
    """Build a lightweight FunctionCall-like object for tests.

    We use the real FunctionCall dataclass so as_dict / attribute access
    behave identically to production.
    """
    from evalyn_sdk.models import FunctionCall

    ts = started_at or (BASE_TS + timedelta(seconds=idx))
    # Use a hex-style id that varies in the first 8 chars across idx values
    # so short-id (first 8 chars) is meaningfully distinct in tests.
    return FunctionCall(
        id=f"{idx:08x}-abcd-efgh",
        function_name=function_name,
        inputs={},
        output=None,
        error=error,
        started_at=ts,
        ended_at=ts + timedelta(milliseconds=duration_ms),
        duration_ms=duration_ms,
        session_id=f"sess-{idx}",
        trace=[],
        metadata={"project_name": project},
    )


class FakeStorage:
    """Storage stub that returns calls in DESC started_at order (like real SQLite).

    Supports the same kwargs as real ``SQLiteStorage.list_calls``:
    limit, project, function_name, lightweight, version.
    """

    def __init__(self, calls: list):
        # Hold by reference so tests can mutate (simulating new arrivals).
        self._calls = calls

    def list_calls(
        self,
        limit: int = 100,
        project: str | None = None,
        function_name: str | None = None,
        lightweight: bool = False,
        version: str | None = None,
    ):
        result = list(self._calls)
        if project:
            result = [
                c
                for c in result
                if (c.metadata or {}).get("project_name") == project
                or (c.metadata or {}).get("project_id") == project
            ]
        if function_name:
            result = [
                c
                for c in result
                if function_name.lower() in (c.function_name or "").lower()
            ]
        # Sort DESC by started_at to match real SQL query.
        result.sort(key=lambda c: c.started_at, reverse=True)
        return result[:limit]


def make_args(**overrides) -> argparse.Namespace:
    """Build a default argparse Namespace for cmd_watch."""
    defaults = dict(
        project=None,
        function=None,
        error_only=False,
        interval=0.0,
        limit=5,
        db=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Backlog rendering
# ---------------------------------------------------------------------------


class TestInitialBacklog:
    """The initial backlog should print most-recent-last."""

    def test_prints_existing_calls_chronologically(self, capsys, monkeypatch):
        calls = [make_call(idx=i) for i in range(5)]
        storage = FakeStorage(calls)

        # Replace _get_storage so we don't hit the real DB.
        monkeypatch.setattr(watch_mod, "_get_storage", lambda args: storage)

        # Make _poll_loop a no-op so we only exercise the initial print.
        monkeypatch.setattr(watch_mod, "_poll_loop", lambda *a, **kw: None)

        args = make_args(limit=5)
        ret = watch_mod.cmd_watch(args)
        assert ret == 0

        out = capsys.readouterr().out.strip().splitlines()
        # 5 calls printed, oldest first so newest is at the bottom.
        assert len(out) == 5
        # storage returns DESC, head=limit, reversed -> oldest first.
        # So out[0] == oldest (idx=0) and out[-1] == newest (idx=4).
        # Short ID = first 8 chars of `{idx:08x}-abcd-efgh` = the idx hex.
        assert out[0].split()[1] == f"{0:08x}"
        assert out[-1].split()[1] == f"{4:08x}"

    def test_initial_respects_limit(self, capsys, monkeypatch):
        calls = [make_call(idx=i) for i in range(20)]
        storage = FakeStorage(calls)
        monkeypatch.setattr(watch_mod, "_get_storage", lambda args: storage)
        monkeypatch.setattr(watch_mod, "_poll_loop", lambda *a, **kw: None)

        args = make_args(limit=3)
        watch_mod.cmd_watch(args)
        out_lines = capsys.readouterr().out.strip().splitlines()
        assert len(out_lines) == 3


# ---------------------------------------------------------------------------
# Poll loop: only NEW calls are printed on subsequent iterations
# ---------------------------------------------------------------------------


class TestPollLoopNewCalls:
    """After the initial print, the poll loop should emit only newly-arrived calls."""

    def test_only_new_calls_are_printed(self, capsys, monkeypatch):
        # Backlog: 3 calls.
        initial_calls = [make_call(idx=i) for i in range(3)]
        storage = FakeStorage(initial_calls)
        monkeypatch.setattr(watch_mod, "_get_storage", lambda args: storage)

        # Stage 2 new calls that will appear on the first poll tick.
        new_calls = [make_call(idx=10), make_call(idx=11)]

        # We run the poll loop for exactly one iteration. Before the fetch,
        # add the new calls; sleep_fn is a no-op.
        def fake_sleep(_):
            return None

        # Monkey-patch _poll_loop's iterations via a wrapper.
        original_poll = watch_mod._poll_loop

        def bounded_poll(*args, **kwargs):
            # Mutate storage to add the new calls before the first poll fetch.
            storage._calls.extend(new_calls)
            kwargs["iterations"] = 1
            kwargs["sleep_fn"] = fake_sleep
            return original_poll(*args, **kwargs)

        monkeypatch.setattr(watch_mod, "_poll_loop", bounded_poll)

        args = make_args(limit=3, interval=0.0)
        ret = watch_mod.cmd_watch(args)
        assert ret == 0

        out_lines = capsys.readouterr().out.strip().splitlines()
        # 3 initial + 2 new = 5 total lines.
        assert len(out_lines) == 5

        # The last 2 lines should be the new calls in chronological order
        # (idx=10, then idx=11). Short id = first 8 chars of `{idx:08x}-abcd...`.
        assert out_lines[-2].split()[1] == f"{10:08x}"
        assert out_lines[-1].split()[1] == f"{11:08x}"
        # Stronger: short ids of new lines disjoint from short ids of initial.
        new_short_ids = {ln.split()[1] for ln in out_lines[-2:]}
        initial_short_ids = {ln.split()[1] for ln in out_lines[:3]}
        assert new_short_ids.isdisjoint(initial_short_ids)

    def test_no_new_calls_no_extra_output(self, capsys, monkeypatch):
        """If nothing changes between polls, no new lines should be printed."""
        calls = [make_call(idx=i) for i in range(3)]
        storage = FakeStorage(calls)
        monkeypatch.setattr(watch_mod, "_get_storage", lambda args: storage)

        def fake_sleep(_):
            return None

        original_poll = watch_mod._poll_loop

        def bounded_poll(*args, **kwargs):
            kwargs["iterations"] = 2  # poll twice, no new data
            kwargs["sleep_fn"] = fake_sleep
            return original_poll(*args, **kwargs)

        monkeypatch.setattr(watch_mod, "_poll_loop", bounded_poll)

        args = make_args(limit=3, interval=0.0)
        watch_mod.cmd_watch(args)

        out_lines = capsys.readouterr().out.strip().splitlines()
        # Only the 3 initial lines; no extra output from idle polls.
        assert len(out_lines) == 3


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestFilters:
    """--project, --function, --error-only filters."""

    def test_project_filter(self, capsys, monkeypatch):
        calls = [
            make_call(idx=0, project="alpha"),
            make_call(idx=1, project="beta"),
            make_call(idx=2, project="alpha"),
        ]
        storage = FakeStorage(calls)
        monkeypatch.setattr(watch_mod, "_get_storage", lambda args: storage)
        monkeypatch.setattr(watch_mod, "_poll_loop", lambda *a, **kw: None)

        args = make_args(project="alpha", limit=10)
        watch_mod.cmd_watch(args)
        out = capsys.readouterr().out
        # Only alpha calls printed.
        assert out.count("alpha") == 2
        assert "beta" not in out

    def test_function_filter(self, capsys, monkeypatch):
        calls = [
            make_call(idx=0, function_name="agent_main"),
            make_call(idx=1, function_name="other_fn"),
            make_call(idx=2, function_name="agent_helper"),
        ]
        storage = FakeStorage(calls)
        monkeypatch.setattr(watch_mod, "_get_storage", lambda args: storage)
        monkeypatch.setattr(watch_mod, "_poll_loop", lambda *a, **kw: None)

        args = make_args(function="agent", limit=10)
        watch_mod.cmd_watch(args)
        out = capsys.readouterr().out
        assert "agent_main" in out
        assert "agent_helper" in out
        assert "other_fn" not in out

    def test_error_only_filter(self, capsys, monkeypatch):
        calls = [
            make_call(idx=0, error=None),
            make_call(idx=1, error="boom"),
            make_call(idx=2, error=None),
            make_call(idx=3, error="kaboom"),
        ]
        storage = FakeStorage(calls)
        monkeypatch.setattr(watch_mod, "_get_storage", lambda args: storage)
        monkeypatch.setattr(watch_mod, "_poll_loop", lambda *a, **kw: None)

        args = make_args(error_only=True, limit=10)
        watch_mod.cmd_watch(args)
        out = capsys.readouterr().out
        # Both error rows print; non-error rows don't.
        # Status column is "ERROR" for errored calls (possibly color-wrapped).
        assert "ERROR" in out
        # OK status should not appear (no non-error calls passed the filter).
        # Count short ids: only call-00000001 and call-00000003 should appear.
        # Their short id prefix differs from the OK rows.
        lines = [ln for ln in out.strip().splitlines() if ln.strip()]
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# KeyboardInterrupt handling
# ---------------------------------------------------------------------------


class TestKeyboardInterrupt:
    """Ctrl+C should exit cleanly with code 0 and a 'watch stopped' message on stderr."""

    def test_keyboard_interrupt_clean_exit(self, capsys, monkeypatch):
        storage = FakeStorage([])
        monkeypatch.setattr(watch_mod, "_get_storage", lambda args: storage)

        def raising_poll(*a, **kw):
            raise KeyboardInterrupt

        monkeypatch.setattr(watch_mod, "_poll_loop", raising_poll)

        args = make_args(limit=0, interval=0.0)
        ret = watch_mod.cmd_watch(args)
        assert ret == 0

        captured = capsys.readouterr()
        assert "watch stopped" in captured.err


# ---------------------------------------------------------------------------
# CLI registration smoke test
# ---------------------------------------------------------------------------


class TestRegistration:
    """register_commands wires up the parser correctly."""

    def test_register_commands_adds_watch(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="cmd")
        watch_mod.register_commands(subparsers)

        # Parse a representative invocation.
        ns = parser.parse_args(
            ["watch", "--project", "p", "--function", "f", "--error-only", "--interval", "1.5", "--limit", "7", "--db", "test"]
        )
        assert ns.cmd == "watch"
        assert ns.project == "p"
        assert ns.function == "f"
        assert ns.error_only is True
        assert ns.interval == 1.5
        assert ns.limit == 7
        assert ns.db == "test"
        # set_defaults wires func to cmd_watch.
        assert ns.func is watch_mod.cmd_watch

    def test_defaults(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="cmd")
        watch_mod.register_commands(subparsers)
        ns = parser.parse_args(["watch"])
        assert ns.interval == 2.0
        assert ns.limit == 5
        assert ns.error_only is False
        assert ns.project is None
        assert ns.function is None
        assert ns.db is None


# ---------------------------------------------------------------------------
# Line format
# ---------------------------------------------------------------------------


class TestFormatCallLine:
    """The per-call output line should follow the documented format."""

    def test_format_contains_required_fields(self):
        call = make_call(idx=0, function_name="agent", project="proj", duration_ms=1234.0)
        line = watch_mod._format_call_line(call)
        # HH:MM:SS short_id function project status duration
        assert "12:00:00" in line  # from BASE_TS
        assert f"{0:08x}" in line  # short id for idx=0
        assert "agent" in line
        assert "proj" in line
        assert "OK" in line
        # 1234ms -> "1.2s" per format_duration.
        assert "1.2s" in line

    def test_format_error_status(self):
        call = make_call(idx=0, error="boom")
        line = watch_mod._format_call_line(call)
        assert "ERROR" in line

    def test_format_missing_project_shows_dash(self):
        from evalyn_sdk.models import FunctionCall

        call = FunctionCall(
            id="abc12345xyz",
            function_name="fn",
            inputs={},
            output=None,
            error=None,
            started_at=BASE_TS,
            ended_at=BASE_TS,
            duration_ms=10.0,
            session_id=None,
            trace=[],
            metadata={},  # no project
        )
        line = watch_mod._format_call_line(call)
        # Project field falls back to "-".
        assert " - " in line
