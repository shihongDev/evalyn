"""Tests for `evalyn search` query DSL over traces."""

from __future__ import annotations

import argparse
import io
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from evalyn_sdk.cli.commands import search
from evalyn_sdk.models import FunctionCall

BASE_TS = datetime(2026, 5, 23, 12, 0, 0)


def make_call(
    idx: int,
    function_name: str = "agent_main",
    project: str = "demo",
    duration_ms: float = 100.0,
    error: str | None = None,
    metadata_extra: dict | None = None,
) -> FunctionCall:
    ts = BASE_TS + timedelta(seconds=idx)
    meta = {"project": project, "project_name": project}
    if metadata_extra:
        meta.update(metadata_extra)
    return FunctionCall(
        id=f"{idx:08x}-aaaa-bbbb",
        function_name=function_name,
        inputs={},
        output=None,
        error=error,
        started_at=ts,
        ended_at=ts + timedelta(milliseconds=duration_ms),
        duration_ms=duration_ms,
        session_id=f"sess-{idx}",
        trace=[],
        metadata=meta,
    )


class FakeStorage:
    """Minimal stand-in for SQLiteStorage; only implements list_calls."""

    def __init__(self, calls: list[FunctionCall]):
        self._calls = calls

    def list_calls(self, limit=None, project=None, function_name=None, lightweight=True):
        return list(self._calls[:limit] if limit else self._calls)


def _make_args(query: str, **overrides) -> argparse.Namespace:
    ns = argparse.Namespace(
        query=query,
        db=None,
        limit=50,
        scan_limit=1000,
        format="table",
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


# ----------------------------------------------------------------------
# Field resolver
# ----------------------------------------------------------------------


def test_resolve_call_field_basic_fields():
    c = make_call(1, function_name="myfunc", project="myproj", duration_ms=42)
    assert search._resolve_call_field(c, "function") == "myfunc"
    assert search._resolve_call_field(c, "function_name") == "myfunc"
    assert search._resolve_call_field(c, "duration_ms") == 42
    assert search._resolve_call_field(c, "project") == "myproj"
    assert search._resolve_call_field(c, "id").startswith("00000001")


def test_resolve_call_field_error_and_status():
    ok_call = make_call(1, error=None)
    err_call = make_call(2, error="boom")
    assert search._resolve_call_field(ok_call, "error") is False
    assert search._resolve_call_field(err_call, "error") is True
    assert search._resolve_call_field(ok_call, "status") == "ok"
    assert search._resolve_call_field(err_call, "status") == "error"


def test_resolve_call_field_metadata_paths():
    c = make_call(1, metadata_extra={"user_id": "alice", "env": "prod"})
    assert search._resolve_call_field(c, "metadata.user_id") == "alice"
    assert search._resolve_call_field(c, "metadata.env") == "prod"
    assert search._resolve_call_field(c, "metadata.missing") is None


# ----------------------------------------------------------------------
# Expression evaluation
# ----------------------------------------------------------------------


def test_numeric_filter_greater_than():
    fast = make_call(1, duration_ms=100)
    slow = make_call(2, duration_ms=5000)
    storage = FakeStorage([fast, slow])
    args = _make_args("duration_ms > 1000")
    buf = io.StringIO()
    with patch.object(search, "_get_storage", return_value=storage), \
         patch("sys.stdout", new=buf):
        search.cmd_search(args)
    output = buf.getvalue()
    assert "00000002" in output  # slow call
    assert "00000001" not in output  # fast call excluded


def test_equality_filter_on_project():
    a = make_call(1, project="app_a")
    b = make_call(2, project="app_b")
    storage = FakeStorage([a, b])
    args = _make_args("project = app_a")
    buf = io.StringIO()
    with patch.object(search, "_get_storage", return_value=storage), \
         patch("sys.stdout", new=buf):
        search.cmd_search(args)
    output = buf.getvalue()
    assert "00000001" in output
    assert "00000002" not in output


def test_boolean_error_filter():
    ok = make_call(1, error=None)
    bad = make_call(2, error="oops")
    storage = FakeStorage([ok, bad])
    args = _make_args("error = true")
    buf = io.StringIO()
    with patch.object(search, "_get_storage", return_value=storage), \
         patch("sys.stdout", new=buf):
        search.cmd_search(args)
    output = buf.getvalue()
    assert "00000002" in output
    assert "00000001" not in output


def test_compound_and_filter():
    a = make_call(1, project="p", duration_ms=100)
    b = make_call(2, project="p", duration_ms=5000)
    c = make_call(3, project="q", duration_ms=5000)
    storage = FakeStorage([a, b, c])
    args = _make_args("project = p and duration_ms > 1000")
    buf = io.StringIO()
    with patch.object(search, "_get_storage", return_value=storage), \
         patch("sys.stdout", new=buf):
        search.cmd_search(args)
    output = buf.getvalue()
    assert "00000002" in output
    assert "00000001" not in output
    assert "00000003" not in output


def test_compound_or_filter():
    a = make_call(1, project="p", duration_ms=100)
    b = make_call(2, project="p", duration_ms=5000)
    c = make_call(3, project="q", duration_ms=100)
    storage = FakeStorage([a, b, c])
    args = _make_args("duration_ms > 1000 or project = q")
    buf = io.StringIO()
    with patch.object(search, "_get_storage", return_value=storage), \
         patch("sys.stdout", new=buf):
        search.cmd_search(args)
    output = buf.getvalue()
    assert "00000002" in output  # slow
    assert "00000003" in output  # project=q
    assert "00000001" not in output


def test_contains_operator_on_function():
    a = make_call(1, function_name="chat_handler")
    b = make_call(2, function_name="search_api")
    storage = FakeStorage([a, b])
    args = _make_args("function contains chat")
    buf = io.StringIO()
    with patch.object(search, "_get_storage", return_value=storage), \
         patch("sys.stdout", new=buf):
        search.cmd_search(args)
    output = buf.getvalue()
    assert "00000001" in output
    assert "00000002" not in output


# ----------------------------------------------------------------------
# Output formats
# ----------------------------------------------------------------------


def test_json_format_output():
    import json

    a = make_call(1, project="p")
    storage = FakeStorage([a])
    args = _make_args("project = p", format="json")
    buf = io.StringIO()
    with patch.object(search, "_get_storage", return_value=storage), \
         patch("sys.stdout", new=buf):
        search.cmd_search(args)
    payload = json.loads(buf.getvalue())
    assert payload["matches"] == 1
    assert payload["calls"][0]["function"] == "agent_main"


def test_no_matches_message():
    storage = FakeStorage([make_call(1, project="other")])
    args = _make_args("project = nonexistent")
    buf = io.StringIO()
    with patch.object(search, "_get_storage", return_value=storage), \
         patch("sys.stdout", new=buf):
        search.cmd_search(args)
    output = buf.getvalue()
    assert "No calls matched" in output


def test_limit_caps_output():
    calls = [make_call(i, project="p") for i in range(20)]
    storage = FakeStorage(calls)
    args = _make_args("project = p", limit=5)
    buf = io.StringIO()
    with patch.object(search, "_get_storage", return_value=storage), \
         patch("sys.stdout", new=buf):
        search.cmd_search(args)
    output = buf.getvalue()
    # Data rows are those containing "agent_main" (the function name)
    data_lines = [l for l in output.splitlines() if "agent_main" in l]
    assert len(data_lines) == 5


# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------


def test_invalid_query_exits_with_help():
    args = _make_args("nonsense without operator")
    buf = io.StringIO()
    with patch("sys.stdout", new=buf), pytest.raises(SystemExit) as exc:
        search.cmd_search(args)
    assert exc.value.code == 2
    assert "Supported fields" in buf.getvalue()


# ----------------------------------------------------------------------
# Argparse wiring
# ----------------------------------------------------------------------


def test_register_commands_wires_subparser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    search.register_commands(subparsers)
    args = parser.parse_args(["search", "duration_ms > 5000"])
    assert args.query == "duration_ms > 5000"
    assert args.limit == 50
    assert args.format == "table"
    assert hasattr(args, "func")
