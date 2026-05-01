"""Tests for ``/api/cli`` and ``/api/cli/run`` (Lane B1.1, B1.2).

Covers:
- catalog endpoint shape + CSRF semantics.
- args validation: unknown flags, missing required, type/options mismatches.
- happy path: spawn a real subprocess via JobManager and observe the
  returned ``job_id``.
"""

from __future__ import annotations

import re
import sys

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from evalyn_dashboard.api.cli import args_to_argv
from evalyn_dashboard.introspect import CliSchema, ParamSchema
from evalyn_dashboard.server import CSRF_HEADER, build_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client_with_token() -> tuple[TestClient, str]:
    app = build_app()
    client = TestClient(app)
    html = client.get("/").text
    m = re.search(r'content="([^"]+)"', html)
    assert m, html[:200]
    return client, m.group(1)


# ---------------------------------------------------------------------------
# args_to_argv unit tests
# ---------------------------------------------------------------------------


def _schema(*params: ParamSchema) -> CliSchema:
    return CliSchema(id="t", name="t", group="x", blurb="", params=list(params))


def test_args_to_argv_string_flag():
    s = _schema(ParamSchema(name="name", kind="string"))
    assert args_to_argv(s, {"name": "alice"}) == ["--name", "alice"]


def test_args_to_argv_kebab_case():
    s = _schema(ParamSchema(name="input_file", kind="path"))
    assert args_to_argv(s, {"input_file": "x.json"}) == ["--input-file", "x.json"]


def test_args_to_argv_bool_true_emits_flag_only():
    s = _schema(ParamSchema(name="verbose", kind="bool", default=False))
    assert args_to_argv(s, {"verbose": True}) == ["--verbose"]


def test_args_to_argv_bool_false_omits_flag():
    s = _schema(ParamSchema(name="verbose", kind="bool", default=False))
    assert args_to_argv(s, {"verbose": False}) == []


def test_args_to_argv_number_int():
    s = _schema(ParamSchema(name="limit", kind="number", default=10))
    assert args_to_argv(s, {"limit": 5}) == ["--limit", "5"]


def test_args_to_argv_number_float():
    s = _schema(ParamSchema(name="threshold", kind="number"))
    assert args_to_argv(s, {"threshold": 0.5}) == ["--threshold", "0.5"]


def test_args_to_argv_number_rejects_bool():
    """Python ``bool`` is a subclass of ``int``; we must reject it."""
    s = _schema(ParamSchema(name="limit", kind="number"))
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        args_to_argv(s, {"limit": True})


def test_args_to_argv_number_rejects_string():
    s = _schema(ParamSchema(name="limit", kind="number"))
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        args_to_argv(s, {"limit": "5"})


def test_args_to_argv_select_valid():
    s = _schema(ParamSchema(name="mode", kind="select", options=["a", "b"]))
    assert args_to_argv(s, {"mode": "a"}) == ["--mode", "a"]


def test_args_to_argv_select_invalid_option():
    s = _schema(ParamSchema(name="mode", kind="select", options=["a", "b"]))
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        args_to_argv(s, {"mode": "z"})


def test_args_to_argv_multiselect_valid():
    s = _schema(ParamSchema(name="tags", kind="multiselect", options=["x", "y", "z"]))
    assert args_to_argv(s, {"tags": ["x", "z"]}) == ["--tags", "x", "z"]


def test_args_to_argv_multiselect_empty_list_omitted():
    s = _schema(ParamSchema(name="tags", kind="multiselect", options=["x", "y"]))
    assert args_to_argv(s, {"tags": []}) == []


def test_args_to_argv_multiselect_invalid_option():
    s = _schema(ParamSchema(name="tags", kind="multiselect", options=["x", "y"]))
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        args_to_argv(s, {"tags": ["x", "bogus"]})


def test_args_to_argv_multiselect_rejects_non_list():
    s = _schema(ParamSchema(name="tags", kind="multiselect", options=["x"]))
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        args_to_argv(s, {"tags": "x"})


def test_args_to_argv_unknown_flag_rejected():
    s = _schema(ParamSchema(name="name", kind="string"))
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        args_to_argv(s, {"bogus": "x"})


def test_args_to_argv_missing_required_rejected():
    s = _schema(ParamSchema(name="name", kind="string", required=True))
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        args_to_argv(s, {})


def test_args_to_argv_skips_none_and_empty_string():
    s = _schema(
        ParamSchema(name="name", kind="string"),
        ParamSchema(name="other", kind="string"),
    )
    assert args_to_argv(s, {"name": None, "other": ""}) == []


# ---------------------------------------------------------------------------
# GET /api/cli
# ---------------------------------------------------------------------------


def test_get_catalog_returns_array():
    client, _ = _client_with_token()
    r = client.get("/api/cli")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    # Every CLI must have id/name/group/blurb/params per schema.
    for entry in body:
        assert isinstance(entry["id"], str)
        assert isinstance(entry["name"], str)
        assert isinstance(entry["group"], str)
        assert "params" in entry
        assert isinstance(entry["params"], list)


def test_get_catalog_includes_well_known_commands():
    client, _ = _client_with_token()
    body = client.get("/api/cli").json()
    ids = {e["id"] for e in body}
    # Core evalyn commands the dashboard must expose.
    for expected in ("list-runs", "analyze", "compare"):
        assert expected in ids, f"missing {expected} from catalog"


def test_get_catalog_does_not_require_csrf():
    """GET routes are exempt from the workbench-token check."""
    client = TestClient(build_app())
    r = client.get("/api/cli")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/cli/run
# ---------------------------------------------------------------------------


def test_run_requires_csrf():
    client = TestClient(build_app())
    r = client.post("/api/cli/run", json={"cli_id": "list-runs", "args": {}})
    assert r.status_code == 403


def test_run_unknown_cli_404():
    client, token = _client_with_token()
    r = client.post(
        "/api/cli/run",
        json={"cli_id": "no-such-cli", "args": {}},
        headers={CSRF_HEADER: token},
    )
    assert r.status_code == 404


def test_run_invalid_body_400():
    client, token = _client_with_token()
    # Missing cli_id.
    r = client.post(
        "/api/cli/run",
        json={"args": {}},
        headers={CSRF_HEADER: token},
    )
    assert r.status_code == 400


def test_run_unknown_arg_400():
    client, token = _client_with_token()
    r = client.post(
        "/api/cli/run",
        json={"cli_id": "list-runs", "args": {"bogus_flag": 1}},
        headers={CSRF_HEADER: token},
    )
    assert r.status_code == 400


def test_run_select_invalid_option_400():
    client, token = _client_with_token()
    # ``list-runs`` has --format with choices ["table", "json"].
    r = client.post(
        "/api/cli/run",
        json={"cli_id": "list-runs", "args": {"format": "yaml"}},
        headers={CSRF_HEADER: token},
    )
    assert r.status_code == 400


def test_run_happy_path_returns_job_id(monkeypatch):
    """Spawn a real subprocess via JobManager and verify the job exists.

    Uses the catalog's ``list-runs`` schema for validation but swaps the
    actual command for ``python -c "print('ok')"`` via a JobManager
    monkeypatch so the test doesn't depend on a real evalyn project on
    disk.
    """
    app = build_app()
    client = TestClient(app)

    # Override JobManager.spawn to record the cmd the API tries to run
    # while still launching a harmless subprocess. We assert the cmd
    # shape directly.
    captured: list[list[str]] = []
    real_spawn = app.state.job_manager.spawn

    async def fake_spawn(cmd):
        captured.append(list(cmd))
        # Replace ``evalyn list-runs ...`` with a no-op python invocation
        # so we don't need a real project.
        return await real_spawn([sys.executable, "-c", "print('ok')"])

    monkeypatch.setattr(app.state.job_manager, "spawn", fake_spawn)

    html = client.get("/").text
    token = re.search(r'content="([^"]+)"', html).group(1)

    r = client.post(
        "/api/cli/run",
        json={"cli_id": "list-runs", "args": {"limit": 5, "format": "json"}},
        headers={CSRF_HEADER: token},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    # Confirm cmd shape.
    assert captured, "spawn was never called"
    cmd = captured[0]
    assert cmd[0] == "evalyn"
    assert cmd[1] == "list-runs"
    assert "--limit" in cmd and "5" in cmd
    assert "--format" in cmd and "json" in cmd

    # Job should exist in the JobManager.
    job = app.state.job_manager.get(body["job_id"])
    assert job is not None
