"""Tests for ``evalyn_dashboard.server`` FastAPI app skeleton (A1.1)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from evalyn_dashboard.server import build_app


def test_healthcheck() -> None:
    client = TestClient(build_app())
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    # Backwards-compatible: ok field still present.
    assert body["ok"] is True
    # Newer fields useful for monitors and build-tagging.
    assert isinstance(body["version"], str) and len(body["version"]) > 0
    assert isinstance(body["started_at"], (int, float))
    assert body["started_at"] > 0
    assert isinstance(body["uptime_seconds"], int)
    assert body["uptime_seconds"] >= 0


def test_index_served() -> None:
    client = TestClient(build_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()


def test_static_assets_served() -> None:
    client = TestClient(build_app())
    r = client.get("/static/index.html")
    assert r.status_code == 200


def test_spa_fallback_returns_index_for_unknown_paths() -> None:
    client = TestClient(build_app())
    r = client.get("/some/unknown/route")
    assert r.status_code == 200
    assert "<html" in r.text.lower()


def test_unknown_api_route_does_not_fall_back_to_index() -> None:
    client = TestClient(build_app())
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404


# ---- A1.4: browser auto-open scheduling -----------------------------------


class _FakeServer:
    def __init__(self, started: bool = True) -> None:
        self.started = started


def test_schedule_browser_open_calls_opener_when_started() -> None:
    from evalyn_dashboard.server import _schedule_browser_open

    seen: list[str] = []
    t = _schedule_browser_open(
        _FakeServer(started=True),
        "http://127.0.0.1:7401/",
        open_delay=0.0,
        opener=lambda url: seen.append(url) or True,
    )
    t.join(timeout=2.0)
    assert seen == ["http://127.0.0.1:7401/"]


def test_schedule_browser_open_swallows_opener_exceptions() -> None:
    from evalyn_dashboard.server import _schedule_browser_open

    def boom(_url: str) -> bool:
        raise RuntimeError("no browser available")

    t = _schedule_browser_open(
        _FakeServer(started=True),
        "http://127.0.0.1:7401/",
        open_delay=0.0,
        opener=boom,
    )
    # Thread must exit cleanly even when opener raises.
    t.join(timeout=2.0)
    assert not t.is_alive()


def test_schedule_browser_open_times_out_when_server_never_starts() -> None:
    from evalyn_dashboard.server import _schedule_browser_open

    seen: list[str] = []
    t = _schedule_browser_open(
        _FakeServer(started=False),
        "http://127.0.0.1:7401/",
        open_delay=0.0,
        timeout=0.1,
        opener=lambda url: seen.append(url) or True,
    )
    t.join(timeout=2.0)
    # Even if the server never reports started, the opener still fires
    # after the timeout window so the operator at least gets a tab.
    assert seen == ["http://127.0.0.1:7401/"]


# ---- /api/files and /api/runs return empty arrays when .evalyn/ is absent --


@pytest.mark.parametrize(
    "path",
    [
        "/api/files/tree",
        "/api/runs",
    ],
)
def test_files_and_runs_return_empty_when_no_evalyn(tmp_path, monkeypatch, path: str) -> None:
    monkeypatch.chdir(tmp_path)
    client = TestClient(build_app())
    r = client.get(path)
    assert r.status_code == 200
    assert r.json() == []


def test_stub_post_routes_require_csrf() -> None:
    """Mutating stub routes still 403 without the workbench token."""

    client = TestClient(build_app())
    r = client.post("/api/cli/run", json={})
    assert r.status_code == 403


def test_settings_get_returns_redacted_view() -> None:
    """``/api/settings`` is no longer a stub; it returns the public view."""
    client = TestClient(build_app())
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert "providers" in body and "active" in body
