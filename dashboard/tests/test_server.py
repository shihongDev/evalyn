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
    assert r.json() == {"ok": True}


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
