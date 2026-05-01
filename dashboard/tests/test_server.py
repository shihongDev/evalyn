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
