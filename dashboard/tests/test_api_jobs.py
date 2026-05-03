"""Tests for ``/api/jobs/*`` REST endpoints (Lane B1.3, B1.4).

We spawn jobs by directly awaiting ``app.state.job_manager.spawn`` inside
the same TestClient ``portal`` so the subprocess transport stays bound to
the loop FastAPI/Starlette will reuse for subsequent requests. Using
``asyncio.run`` to spawn jobs does NOT work: it tears down the loop the
process handle was created on, and any later request that touches that
handle raises ``RuntimeError: Event loop is closed``.
"""

from __future__ import annotations

import re
import sys

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from evalyn_dashboard.server import CSRF_HEADER, build_app


def _token_from(client: TestClient) -> str:
    html = client.get("/").text
    m = re.search(r'content="([^"]+)"', html)
    assert m
    return m.group(1)


def _spawn(client: TestClient, app, cmd) -> str:
    """Spawn a job on the TestClient's portal loop.

    ``client.portal`` is the anyio portal Starlette uses to drive async
    handlers; calling into it keeps the spawned subprocess on the same
    event loop the API routes will hit later.
    """
    return client.portal.call(app.state.job_manager.spawn, cmd)


def _wait(client: TestClient, app, job_id: str, timeout: float = 5.0) -> None:
    client.portal.call(app.state.job_manager.wait, job_id, timeout)


def test_recent_returns_array_default_limit():
    client = TestClient(build_app())
    r = client.get("/api/jobs/recent")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_recent_with_limit_param():
    app = build_app()
    with TestClient(app) as client:
        ids = []
        for _ in range(3):
            ids.append(_spawn(client, app, [sys.executable, "-c", "pass"]))
        for jid in ids:
            _wait(client, app, jid)
        r = client.get("/api/jobs/recent?limit=2")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 2
        # Reverse chronological: most recent first.
        assert body[0]["id"] == ids[-1]


def test_recent_invalid_limit_400():
    client = TestClient(build_app())
    r = client.get("/api/jobs/recent?limit=0")
    assert r.status_code == 400
    r = client.get("/api/jobs/recent?limit=99999")
    assert r.status_code == 400


def test_get_unknown_job_404():
    client = TestClient(build_app())
    r = client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404


def test_get_known_job_returns_metadata():
    app = build_app()
    with TestClient(app) as client:
        job_id = _spawn(client, app, [sys.executable, "-c", "print('hi')"])
        _wait(client, app, job_id)
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == job_id
        assert "state" in body
        assert "cmd" in body
        assert body["cmd"][:2] == [sys.executable, "-c"]
        # Private fields must not leak.
        for forbidden in ("_process", "_subscribers", "events", "_capture_tasks"):
            assert forbidden not in body


def test_cancel_unknown_job_404():
    client = TestClient(build_app())
    token = _token_from(client)
    r = client.post(
        "/api/jobs/nope/cancel",
        headers={CSRF_HEADER: token},
    )
    assert r.status_code == 404


def test_cancel_requires_csrf():
    client = TestClient(build_app())
    r = client.post("/api/jobs/anything/cancel")
    assert r.status_code == 403


def test_cancel_running_job_returns_cancelled_state():
    app = build_app()
    # Use a small grace_seconds so the test stays fast.
    app.state.job_manager.grace_seconds = 0.3
    with TestClient(app) as client:
        token = _token_from(client)
        job_id = _spawn(
            client, app, [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        r = client.post(
            f"/api/jobs/{job_id}/cancel",
            headers={CSRF_HEADER: token},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["state"] == "cancelled"


def test_cancel_completed_job_is_idempotent():
    app = build_app()
    with TestClient(app) as client:
        token = _token_from(client)
        job_id = _spawn(client, app, [sys.executable, "-c", "pass"])
        _wait(client, app, job_id)
        r = client.post(
            f"/api/jobs/{job_id}/cancel",
            headers={CSRF_HEADER: token},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["state"] in ("complete", "failed")
