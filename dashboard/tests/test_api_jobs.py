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


def test_output_endpoint_returns_in_memory_tails():
    """GET /api/jobs/{id}/output returns assembled stdout/stderr tails
    for a finished in-memory job. Useful for clients that want the
    final output without setting up a WebSocket."""
    app = build_app()
    with TestClient(app) as client:
        job_id = _spawn(
            client,
            app,
            [
                sys.executable,
                "-c",
                "import sys\nprint('hello')\nsys.stderr.write('boom\\n')\n",
            ],
        )
        _wait(client, app, job_id)
        r = client.get(f"/api/jobs/{job_id}/output")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == job_id
        assert body["state"] in ("complete", "failed")
        assert "hello" in body["stdout_tail"]
        assert "boom" in body["stderr_tail"]
        assert body["stderr_count"] == 1
        assert body["total_chars"] == len(body["stdout_tail"]) + len(
            body["stderr_tail"]
        )


def test_output_endpoint_unknown_job_404():
    app = build_app()
    with TestClient(app) as client:
        r = client.get("/api/jobs/does-not-exist/output")
        assert r.status_code == 404


def test_delete_finished_job_removes_from_history():
    """DELETE /api/jobs/{id} removes a finished job from in-memory.
    A subsequent GET returns 404."""
    app = build_app()
    with TestClient(app) as client:
        token = _token_from(client)
        job_id = _spawn(client, app, [sys.executable, "-c", "pass"])
        _wait(client, app, job_id)

        # Sanity: GET works pre-delete.
        assert client.get(f"/api/jobs/{job_id}").status_code == 200

        r = client.delete(
            f"/api/jobs/{job_id}", headers={CSRF_HEADER: token}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == {"ok": True, "id": job_id}

        # Now 404.
        assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_delete_running_job_returns_409():
    """A queued/running job cannot be deleted - client must cancel first.
    409 prevents an orphaned subprocess + reaper task."""
    app = build_app()
    app.state.job_manager.grace_seconds = 0.3
    with TestClient(app) as client:
        token = _token_from(client)
        job_id = _spawn(
            client, app, [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        try:
            r = client.delete(
                f"/api/jobs/{job_id}", headers={CSRF_HEADER: token}
            )
            assert r.status_code == 409
            # Body should hint at the cancel-first remedy.
            assert "cancel" in r.json()["detail"].lower()
        finally:
            client.post(
                f"/api/jobs/{job_id}/cancel", headers={CSRF_HEADER: token}
            )
            _wait(client, app, job_id)


def test_recent_csv_returns_text_csv_with_header():
    """GET /api/jobs/recent.csv returns a CSV with a header row, one
    data row per matching job, and a Content-Disposition attachment
    header for browser download UX."""
    app = build_app()
    with TestClient(app) as client:
        ok_id = _spawn(client, app, [sys.executable, "-c", "print('ok')"])
        _wait(client, app, ok_id)
        fail_id = _spawn(
            client,
            app,
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('boom\\n'); sys.exit(2)",
            ],
        )
        _wait(client, app, fail_id)

        r = client.get("/api/jobs/recent.csv")
        assert r.status_code == 200
        assert "text/csv" in r.headers["content-type"]
        assert "attachment" in r.headers.get("content-disposition", "")
        body = r.text
        # Header row.
        first = body.splitlines()[0]
        for col in (
            "id",
            "cli_id",
            "state",
            "started_at",
            "ended_at",
            "exit_code",
            "duration",
            "stderr_count",
        ):
            assert col in first
        # Both jobs should appear in subsequent rows.
        assert ok_id in body
        assert fail_id in body


def test_recent_csv_status_filter_narrows_rows():
    """The same filter query params (cli_id, status, since) work on
    the CSV endpoint."""
    app = build_app()
    with TestClient(app) as client:
        ok_id = _spawn(client, app, [sys.executable, "-c", "print('ok')"])
        _wait(client, app, ok_id)
        fail_id = _spawn(
            client, app, [sys.executable, "-c", "import sys; sys.exit(2)"]
        )
        _wait(client, app, fail_id)

        r = client.get("/api/jobs/recent.csv?status=failed")
        assert r.status_code == 200
        body = r.text
        # Only the failed row, not the ok one.
        assert fail_id in body
        assert ok_id not in body


def test_stats_endpoint_aggregates_counts():
    """GET /api/jobs/stats returns counts by status, total stderr, and
    recent failures. Uses real spawned jobs so the persistence path
    runs end-to-end."""
    app = build_app()
    with TestClient(app) as client:
        # Spawn one ok job and one failing job with stderr output.
        ok_id = _spawn(client, app, [sys.executable, "-c", "print('ok')"])
        _wait(client, app, ok_id)
        fail_id = _spawn(
            client,
            app,
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('boom\\n'); sys.exit(2)",
            ],
        )
        _wait(client, app, fail_id)

        r = client.get("/api/jobs/stats")
        assert r.status_code == 200, r.text
        body = r.json()
        # Shape contract.
        assert "total" in body
        assert "by_status" in body
        assert "total_stderr" in body
        assert "recent_failures" in body
        # Both jobs counted.
        assert body["total"] >= 2
        # Status breakdown.
        assert body["by_status"].get("complete", 0) >= 1
        assert body["by_status"].get("failed", 0) >= 1
        # Stderr total >= 1 from the failing job.
        assert body["total_stderr"] >= 1
        # The fail just happened so it falls inside the 24h window.
        assert body["recent_failures"] >= 1


def test_stats_endpoint_rejects_negative_window():
    app = build_app()
    with TestClient(app) as client:
        r = client.get("/api/jobs/stats?recent_window_s=-1")
        assert r.status_code == 400


def test_delete_unknown_job_returns_404():
    app = build_app()
    with TestClient(app) as client:
        token = _token_from(client)
        r = client.delete(
            "/api/jobs/does-not-exist", headers={CSRF_HEADER: token}
        )
        assert r.status_code == 404
