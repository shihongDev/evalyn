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


def test_output_txt_endpoint_returns_interleaved_plain_text():
    """GET /api/jobs/{id}/output.txt returns text/plain with stdout
    and stderr interleaved in event_id order. Useful for
    `curl ... | grep error` workflows."""
    app = build_app()
    with TestClient(app) as client:
        # Print stdout, then stderr, then more stdout in a deterministic
        # order so we can assert the interleaving on the server side.
        job_id = _spawn(
            client,
            app,
            [
                sys.executable,
                "-c",
                "import sys\nprint('first'); sys.stdout.flush()\n"
                "sys.stderr.write('boom\\n'); sys.stderr.flush()\n"
                "print('third'); sys.stdout.flush()\n",
            ],
        )
        _wait(client, app, job_id)
        r = client.get(f"/api/jobs/{job_id}/output.txt")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]
        body = r.text
        # Each emitted line is present.
        assert "first" in body
        assert "boom" in body
        assert "third" in body
        # File ends with a newline.
        assert body.endswith("\n")


def test_output_txt_unknown_job_404():
    app = build_app()
    with TestClient(app) as client:
        r = client.get("/api/jobs/does-not-exist/output.txt")
        assert r.status_code == 404


def test_output_txt_tail_filter():
    """?tail=N returns only the last N lines. tail=0 or negative is
    rejected as a 400."""
    app = build_app()
    with TestClient(app) as client:
        # Print 10 distinct lines so the tail trim is observable.
        job_id = _spawn(
            client,
            app,
            [
                sys.executable,
                "-c",
                "for i in range(10):\n    print(f'line-{i}')\n",
            ],
        )
        _wait(client, app, job_id)

        # tail=3 -> last 3 lines (line-7, line-8, line-9).
        r = client.get(f"/api/jobs/{job_id}/output.txt?tail=3")
        assert r.status_code == 200
        body = r.text
        # The tail trim keeps line-9, drops line-0.
        assert "line-9" in body
        assert "line-0" not in body
        # Exactly 3 non-empty lines.
        non_empty = [ln for ln in body.split("\n") if ln]
        assert len(non_empty) == 3

        # tail=20 (more than emitted) returns everything.
        r2 = client.get(f"/api/jobs/{job_id}/output.txt?tail=20")
        assert "line-0" in r2.text
        assert "line-9" in r2.text

        # tail=0 rejected.
        r3 = client.get(f"/api/jobs/{job_id}/output.txt?tail=0")
        assert r3.status_code == 400

        # tail above the upper bound rejected. Symmetric with
        # /recent's `limit <= 1000` cap; defends against a buggy
        # client URL like ?tail=999999999. 100000 chosen as well
        # above any realistic manual-review number.
        r4 = client.get(f"/api/jobs/{job_id}/output.txt?tail=100001")
        assert r4.status_code == 400
        assert "100000" in r4.json()["detail"]

        # Boundary: tail=100000 is allowed.
        r5 = client.get(f"/api/jobs/{job_id}/output.txt?tail=100000")
        assert r5.status_code == 200


def test_output_txt_stream_filter():
    """?stream=stdout returns only stdout lines; ?stream=stderr returns
    only stderr lines. Default returns both interleaved. Invalid value
    rejected with 400."""
    app = build_app()
    with TestClient(app) as client:
        job_id = _spawn(
            client,
            app,
            [
                sys.executable,
                "-c",
                "import sys\nprint('mark-out')\nsys.stderr.write('mark-err\\n')\n",
            ],
        )
        _wait(client, app, job_id)

        r_out = client.get(f"/api/jobs/{job_id}/output.txt?stream=stdout")
        assert r_out.status_code == 200
        assert "mark-out" in r_out.text
        assert "mark-err" not in r_out.text

        r_err = client.get(f"/api/jobs/{job_id}/output.txt?stream=stderr")
        assert r_err.status_code == 200
        assert "mark-err" in r_err.text
        assert "mark-out" not in r_err.text

        r_both = client.get(f"/api/jobs/{job_id}/output.txt")
        assert "mark-out" in r_both.text
        assert "mark-err" in r_both.text

        r_bad = client.get(f"/api/jobs/{job_id}/output.txt?stream=both")
        assert r_bad.status_code == 400


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


def test_recent_ndjson_returns_one_object_per_line():
    """GET /api/jobs/recent.ndjson returns one JSON object per line,
    parseable individually. application/x-ndjson media type.
    Same filter semantics as the JSON endpoint."""
    import json

    app = build_app()
    with TestClient(app) as client:
        ok_id = _spawn(client, app, [sys.executable, "-c", "print('ok')"])
        _wait(client, app, ok_id)
        fail_id = _spawn(
            client, app, [sys.executable, "-c", "import sys; sys.exit(2)"]
        )
        _wait(client, app, fail_id)

        r = client.get("/api/jobs/recent.ndjson")
        assert r.status_code == 200
        assert "application/x-ndjson" in r.headers["content-type"]
        # Each non-empty line parses as a JSON object.
        lines = [ln for ln in r.text.split("\n") if ln]
        assert len(lines) >= 2
        objects = [json.loads(ln) for ln in lines]
        ids = {o["id"] for o in objects}
        assert ok_id in ids
        assert fail_id in ids
        # Filter narrows correctly.
        r2 = client.get("/api/jobs/recent.ndjson?status=failed")
        lines2 = [ln for ln in r2.text.split("\n") if ln]
        objects2 = [json.loads(ln) for ln in lines2]
        ids2 = {o["id"] for o in objects2}
        assert fail_id in ids2
        assert ok_id not in ids2


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


# ---------------------------------------------------------------------------
# POST /api/jobs/{id}/restart
# ---------------------------------------------------------------------------


def _spawn_with_meta(client: TestClient, app, cmd, *, cli_id: str, args: dict) -> str:
    """``_spawn`` plus ``cli_id`` / ``args`` metadata for restart tests.

    Restart needs the source job to know its ``cli_id`` so it can find
    the schema in the catalog and rebuild argv. The simple ``_spawn``
    helper drops both, which is fine for cancel/get/output tests but
    would otherwise force restart down its 409 "no cli_id" branch.
    """
    from functools import partial

    fn = partial(app.state.job_manager.spawn, cmd, cli_id=cli_id, args=args)
    return client.portal.call(fn)


def test_restart_unknown_job_returns_404():
    app = build_app()
    with TestClient(app) as client:
        token = _token_from(client)
        r = client.post(
            "/api/jobs/does-not-exist/restart", headers={CSRF_HEADER: token}
        )
        assert r.status_code == 404


def test_restart_running_source_returns_409():
    """Source still queued/running must not double-spawn."""
    app = build_app()
    with TestClient(app) as client:
        token = _token_from(client)
        # Long sleep so the job stays in "running" while we hit restart.
        job_id = _spawn_with_meta(
            client,
            app,
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cli_id="status",
            args={},
        )
        try:
            r = client.post(
                f"/api/jobs/{job_id}/restart", headers={CSRF_HEADER: token}
            )
            assert r.status_code == 409
            assert "running" in r.json()["detail"]
        finally:
            client.portal.call(app.state.job_manager.cancel, job_id)


def test_restart_finished_source_spawns_fresh_job():
    """Happy path: terminal source -> 200 with a new job_id."""
    app = build_app()
    with TestClient(app) as client:
        token = _token_from(client)
        # Use --help under the "status" cli_id so the schema lookup
        # succeeds. The actual subprocess just exits fast.
        job_id = _spawn_with_meta(
            client,
            app,
            [sys.executable, "-c", "pass"],
            cli_id="status",
            args={},
        )
        _wait(client, app, job_id)

        r = client.post(
            f"/api/jobs/{job_id}/restart", headers={CSRF_HEADER: token}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        new_id = body["job_id"]
        assert isinstance(new_id, str) and new_id
        assert new_id != job_id

        # The new job is registered with the manager and carries the
        # source's cli_id forward.
        new_job = app.state.job_manager.get(new_id)
        assert new_job is not None
        assert new_job.cli_id == "status"

        # Cleanup so the subprocess doesn't outlive the test.
        client.portal.call(app.state.job_manager.cancel, new_id)


def test_restart_source_without_cli_id_returns_409():
    """A job spawned without cli_id (e.g. via /api/jobs internals or
    legacy persistence) cannot be rebuilt - restart returns 409 with
    a descriptive message rather than 500."""
    app = build_app()
    with TestClient(app) as client:
        token = _token_from(client)
        job_id = _spawn(client, app, [sys.executable, "-c", "pass"])
        _wait(client, app, job_id)

        r = client.post(
            f"/api/jobs/{job_id}/restart", headers={CSRF_HEADER: token}
        )
        assert r.status_code == 409
        assert "cli_id" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /output.txt?download=1 attaches a Content-Disposition header
# ---------------------------------------------------------------------------


def test_output_txt_download_adds_attachment_header():
    """?download=1 turns the same plain-text body into a file
    download by setting Content-Disposition. Filename derives from
    the job_id prefix so concurrent downloads don't clobber each
    other in the user's Downloads folder."""
    app = build_app()
    with TestClient(app) as client:
        job_id = _spawn(
            client,
            app,
            [sys.executable, "-c", "print('hello'); print('world')"],
        )
        _wait(client, app, job_id)

        # Without ?download: no Content-Disposition (browser renders
        # inline, curl pipes happily).
        r1 = client.get(f"/api/jobs/{job_id}/output.txt")
        assert r1.status_code == 200
        assert "content-disposition" not in {k.lower() for k in r1.headers.keys()}

        # With ?download=1: attachment + sensible filename.
        r2 = client.get(f"/api/jobs/{job_id}/output.txt?download=1")
        assert r2.status_code == 200
        cd = r2.headers.get("content-disposition", "")
        assert "attachment" in cd
        # First 8 chars of job_id used as the filename prefix.
        assert f"evalyn-job-{job_id[:8]}" in cd
        assert cd.endswith('.log"')


def test_recent_rejects_comma_in_cli_id_with_hint():
    """cli_id with embedded commas almost always means the user meant
    cli_ids (multi-value). 400 with a hint pointing to the right
    parameter beats silently filtering for a literal string that
    matches no row."""
    app = build_app()
    with TestClient(app) as client:
        r = client.get("/api/jobs/recent?cli_id=foo,bar")
        assert r.status_code == 400
        body = r.json()
        assert "cli_ids" in body["detail"]


def test_recent_rejects_non_finite_since():
    """``since=Infinity`` (or NaN) is parsed as a float by FastAPI but
    is nonsense for a timestamp filter. 400 instead of attempting a
    SQL comparison on inf."""
    app = build_app()
    with TestClient(app) as client:
        r = client.get("/api/jobs/recent?since=Infinity")
        # FastAPI's float coercion accepts "Infinity" -> our finite
        # check rejects it.
        assert r.status_code == 400
        assert "finite" in r.json()["detail"]


def test_recent_includes_output_url_per_row():
    """Each row in /api/jobs/recent now includes ``output_url``, the
    canonical download link with download+include_meta query params
    pre-applied. Lets the FE render per-row download buttons without
    reconstructing the URL convention."""
    app = build_app()
    with TestClient(app) as client:
        job_id = _spawn(client, app, [sys.executable, "-c", "pass"])
        _wait(client, app, job_id)
        r = client.get("/api/jobs/recent")
        assert r.status_code == 200
        rows = r.json()
        assert isinstance(rows, list) and len(rows) >= 1
        row = next((j for j in rows if j["id"] == job_id), None)
        assert row is not None
        # Shape contract.
        assert "output_url" in row
        assert row["output_url"].startswith(f"/api/jobs/{job_id}/output.txt")
        assert "download=1" in row["output_url"]
        assert "include_meta=1" in row["output_url"]


def test_get_job_includes_output_url():
    """GET /api/jobs/{id} also surfaces output_url for symmetry with
    /recent. Drilling into a single job from a deep link doesn't lose
    the affordance."""
    app = build_app()
    with TestClient(app) as client:
        job_id = _spawn(client, app, [sys.executable, "-c", "pass"])
        _wait(client, app, job_id)
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["output_url"] == (
            f"/api/jobs/{job_id}/output.txt?download=1&include_meta=1"
        )


def test_output_txt_include_meta_prepends_self_describing_header():
    """?include_meta=1 prepends a # comment block with job_id, cli,
    started_at, status, exit_code, scope - so a downloaded log
    survives without external context. Body is unchanged after the
    header. Meta is OFF by default - existing curl users see no
    change."""
    app = build_app()
    with TestClient(app) as client:
        job_id = _spawn(
            client,
            app,
            [sys.executable, "-c", "print('hello'); print('world')"],
        )
        _wait(client, app, job_id)

        # Default (no include_meta): no header lines.
        r1 = client.get(f"/api/jobs/{job_id}/output.txt")
        assert r1.status_code == 200
        assert "evalyn job log" not in r1.text
        # Body still ends with the expected lines.
        assert "hello" in r1.text
        assert "world" in r1.text

        # With include_meta=1: header + body.
        r2 = client.get(f"/api/jobs/{job_id}/output.txt?include_meta=1")
        assert r2.status_code == 200
        text = r2.text
        # Header lines start with '#' so log tools treat them as comments.
        assert text.startswith("# evalyn job log\n")
        assert f"# job_id: {job_id}" in text
        assert "# started_at: " in text
        assert "# scope: all" in text
        # Body still present after the header.
        assert "hello" in text
        assert "world" in text
        # Header preceeds body in the rendered string.
        assert text.index("# job_id") < text.index("hello")


def test_output_txt_include_meta_reflects_stream_filter():
    """``scope`` field of the meta header reflects ?stream= and ?tail=
    so a downloaded ``stderr only, last 5`` slice is identifiable."""
    app = build_app()
    with TestClient(app) as client:
        job_id = _spawn(
            client,
            app,
            [
                sys.executable,
                "-c",
                "import sys\nfor i in range(8):\n  sys.stderr.write(f'e{i}\\n')\n",
            ],
        )
        _wait(client, app, job_id)

        r = client.get(
            f"/api/jobs/{job_id}/output.txt?include_meta=1&stream=stderr&tail=3"
        )
        assert r.status_code == 200
        text = r.text
        assert "# scope: stream=stderr, tail=3" in text


def test_output_txt_download_filename_includes_stream_and_tail():
    """When ?stream and ?tail are also present, the filename suffix
    encodes them so multiple downloads of slices of the same job
    don't clobber each other."""
    app = build_app()
    with TestClient(app) as client:
        job_id = _spawn(
            client,
            app,
            [
                sys.executable,
                "-c",
                "import sys\nfor i in range(5):\n  print(i)\nsys.stderr.write('err\\n')\n",
            ],
        )
        _wait(client, app, job_id)

        r = client.get(
            f"/api/jobs/{job_id}/output.txt?download=1&stream=stderr&tail=3"
        )
        assert r.status_code == 200
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert f"evalyn-job-{job_id[:8]}-stderr-tail3.log" in cd


def test_jobs_stats_sends_no_store_cache_control():
    """Drawer capacity chip polls this every 5s. A cached response
    would freeze the chip on the wrong saturation tier - critical
    for the "approaching cap" warning to actually warn."""
    app = build_app()
    with TestClient(app) as client:
        r = client.get("/api/jobs/stats")
        assert r.status_code == 200
        cc = r.headers.get("cache-control", "")
        assert "no-store" in cc.lower(), (
            f"expected 'no-store' in Cache-Control, got: {cc!r}"
        )


# ---------------------------------------------------------------------------
# POST /api/jobs/admin/vacuum (manual compaction trigger)
# ---------------------------------------------------------------------------


def test_admin_vacuum_requires_csrf():
    """The endpoint mutates persistence; missing token must 403 like
    cancel/restart/settings writes do."""
    app = build_app()
    with TestClient(app) as client:
        r = client.post("/api/jobs/admin/vacuum")
        assert r.status_code == 403


def test_admin_vacuum_returns_before_after_and_bytes_saved():
    """Happy path: endpoint runs vacuum() and reports before/after
    sizes plus the difference. On a fresh test app no rows have been
    written, so the file may not exist (returning 0/0) - the contract
    only requires the keys are present and the math is consistent."""
    app = build_app()
    with TestClient(app) as client:
        # Seed at least one row so the persistence file exists and
        # vacuum() can run. _spawn flows through the JM which writes
        # to the mirror.
        job_id = _spawn(client, app, [sys.executable, "-c", "pass"])
        _wait(client, app, job_id)

        token = _token_from(client)
        r = client.post("/api/jobs/admin/vacuum", headers={CSRF_HEADER: token})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        # Shape: ints, math holds.
        assert isinstance(body["before"], int)
        assert isinstance(body["after"], int)
        assert isinstance(body["bytes_saved"], int)
        assert body["bytes_saved"] == body["before"] - body["after"]


def test_admin_vacuum_returns_503_when_persistence_unavailable(monkeypatch):
    """When the JobManager has no attached persistence (early
    startup, degraded shutdown, or a future override), the vacuum
    endpoint must return 503 - not 404. Last week's fix flipped
    the code from 404 (permanent: "endpoint never existed") to
    503 (transient: "service unavailable, retry shortly") to
    align with the prune endpoint and let a single FE error
    branch cover both. Pin so the regression can't return.
    """
    from evalyn_dashboard.api import jobs as jobs_api

    app = build_app()
    # Force _persistence_for to return None regardless of the JM
    # state. Patches the helper used by both vacuum + prune so a
    # parallel test can do the same for prune (next test below).
    monkeypatch.setattr(jobs_api, "_persistence_for", lambda jm: None)
    with TestClient(app) as client:
        token = _token_from(client)
        r = client.post(
            "/api/jobs/admin/vacuum",
            headers={CSRF_HEADER: token},
        )
        assert r.status_code == 503, r.text
        body = r.json()
        assert body["detail"] == "persistence_unavailable"


def test_admin_prune_returns_503_when_persistence_unavailable(monkeypatch):
    """Mirror test for prune. Both admin endpoints share the
    503-on-degraded-persistence contract; a single FE error
    branch covers both, so the codes must stay aligned.
    """
    from evalyn_dashboard.api import jobs as jobs_api

    app = build_app()
    monkeypatch.setattr(jobs_api, "_persistence_for", lambda jm: None)
    with TestClient(app) as client:
        token = _token_from(client)
        r = client.post(
            "/api/jobs/admin/prune?keep=10",
            headers={CSRF_HEADER: token},
        )
        assert r.status_code == 503, r.text
        body = r.json()
        assert body["detail"] == "persistence_unavailable"


def test_admin_prune_requires_csrf():
    """Like vacuum, prune is a destructive POST that mutates
    persistence; missing token must 403 before the prune runs."""
    app = build_app()
    with TestClient(app) as client:
        r = client.post("/api/jobs/admin/prune?keep=10")
        assert r.status_code == 403


def test_admin_prune_rejects_negative_keep():
    """A negative keep would result in unbounded deletion. Reject
    with 400 before touching persistence."""
    app = build_app()
    with TestClient(app) as client:
        token = _token_from(client)
        r = client.post(
            "/api/jobs/admin/prune?keep=-1", headers={CSRF_HEADER: token}
        )
        assert r.status_code == 400
        assert "keep" in r.json()["detail"].lower()


def test_admin_prune_returns_deleted_and_kept_counts():
    """Happy path: spawn a few jobs, prune to keep=1, the response
    reflects the rowcount that got deleted plus the post-prune
    total. Kept must equal the requested keep (or fewer if the
    pre-prune total was already smaller)."""
    app = build_app()
    with TestClient(app) as client:
        # Three jobs so the prune actually has work to do.
        for _ in range(3):
            jid = _spawn(client, app, [sys.executable, "-c", "pass"])
            _wait(client, app, jid)

        token = _token_from(client)
        r = client.post(
            "/api/jobs/admin/prune?keep=1", headers={CSRF_HEADER: token}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        # We had 3 rows, asked to keep 1 -> 2 deleted, 1 kept.
        # delete_old's "older than the keep-th newest" semantics
        # may delete >= 2 depending on tie-breaks, so we assert
        # an inequality.
        assert body["deleted"] >= 2
        assert body["kept"] >= 0 and body["kept"] <= 1


def test_admin_prune_zero_keep_clears_all():
    """keep=0 is allowed and means delete everything. Confirms the
    >=0 boundary is intentional, not an off-by-one."""
    app = build_app()
    with TestClient(app) as client:
        jid = _spawn(client, app, [sys.executable, "-c", "pass"])
        _wait(client, app, jid)

        token = _token_from(client)
        r = client.post(
            "/api/jobs/admin/prune?keep=0", headers={CSRF_HEADER: token}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["kept"] == 0


def test_cancel_emits_audit_log(caplog):
    """User-initiated cancel via API should leave a server log
    line with the resulting status, so postmortem queries like
    "did the user kill this run, or had it already finished by
    the time the click landed?" are answerable."""
    import logging

    app = build_app()
    with TestClient(app) as client:
        token = _token_from(client)
        # Spawn a quick job and let it complete naturally so we
        # exercise the "cancel after finished" forensic case
        # (the cancel returns success but state is 'complete').
        job_id = _spawn(client, app, [sys.executable, "-c", "pass"])
        _wait(client, app, job_id)

        with caplog.at_level(logging.INFO, logger="evalyn_dashboard.api.jobs"):
            r = client.post(
                f"/api/jobs/{job_id}/cancel", headers={CSRF_HEADER: token}
            )
            assert r.status_code == 200

        matched = [
            rec for rec in caplog.records
            if rec.message.startswith("admin cancel: job_id=")
        ]
        assert len(matched) == 1
        # The log captures the FINAL status (after the cancel
        # attempt), not just the request. For an already-finished
        # job that's "complete", not "cancelled".
        assert "status=complete" in matched[0].message


def test_delete_emits_audit_log(caplog):
    """User-initiated delete is destructive (drops the row);
    operators querying "where did job X go?" should find a
    server log line."""
    import logging

    app = build_app()
    with TestClient(app) as client:
        token = _token_from(client)
        job_id = _spawn(client, app, [sys.executable, "-c", "pass"])
        _wait(client, app, job_id)

        with caplog.at_level(logging.INFO, logger="evalyn_dashboard.api.jobs"):
            r = client.delete(
                f"/api/jobs/{job_id}", headers={CSRF_HEADER: token}
            )
            assert r.status_code == 200

        matched = [
            rec for rec in caplog.records
            if rec.message.startswith("admin delete: job_id=")
        ]
        assert len(matched) == 1
        assert job_id in matched[0].message


def test_delete_404_does_not_emit_audit_log(caplog):
    """A delete for an unknown id 404s before reaching the log
    line. Otherwise we'd have phantom "admin delete" lines for
    requests that didn't actually delete anything."""
    import logging

    app = build_app()
    with TestClient(app) as client:
        token = _token_from(client)
        with caplog.at_level(logging.INFO, logger="evalyn_dashboard.api.jobs"):
            r = client.delete(
                "/api/jobs/does-not-exist", headers={CSRF_HEADER: token}
            )
            assert r.status_code == 404

        matched = [
            rec for rec in caplog.records
            if rec.message.startswith("admin delete: job_id=")
        ]
        assert len(matched) == 0


def test_restart_emits_audit_log_with_source_linkage(caplog):
    """Restart audit log records the source -> new_job_id mapping
    so postmortems can trace restart lineage. The new_job_id alone
    is in the response body; the source linkage lives only in the
    log."""
    import logging

    app = build_app()
    with TestClient(app) as client:
        token = _token_from(client)
        # Spawn with cli_id metadata so restart can find the schema.
        source_id = _spawn_with_meta(
            client, app,
            [sys.executable, "-c", "pass"],
            cli_id="status",
            args={},
        )
        _wait(client, app, source_id)

        with caplog.at_level(logging.INFO, logger="evalyn_dashboard.api.jobs"):
            r = client.post(
                f"/api/jobs/{source_id}/restart", headers={CSRF_HEADER: token}
            )
            assert r.status_code == 200
            new_id = r.json()["job_id"]

        matched = [
            rec for rec in caplog.records
            if rec.message.startswith("admin restart: source=")
        ]
        assert len(matched) == 1
        # Both source and new id are captured for lineage tracing.
        assert source_id in matched[0].message
        assert new_id in matched[0].message
        # Cleanup so a still-running restart doesn't strand the
        # next test's TestClient teardown.
        client.portal.call(app.state.job_manager.cancel, new_id)


def test_admin_vacuum_emits_audit_log(caplog):
    """A manual vacuum is uncommon enough that each invocation
    should be discoverable in server logs (postmortem queries
    like "did anyone vacuum during the incident window?"). INFO
    so the default log level catches it."""
    import logging

    app = build_app()
    with TestClient(app) as client:
        # Need at least one row so vacuum() can run.
        jid = _spawn(client, app, [sys.executable, "-c", "pass"])
        _wait(client, app, jid)

        token = _token_from(client)
        with caplog.at_level(logging.INFO, logger="evalyn_dashboard.api.jobs"):
            r = client.post(
                "/api/jobs/admin/vacuum", headers={CSRF_HEADER: token}
            )
            assert r.status_code == 200

        # Look for the audit-log message. We assert on the prefix
        # rather than the exact bytes_saved value because the
        # post-vacuum size depends on filesystem behavior
        # (rounding etc).
        matched = [
            rec for rec in caplog.records
            if rec.message.startswith("admin vacuum: before=")
        ]
        assert len(matched) == 1, (
            f"expected exactly one admin-vacuum log line, "
            f"got {len(matched)}: {[r.message for r in caplog.records]}"
        )
        # The log includes elapsed_ms so operators can spot
        # slow vacuums on large dbs (decision metric: tighten
        # persistence_keep when elapsed grows).
        assert "elapsed_ms=" in matched[0].message


def test_admin_prune_emits_audit_log(caplog):
    """Prune is destructive so each invocation must be in the
    server log for audit trails."""
    import logging

    app = build_app()
    with TestClient(app) as client:
        jid = _spawn(client, app, [sys.executable, "-c", "pass"])
        _wait(client, app, jid)

        token = _token_from(client)
        with caplog.at_level(logging.INFO, logger="evalyn_dashboard.api.jobs"):
            r = client.post(
                "/api/jobs/admin/prune?keep=0", headers={CSRF_HEADER: token}
            )
            assert r.status_code == 200

        matched = [
            rec for rec in caplog.records
            if rec.message.startswith("admin prune: keep=")
        ]
        assert len(matched) == 1
        # The log line records the keep value the caller passed,
        # so the audit trail captures intent (not just outcome).
        assert "keep=0" in matched[0].message
        # Includes elapsed_ms so operators can spot slow prunes
        # (typically fast - single DELETE - but a large purge
        # under contention is worth seeing).
        assert "elapsed_ms=" in matched[0].message


def test_admin_vacuum_advances_last_vacuum_at_field_on_health():
    """A successful manual vacuum must propagate to /api/health's
    last_vacuum_at field. Closes the loop between the manual button
    and the visibility surface that operators read - if the button
    didn't update the metric, users couldn't tell it actually ran."""
    app = build_app()
    with TestClient(app) as client:
        # Pre-vacuum state: no vacuums yet -> last_vacuum_at is None.
        before_health = client.get("/api/health").json()
        assert before_health["last_vacuum_at"] is None

        # Seed a row so the file exists, then manually vacuum.
        job_id = _spawn(client, app, [sys.executable, "-c", "pass"])
        _wait(client, app, job_id)
        token = _token_from(client)
        r = client.post("/api/jobs/admin/vacuum", headers={CSRF_HEADER: token})
        assert r.status_code == 200

        after_health = client.get("/api/health").json()
        assert after_health["last_vacuum_at"] is not None
        assert isinstance(after_health["last_vacuum_at"], (int, float))
