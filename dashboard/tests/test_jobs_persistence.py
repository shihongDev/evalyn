"""Tests for the sqlite-backed JobManager mirror.

Covers the iteration-7 backend gap: jobs that have been evicted from the
in-memory JobManager (server restart, LRU prune) must still be queryable
via ``GET /api/jobs/{id}`` so the Recent Jobs drawer can show their
status instead of "unknown".
"""

from __future__ import annotations

import re
import sys
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from evalyn_dashboard.jobs import JobManager
from evalyn_dashboard.jobs_persistence import JobPersistence, MAX_PERSISTED_OUTPUT
from evalyn_dashboard.server import build_app


def _token_from(client: TestClient) -> str:
    html = client.get("/").text
    m = re.search(r'content="([^"]+)"', html)
    assert m
    return m.group(1)


def _spawn(client: TestClient, app, cmd, *, cli_id: str = "", args: dict | None = None):
    return client.portal.call(
        lambda: app.state.job_manager.spawn(cmd, cli_id=cli_id, args=args or {})
    )


def _wait(client: TestClient, app, job_id: str, timeout: float = 5.0) -> None:
    client.portal.call(app.state.job_manager.wait, job_id, timeout)


# ---------------------------------------------------------------------------
# Test 1: spawn job, verify sqlite has terminal-status row
# ---------------------------------------------------------------------------


def test_spawn_persists_terminal_status(tmp_path: Path) -> None:
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    app = build_app()
    app.state.job_manager = JobManager(persistence=persistence)

    with TestClient(app) as client:
        job_id = _spawn(
            client,
            app,
            [sys.executable, "-c", "print('hi'); import sys; sys.stderr.write('err\\n')"],
            cli_id="echo-test",
            args={"foo": 1},
        )
        _wait(client, app, job_id)

    row = persistence.get(job_id)
    assert row is not None
    assert row["status"] == "complete"
    assert row["exit_code"] == 0
    assert row["cli_id"] == "echo-test"
    assert row["args"] == {"foo": 1}
    assert "hi" in (row["stdout_tail"] or "")
    assert "err" in (row["stderr_tail"] or "")
    assert row["duration_s"] is not None and row["duration_s"] >= 0


# ---------------------------------------------------------------------------
# Test 2: After in-memory eviction, /api/jobs/{id} still resolves
# ---------------------------------------------------------------------------


def test_api_falls_through_to_sqlite_after_eviction(tmp_path: Path) -> None:
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    app = build_app()
    app.state.job_manager = JobManager(persistence=persistence)

    with TestClient(app) as client:
        job_id = _spawn(
            client,
            app,
            [sys.executable, "-c", "print('alive')"],
            cli_id="echo",
            args={},
        )
        _wait(client, app, job_id)

        # Simulate restart: drop the in-memory store entirely but keep the
        # same sqlite mirror attached.
        app.state.job_manager = JobManager(persistence=persistence)
        assert app.state.job_manager.get(job_id) is None

        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == job_id
        assert body["state"] == "complete"
        assert body["exit_code"] == 0


# ---------------------------------------------------------------------------
# Test 3: stdout/stderr tails capped at MAX_PERSISTED_OUTPUT
# ---------------------------------------------------------------------------


def test_output_tails_capped(tmp_path: Path) -> None:
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    jm = JobManager(persistence=persistence)

    # Write a row directly with oversized output via append_output, simulating
    # a chatty job. Easier than producing >16KB from a subprocess in CI.
    persistence.upsert_job(
        job_id="abc",
        cli_id="x",
        args={},
        cmd="x",
        status="running",
        started_at_iso="2026-01-01T00:00:00+00:00",
    )
    big_chunk = "x" * 5000
    for _ in range(10):  # 50_000 chars total -> way past 16384 cap
        persistence.append_output("abc", "stdout", big_chunk)

    row = persistence.get("abc")
    assert row is not None
    assert len(row["stdout_tail"]) <= MAX_PERSISTED_OUTPUT

    # Also exercise set_output_tails directly.
    persistence.set_output_tails("abc", "y" * 50000, "z" * 50000)
    row = persistence.get("abc")
    assert len(row["stdout_tail"]) == MAX_PERSISTED_OUTPUT
    assert len(row["stderr_tail"]) == MAX_PERSISTED_OUTPUT
    # Tail is the last N chars (rolling window).
    assert row["stdout_tail"].endswith("y")
    assert row["stderr_tail"].endswith("z")

    # The unused jm reference keeps a JobManager wired to the same persistence
    # exercised end-to-end by other tests; here we just assert it constructed.
    assert jm._persistence is persistence


# ---------------------------------------------------------------------------
# Test 4: delete_old keeps only the N most recent rows
# ---------------------------------------------------------------------------


def test_delete_old_keeps_n_most_recent(tmp_path: Path) -> None:
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)

    # Insert 150 rows with strictly increasing started_at_iso so ordering
    # is deterministic.
    for i in range(150):
        persistence.upsert_job(
            job_id=f"job-{i:03d}",
            cli_id="x",
            args={},
            cmd="x",
            status="complete",
            started_at_iso=f"2026-01-01T00:00:{i:02d}.000000+00:00"
            if i < 60
            else f"2026-01-01T00:{(i // 60):02d}:{i % 60:02d}.000000+00:00",
        )

    # Sanity: 150 rows persisted.
    with sqlite3.connect(str(db)) as conn:
        before = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    assert before == 150

    deleted = persistence.delete_old(keep=100)
    assert deleted >= 50

    with sqlite3.connect(str(db)) as conn:
        after = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    assert after == 100

    # Surviving rows must be the 100 newest.
    rows = persistence.list_recent(limit=200)
    assert len(rows) == 100
    surviving_ids = {r["job_id"] for r in rows}
    # The newest 100 (job-050 .. job-149) should survive.
    assert "job-149" in surviving_ids
    assert "job-049" not in surviving_ids


# ---------------------------------------------------------------------------
# Test 5: /api/jobs/recent merges in-memory + persisted, in-memory wins
# ---------------------------------------------------------------------------


def test_recent_merges_persisted_with_in_memory(tmp_path: Path) -> None:
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    app = build_app()
    app.state.job_manager = JobManager(persistence=persistence)

    with TestClient(app) as client:
        # Spawn two real jobs through the manager so they hit sqlite.
        job_a = _spawn(client, app, [sys.executable, "-c", "print('a')"])
        _wait(client, app, job_a)
        job_b = _spawn(client, app, [sys.executable, "-c", "print('b')"])
        _wait(client, app, job_b)

        # Drop the in-memory store. /api/jobs/recent should still return the
        # rows from sqlite.
        app.state.job_manager = JobManager(persistence=persistence)

        r = client.get("/api/jobs/recent?limit=10")
        assert r.status_code == 200
        body = r.json()
        ids = [entry["id"] for entry in body]
        assert job_a in ids
        assert job_b in ids
        # Reverse chronological: job_b (newer) before job_a.
        assert ids.index(job_b) < ids.index(job_a)


# ---------------------------------------------------------------------------
# Test 6: JobManager periodic GC bounds sqlite mirror size
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persistence_gc_runs_on_terminal(tmp_path: Path) -> None:
    """Each terminal event increments a counter; every Nth terminal calls
    delete_old(keep=K) so the sqlite jobs table stays bounded. Without
    this, the table grew unboundedly because no caller invoked
    JobPersistence.delete_old."""
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    # Pre-populate 50 historical rows so a single GC has work to do.
    for i in range(50):
        persistence.upsert_job(
            job_id=f"old-{i:03d}",
            cli_id="x",
            args={},
            cmd="x",
            status="complete",
            started_at_iso=f"2026-01-01T00:00:{i:02d}.000000+00:00",
        )

    # Tiny thresholds: GC fires on the very first terminal, keep only 10.
    jm = JobManager(
        persistence=persistence,
        persistence_keep=10,
        persistence_gc_interval=1,
    )

    job_id = await jm.spawn([sys.executable, "-c", "pass"])
    await jm.wait(job_id, timeout=5)

    # After the terminal, GC ran with keep=10. The rows that survive are
    # the 10 newest; "old-049" plus the freshly-finished job, etc.
    rows = persistence.list_recent(limit=200)
    assert len(rows) == 10, f"expected GC to cap at 10, got {len(rows)} rows"
    # The just-finished job survives (it's the newest).
    surviving_ids = {r["job_id"] for r in rows}
    assert job_id in surviving_ids


@pytest.mark.asyncio
async def test_persistence_gc_skipped_below_interval(tmp_path: Path) -> None:
    """GC fires every Nth terminal; before that it must not prune. Verifies
    we are not running delete_old on every terminal (which would be
    wasteful)."""
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    for i in range(50):
        persistence.upsert_job(
            job_id=f"old-{i:03d}",
            cli_id="x",
            args={},
            cmd="x",
            status="complete",
            started_at_iso=f"2026-01-01T00:00:{i:02d}.000000+00:00",
        )

    # gc_interval=10 means the first 9 terminals do not trigger GC.
    jm = JobManager(
        persistence=persistence,
        persistence_keep=5,
        persistence_gc_interval=10,
    )

    job_id = await jm.spawn([sys.executable, "-c", "pass"])
    await jm.wait(job_id, timeout=5)

    # Only one terminal so far; GC has not fired (interval=10). The 50
    # pre-populated rows plus the new one are all still present.
    rows = persistence.list_recent(limit=200)
    assert len(rows) == 51


def test_persistence_gc_no_op_when_persistence_disabled() -> None:
    """When the manager has no persistence, _persist_gc_maybe must be a
    safe no-op rather than raising on a None deref."""
    jm = JobManager()  # default: persistence=None
    # Call directly; would raise if it tried to use a None persistence.
    jm._persist_gc_maybe()
    jm._persist_gc_maybe()
    jm._persist_gc_maybe()
