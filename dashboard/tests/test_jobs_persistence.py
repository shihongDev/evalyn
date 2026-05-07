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


# ---------------------------------------------------------------------------
# Test 7: stderr_count persisted across runs and surfaces in get/list_recent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persistence_stores_stderr_count(tmp_path: Path) -> None:
    """JobManager passes stderr_count to JobPersistence on terminal status;
    a fresh JobPersistence reading the same DB returns it via get() and
    list_recent()."""
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    jm = JobManager(persistence=persistence)

    job_id = await jm.spawn(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "for i in range(4):\n"
            "    sys.stderr.write(f'err {i}\\n')\n",
        ]
    )
    await jm.wait(job_id, timeout=5)

    # In-memory says 4. Persistence should have it too.
    job = jm.get(job_id)
    assert job is not None
    assert job.stderr_count == 4

    fresh = JobPersistence(db_path=db)
    row = fresh.get(job_id)
    assert row is not None
    assert row.get("stderr_count") == 4


def test_list_recent_filters_by_cli_ids_in_list(tmp_path: Path) -> None:
    """list_recent(cli_ids=[a, b]) returns rows whose cli_id IN that
    list. Empty list is treated as 'no filter' so callers do not need
    to special-case."""
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    for i, cli in enumerate(["alpha", "beta", "gamma", "delta"]):
        persistence.upsert_job(
            job_id=f"j-{cli}",
            cli_id=cli,
            args={},
            cmd=cli,
            status="complete",
            started_at_iso=f"2026-01-01T00:{i:02d}:00.000000+00:00",
        )

    # Two-id list.
    rows = persistence.list_recent(limit=100, cli_ids=["alpha", "gamma"])
    assert {r["cli_id"] for r in rows} == {"alpha", "gamma"}

    # Single-id list (acts like cli_id=).
    rows_single = persistence.list_recent(limit=100, cli_ids=["beta"])
    assert {r["cli_id"] for r in rows_single} == {"beta"}

    # Empty list short-circuits to no filter.
    rows_empty = persistence.list_recent(limit=100, cli_ids=[])
    assert len(rows_empty) == 4

    # cli_ids takes precedence over cli_id when both set.
    rows_both = persistence.list_recent(
        limit=100, cli_id="alpha", cli_ids=["beta", "gamma"]
    )
    assert {r["cli_id"] for r in rows_both} == {"beta", "gamma"}


def test_list_recent_filters_by_before_and_window(tmp_path: Path) -> None:
    """list_recent(before_iso=X) keeps only rows whose started_at_iso < X.
    Combined with since_iso, both clauses give a windowed query."""
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    iso_t0 = "2026-01-01T00:00:00.000000+00:00"
    iso_t5 = "2026-01-01T00:00:05.000000+00:00"
    iso_t10 = "2026-01-01T00:00:10.000000+00:00"
    iso_t15 = "2026-01-01T00:00:15.000000+00:00"
    for jid, ts in [
        ("a", iso_t0),
        ("b", iso_t5),
        ("c", iso_t10),
        ("d", iso_t15),
    ]:
        persistence.upsert_job(
            job_id=jid,
            cli_id="x",
            args={},
            cmd="x",
            status="complete",
            started_at_iso=ts,
        )

    # before only: strictly less than t10 -> a, b
    only_before = persistence.list_recent(limit=100, before_iso=iso_t10)
    assert {r["job_id"] for r in only_before} == {"a", "b"}

    # since AND before: strictly between t0 and t15 -> b, c
    windowed = persistence.list_recent(
        limit=100, since_iso=iso_t0, before_iso=iso_t15
    )
    assert {r["job_id"] for r in windowed} == {"b", "c"}


def test_list_recent_filters_by_since_iso(tmp_path: Path) -> None:
    """list_recent(since_iso=X) keeps only rows whose started_at_iso > X.
    Useful for polling: the caller hands back the previous response's
    max timestamp to fetch only the delta."""
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    iso_baseline = "2026-01-01T00:00:00.000000+00:00"
    iso_after_5s = "2026-01-01T00:00:05.000000+00:00"
    iso_after_10s = "2026-01-01T00:00:10.000000+00:00"
    persistence.upsert_job(
        job_id="early",
        cli_id="x",
        args={},
        cmd="x",
        status="complete",
        started_at_iso=iso_baseline,
    )
    persistence.upsert_job(
        job_id="middle",
        cli_id="x",
        args={},
        cmd="x",
        status="complete",
        started_at_iso=iso_after_5s,
    )
    persistence.upsert_job(
        job_id="late",
        cli_id="x",
        args={},
        cmd="x",
        status="complete",
        started_at_iso=iso_after_10s,
    )

    after_5s = persistence.list_recent(limit=100, since_iso=iso_after_5s)
    # Strictly greater than: 'middle' should NOT be included.
    ids = {r["job_id"] for r in after_5s}
    assert ids == {"late"}

    after_baseline = persistence.list_recent(limit=100, since_iso=iso_baseline)
    ids2 = {r["job_id"] for r in after_baseline}
    assert ids2 == {"middle", "late"}


def test_list_recent_filters_by_status(tmp_path: Path) -> None:
    """list_recent(status=X) pushes the filter to SQL. Combined with
    cli_id, both clauses AND together."""
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    rows_seed = [
        ("alpha-ok-1", "alpha", "complete"),
        ("alpha-ok-2", "alpha", "complete"),
        ("alpha-fail-1", "alpha", "failed"),
        ("beta-ok-1", "beta", "complete"),
        ("beta-fail-1", "beta", "failed"),
        ("beta-fail-2", "beta", "failed"),
    ]
    for i, (jid, cli, status) in enumerate(rows_seed):
        persistence.upsert_job(
            job_id=jid,
            cli_id=cli,
            args={},
            cmd=cli,
            status=status,
            started_at_iso=f"2026-01-01T00:{i:02d}:00.000000+00:00",
        )

    # Status filter alone.
    failed_only = persistence.list_recent(limit=100, status="failed")
    assert len(failed_only) == 3
    assert all(r["status"] == "failed" for r in failed_only)

    # Combined status + cli_id.
    beta_failures = persistence.list_recent(
        limit=100, cli_id="beta", status="failed"
    )
    assert len(beta_failures) == 2
    assert all(
        r["cli_id"] == "beta" and r["status"] == "failed" for r in beta_failures
    )

    # Unknown status returns empty.
    assert persistence.list_recent(limit=100, status="zeta") == []


def test_list_recent_filters_by_cli_id(tmp_path: Path) -> None:
    """list_recent(cli_id=X) pushes the filter to SQL so installations
    with thousands of persisted rows do not pay a fetch+filter cost in
    Python."""
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    # Pre-populate three CLIs with three rows each, distinct timestamps.
    for cli in ("alpha", "beta", "gamma"):
        for i in range(3):
            persistence.upsert_job(
                job_id=f"{cli}-{i:02d}",
                cli_id=cli,
                args={},
                cmd=cli,
                status="complete",
                started_at_iso=f"2026-01-01T00:{i:02d}:00.000000+00:00",
            )

    # Filter to "beta": exactly 3 rows, all cli_id == 'beta'.
    rows = persistence.list_recent(limit=100, cli_id="beta")
    assert len(rows) == 3
    assert all(r["cli_id"] == "beta" for r in rows)

    # No filter: all 9 rows surface.
    rows_all = persistence.list_recent(limit=100)
    assert len(rows_all) == 9

    # Filter for an unknown cli_id: empty list, not error.
    rows_none = persistence.list_recent(limit=100, cli_id="zeta")
    assert rows_none == []


def test_persistence_migrates_stderr_count_into_existing_db(tmp_path: Path) -> None:
    """Older installations have a jobs.sqlite that predates the
    stderr_count column. The forward-migration ALTER must run cleanly
    on first load and back-fill 0 for existing rows."""
    db = tmp_path / "jobs.sqlite"
    # Hand-craft a legacy DB without the stderr_count column.
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            """
            CREATE TABLE jobs (
              job_id TEXT PRIMARY KEY,
              cli_id TEXT NOT NULL,
              args_json TEXT NOT NULL,
              cmd TEXT,
              status TEXT NOT NULL,
              exit_code INTEGER,
              started_at_iso TEXT NOT NULL,
              ended_at_iso TEXT,
              duration_s REAL,
              stdout_tail TEXT NOT NULL DEFAULT '',
              stderr_tail TEXT NOT NULL DEFAULT ''
            );
            """
        )
        conn.execute(
            "INSERT INTO jobs (job_id, cli_id, args_json, cmd, status, "
            "started_at_iso) VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy-1", "x", "{}", "x", "complete", "2026-01-01T00:00:00.000000+00:00"),
        )

    # Loading the persistence should migrate the schema and the legacy
    # row should now have stderr_count = 0 by virtue of the DEFAULT.
    persistence = JobPersistence(db_path=db)
    row = persistence.get("legacy-1")
    assert row is not None
    assert row.get("stderr_count") == 0

    # And future writes should be able to set it.
    persistence.patch_status(
        job_id="legacy-1",
        status="complete",
        stderr_count=7,
    )
    row2 = persistence.get("legacy-1")
    assert row2 is not None
    assert row2.get("stderr_count") == 7


# ---------------------------------------------------------------------------
# vacuum: reclaims space after delete_old
# ---------------------------------------------------------------------------


def test_vacuum_shrinks_file_after_delete_old(tmp_path: Path) -> None:
    """``vacuum()`` rewrites the sqlite file as a tightly-packed copy
    so the on-disk size shrinks after a large ``delete_old`` purge.
    Without VACUUM the deleted pages are reused but the file never
    shrinks - this is the customer-visible reliability win for
    long-running dashboards."""
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)

    # Insert enough rows with non-trivial output tails to make the
    # file size meaningful (a near-empty sqlite file won't shrink
    # measurably even after vacuum because the page count is already
    # at the minimum).
    big_tail = "x" * 4_000  # 4KB stdout tail per row
    for i in range(200):
        job_id = f"job-{i:03d}"
        persistence.upsert_job(
            job_id=job_id,
            cli_id="x",
            args={},
            cmd="x",
            status="complete",
            started_at_iso=f"2026-01-01T{(i // 60):02d}:{(i % 60):02d}:00.000000+00:00",
        )
        persistence.set_output_tails(
            job_id=job_id, stdout_tail=big_tail, stderr_tail="",
        )

    # Measure total disk footprint across all sqlite files (.sqlite,
    # .sqlite-wal, .sqlite-shm). In WAL mode most writes land in the
    # -wal file, so the main file alone wouldn't show the savings.
    def total_disk_bytes() -> int:
        return sum(
            f.stat().st_size for f in db.parent.iterdir()
            if f.name.startswith("jobs.sqlite")
        )

    size_before = total_disk_bytes()
    deleted = persistence.delete_old(keep=10)
    assert deleted >= 100  # we wiped most of them

    # Total footprint may have grown slightly (DELETE writes tombstones
    # to WAL) - the point is vacuum reclaims it.
    size_after_delete = total_disk_bytes()

    # Vacuum should reclaim the freed pages AND checkpoint+truncate
    # the WAL.
    assert persistence.vacuum() is True
    size_after_vacuum = total_disk_bytes()
    assert size_after_vacuum < size_after_delete, (
        f"vacuum should shrink total disk usage: "
        f"before_delete={size_before}, after_delete={size_after_delete}, "
        f"after_vacuum={size_after_vacuum}"
    )

    # And the surviving 10 rows should still be queryable.
    rows = persistence.list_recent(limit=20)
    assert len(rows) == 10


def test_vacuum_returns_false_on_unwritable_path(tmp_path: Path) -> None:
    """``vacuum()`` must not raise if the DB file is missing or the
    underlying connection cannot open. The shutdown hook calls this
    best-effort and a misconfigured environment must not block exit."""
    # _readable() returns False when the file doesn't exist yet (no
    # writes have happened). Newly-constructed JobPersistence with no
    # ops issued falls into this branch.
    persistence = JobPersistence(db_path=tmp_path / "nonexistent.sqlite")
    # Without any writes the DB file isn't created; vacuum should
    # report false rather than create the file or raise.
    assert persistence.vacuum() is False


def test_last_vacuum_at_none_until_first_vacuum(tmp_path: Path) -> None:
    """Fresh JobPersistence reports `None` for `last_vacuum_at` until
    a VACUUM has actually succeeded. The /api/health endpoint relies
    on this default to render 'never (this session)' rather than a
    misleading zero or current-time value."""
    persistence = JobPersistence(db_path=tmp_path / "jobs.sqlite")
    assert persistence.last_vacuum_at() is None


def test_last_vacuum_at_set_on_successful_vacuum(tmp_path: Path) -> None:
    """A successful vacuum() advances the recorded timestamp to
    approximately wall-clock now."""
    import time as _t

    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    # Need at least one write so the file exists and vacuum can run.
    persistence.upsert_job(
        job_id="j1",
        cli_id="x",
        args={},
        cmd="x",
        status="complete",
        started_at_iso="2026-01-01T00:00:00.000000+00:00",
    )
    before = _t.time()
    assert persistence.vacuum() is True
    after = _t.time()
    ts = persistence.last_vacuum_at()
    assert ts is not None
    # Allow a small wall-clock window in case of CI clock skew.
    assert before - 1.0 <= ts <= after + 1.0


def test_last_vacuum_at_unchanged_on_failed_vacuum(tmp_path: Path) -> None:
    """A failed vacuum (file missing, never written) must NOT advance
    the timestamp - the metric should report the most recent
    *successful* compaction, not 'attempted'."""
    persistence = JobPersistence(db_path=tmp_path / "missing.sqlite")
    # Pre-populate with a known timestamp so we can verify it
    # survives a failed vacuum.
    persistence._last_vacuum_at = 1234.5
    # vacuum() returns False when the file doesn't exist (readable
    # check fails), so this exercises the failure path without
    # touching the filesystem.
    assert persistence.vacuum() is False
    assert persistence.last_vacuum_at() == 1234.5


def test_db_size_bytes_zero_before_first_write(tmp_path: Path) -> None:
    """Before any write, the db file does not exist on disk. Health
    snapshots taken in this state must report 0 rather than raising
    or creating the file as a side effect."""
    persistence = JobPersistence(db_path=tmp_path / "fresh.sqlite")
    assert persistence.db_size_bytes() == 0
    # The probe must NOT create the file; otherwise repeated probes
    # would scribble on disk for a healthcheck.
    assert not (tmp_path / "fresh.sqlite").exists()


def test_db_size_bytes_includes_wal_sidecar(tmp_path: Path) -> None:
    """In WAL mode, recent writes land in the ``-wal`` sidecar until
    a checkpoint runs. ``db_size_bytes()`` must sum the main file
    plus the -wal/-shm sidecars so the reported size matches the
    actual disk footprint a user pays for. Without this, the metric
    would under-count writes-since-last-checkpoint and SREs watching
    for bloat would miss real growth."""
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    # Trigger a write to materialize the file + WAL.
    persistence.upsert_job(
        job_id="j1",
        cli_id="run-eval",
        args={},
        cmd="x",
        status="complete",
        started_at_iso="2026-01-01T00:00:00.000000+00:00",
    )
    main_size = db.stat().st_size if db.exists() else 0
    reported = persistence.db_size_bytes()
    # Reported total must be >= the main file alone, since it adds
    # any sidecars on top.
    assert reported >= main_size
    assert reported > 0
    # If the WAL exists (it usually does after a write in WAL mode),
    # the reported total must strictly include it.
    wal = db.with_name(db.name + "-wal")
    if wal.exists():
        assert reported >= main_size + wal.stat().st_size


# ---------------------------------------------------------------------------
# Composite indexes for filtered list_recent queries
# ---------------------------------------------------------------------------


def test_init_creates_composite_indexes(tmp_path: Path) -> None:
    """Init creates composite indexes covering the common filter
    patterns (cli_id, started_at_iso) and (status, started_at_iso) so
    a ?cli_id=X or ?status=Y filter can seek directly rather than
    scan. Without these, large persistence tables would degrade
    under filter+sort queries."""
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    # Force schema init by writing a row.
    persistence.upsert_job(
        job_id="j1",
        cli_id="run-eval",
        args={},
        cmd="x",
        status="complete",
        started_at_iso="2026-05-07T00:00:00.000000+00:00",
    )

    with sqlite3.connect(str(db)) as conn:
        index_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
    assert "idx_jobs_cli_id_started_at" in index_names
    assert "idx_jobs_status_started_at" in index_names
    # The original single-column index is still present.
    assert "idx_jobs_started_at" in index_names


def test_filtered_query_uses_composite_index(tmp_path: Path) -> None:
    """EXPLAIN QUERY PLAN confirms a ?cli_id-filtered list_recent
    SELECT uses the composite index rather than a full scan. SQLite's
    plan output names the index it picked - asserting on substring
    match is robust across SQLite minor versions."""
    db = tmp_path / "jobs.sqlite"
    persistence = JobPersistence(db_path=db)
    persistence.upsert_job(
        job_id="j1",
        cli_id="run-eval",
        args={},
        cmd="x",
        status="complete",
        started_at_iso="2026-05-07T00:00:00.000000+00:00",
    )

    with sqlite3.connect(str(db)) as conn:
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM jobs "
            "WHERE cli_id=? "
            "ORDER BY started_at_iso DESC LIMIT 50",
            ("run-eval",),
        ).fetchall()
    plan_text = " ".join(str(row) for row in plan).lower()
    # Either the planner picks the cli_id composite index OR
    # (degenerately, on a single-row table) it does a search using
    # one of the indexes. Assert the composite is referenced.
    assert "idx_jobs_cli_id_started_at" in plan_text, (
        f"expected composite cli_id index in plan; got: {plan_text}"
    )


def test_legacy_db_without_composite_indexes_gets_them_on_init(
    tmp_path: Path,
) -> None:
    """A jobs.sqlite that pre-dates the composite-index commit should
    pick up the new indexes on next startup. CREATE INDEX IF NOT
    EXISTS is the mechanism; this test pins that the existing
    _ensure_schema flow runs the new statements (not just on
    fresh DBs)."""
    db = tmp_path / "jobs.sqlite"
    # Hand-craft a legacy DB with the table + only the original
    # single-column index. Mirrors the shape installations had
    # before composite indexes shipped.
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            """
            CREATE TABLE jobs (
              job_id TEXT PRIMARY KEY,
              cli_id TEXT NOT NULL,
              args_json TEXT NOT NULL,
              cmd TEXT,
              status TEXT NOT NULL,
              exit_code INTEGER,
              started_at_iso TEXT NOT NULL,
              ended_at_iso TEXT,
              duration_s REAL,
              stdout_tail TEXT NOT NULL DEFAULT '',
              stderr_tail TEXT NOT NULL DEFAULT '',
              stderr_count INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.execute(
            "CREATE INDEX idx_jobs_started_at ON jobs(started_at_iso DESC)"
        )

    # Opening with JobPersistence triggers _ensure_schema, which runs
    # CREATE INDEX IF NOT EXISTS for the composites.
    persistence = JobPersistence(db_path=db)
    # Force schema init via a write.
    persistence.upsert_job(
        job_id="legacy-2",
        cli_id="run-eval",
        args={},
        cmd="x",
        status="complete",
        started_at_iso="2026-05-07T00:00:00.000000+00:00",
    )

    with sqlite3.connect(str(db)) as conn:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
    # Both composite indexes should now exist (added by the ALTER-
    # equivalent forward migration, even though they're CREATE
    # INDEX IF NOT EXISTS rather than ALTER TABLE).
    assert "idx_jobs_cli_id_started_at" in names
    assert "idx_jobs_status_started_at" in names
    # And the original index is still there.
    assert "idx_jobs_started_at" in names
