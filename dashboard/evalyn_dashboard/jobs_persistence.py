"""SQLite-backed mirror for the JobManager so jobs survive restarts.

The in-memory :class:`JobManager` is the source of truth while a job is
alive. This module provides a small best-effort persistence layer that
mirrors job rows to a sqlite file at ``.evalyn/data/jobs.sqlite`` so the
dashboard's Recent Jobs drawer can reconstruct state after the server
restarts (or after the in-memory LRU evicts a row).

Design notes:
- Pure stdlib (``sqlite3``); no new dependencies.
- One fresh ``sqlite3.connect()`` per call (cheap locally; ~0.1ms).
  WAL mode is set once on first init for safe concurrent reads.
- Every IO is wrapped in try/except OSError + sqlite3.Error and logged
  at WARN; persistence failures must NEVER crash a job.
- Output streams are capped at ``MAX_PERSISTED_OUTPUT`` chars per stream
  (32 KB total per job) to keep the DB small. The append helper rolls
  the tail in-place when the cap is exceeded.

Schema (one table)::

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

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_PERSISTED_OUTPUT = 16384  # chars per stream
DEFAULT_DB_PATH = Path(".evalyn") / "data" / "jobs.sqlite"

# Override the default DB path for tests / multi-worker isolation.
_DB_PATH_ENV_VAR = "EVALYN_DASHBOARD_JOBS_DB"


def _resolve_default_path() -> Path:
    override = os.environ.get(_DB_PATH_ENV_VAR)
    if override:
        return Path(override)
    return DEFAULT_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
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

_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_jobs_started_at "
    "ON jobs(started_at_iso DESC);"
)

# Composite indexes for the most common filtered+ordered queries on
# list_recent. Without these, a "?cli_id=run-eval" filter on a table
# with thousands of rows scans the table in started_at_iso order and
# discards non-matches. The composite indexes let SQLite seek directly
# to the cli_id (or status) prefix, then walk in started_at_iso order.
#
# These are CREATE INDEX IF NOT EXISTS so re-running on an existing
# DB is a no-op. The single-column idx_jobs_started_at above stays
# useful for unfiltered list_recent calls and for ?since/?before-only
# windowed queries.
_COMPOSITE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_jobs_cli_id_started_at "
    "ON jobs(cli_id, started_at_iso DESC);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_status_started_at "
    "ON jobs(status, started_at_iso DESC);",
)

# Forward migrations for installations whose jobs.sqlite predates a
# new column. Each entry is the ALTER statement; we catch sqlite3
# OperationalError "duplicate column name" so re-running on an
# already-migrated DB is a no-op. Order matters - newest at the bottom.
_MIGRATIONS = [
    "ALTER TABLE jobs ADD COLUMN stderr_count INTEGER NOT NULL DEFAULT 0;",
]


class JobPersistence:
    """Tiny sqlite mirror for job rows. Best-effort; never raises."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else _resolve_default_path()
        # Disabled flips True only after a write fails irrecoverably so reads
        # don't keep retrying. Init is deferred to first write so an unused
        # JobPersistence (e.g. dashboard idle, no jobs spawned) never touches
        # the filesystem and tests with monkeypatched cwd stay clean.
        self._disabled = False
        self._schema_ready = False
        # Unix epoch float of the last successful VACUUM, or None when
        # never vacuumed in this process. Operators want to know how
        # long ago the file was compacted - if hours have passed and
        # the DB has grown, a manual restart-vacuum may be worth doing.
        # Process-local: a fresh process starts at None even if the
        # previous run vacuumed at shutdown. That's the right default
        # since the metric communicates "in this dashboard session".
        self._last_vacuum_at: float | None = None
        # Short-TTL cache for stats(). Keyed by the
        # `recent_failure_window_seconds` argument since different
        # callers can pass different windows. Each entry is
        # (cached_at_ts, value).
        #
        # Why cache: /api/health is polled every 15s by every
        # dashboard tab. With multiple tabs open the cumulative SQL
        # cost of repeating 4 aggregates per poll is real. A 5s TTL
        # collapses concurrent multi-tab polls to ~1 query/poll/window
        # without forcing fresh-data callers to wait too long.
        #
        # Why 5s: smaller than the 15s health-poll cadence so
        # single-tab pollers always see fresh data; large enough to
        # collapse the bursts when many tabs are open. Anything
        # tighter would defeat the cache; anything looser would risk
        # showing visibly-stale "Persisted jobs" counts after a
        # bulk purge.
        #
        # Invalidation strategy: row-count-changing writes
        # (upsert_job, patch_status, delete, delete_old, vacuum) call
        # _invalidate_stats_cache() so the next stats() reads fresh.
        # append_output / set_output_tails do NOT invalidate even
        # though they change SUM(stderr_count) - those fire many
        # times per second on a hot run, and clearing the cache that
        # often defeats the whole point. The accepted staleness for
        # stderr_count is bounded by the 5s TTL.
        self._stats_cache: dict[int, tuple[float, dict]] = {}

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> bool:
        """Create parent dir, run schema, set WAL mode. Lazy + idempotent."""
        if self._schema_ready:
            return True
        if self._disabled:
            return False
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute(_SCHEMA)
                conn.execute(_INDEX)
                # Composite indexes for filtered list_recent queries.
                # IF NOT EXISTS makes them no-ops on already-migrated
                # DBs. Wrap each in try/except so an unrelated failure
                # on one (e.g. read-only filesystem mid-creation) does
                # not abort the rest of init.
                for idx_stmt in _COMPOSITE_INDEXES:
                    try:
                        conn.execute(idx_stmt)
                    except sqlite3.OperationalError as exc:
                        logger.warning(
                            "JobPersistence index creation failed: %s", exc
                        )
                # Forward migrations for older installations. Each ALTER
                # is wrapped in its own try/except so a "duplicate column
                # name" error (already migrated) is silently ignored, and
                # an unrelated failure on one migration does not abort
                # the others. New columns get NOT NULL DEFAULT so existing
                # rows back-fill automatically.
                for stmt in _MIGRATIONS:
                    try:
                        conn.execute(stmt)
                    except sqlite3.OperationalError as exc:
                        msg = str(exc).lower()
                        if "duplicate column name" not in msg:
                            logger.warning(
                                "JobPersistence migration skipped (%s): %s", exc, stmt
                            )
                # WAL gives better concurrent read behaviour. Set once.
                try:
                    conn.execute("PRAGMA journal_mode=WAL;")
                except sqlite3.Error:
                    # WAL is a nice-to-have; default rollback journal is fine.
                    pass
            self._schema_ready = True
            return True
        except (OSError, sqlite3.Error) as exc:
            logger.warning("JobPersistence init failed (%s); disabling persistence", exc)
            self._disabled = True
            return False

    def _connect(self) -> sqlite3.Connection:
        """Open a fresh autocommit connection."""
        conn = sqlite3.connect(str(self.db_path), isolation_level=None, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _readable(self) -> bool:
        """Return True iff the DB file already exists (read-only operations).

        When the file exists but our schema is not yet marked ready, we
        still trigger ``_ensure_schema()`` so any forward migrations run
        BEFORE the first read. Without this, an older jobs.sqlite would
        be read with the old schema (missing newer columns) until a
        write path forced the migration.
        """
        if self._disabled:
            return False
        if self._schema_ready:
            return True
        try:
            if not self.db_path.exists():
                return False
        except OSError:
            return False
        # File exists but not yet migrated. Run schema + migrations now
        # so the read sees the up-to-date column set. _ensure_schema is
        # idempotent and self-disables on hard failure.
        return self._ensure_schema()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert_job(
        self,
        job_id: str,
        cli_id: str,
        args: dict,
        cmd: str,
        status: str,
        started_at_iso: str,
    ) -> None:
        """Insert or update a job row at spawn time."""
        if not self._ensure_schema():
            return
        try:
            payload = json.dumps(args, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning("JobPersistence: failed to serialize args for %s: %s", job_id, exc)
            payload = "{}"
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs
                      (job_id, cli_id, args_json, cmd, status, started_at_iso)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                      cli_id=excluded.cli_id,
                      args_json=excluded.args_json,
                      cmd=excluded.cmd,
                      status=excluded.status,
                      started_at_iso=excluded.started_at_iso
                    """,
                    (job_id, cli_id, payload, cmd, status, started_at_iso),
                )
            self._invalidate_stats_cache()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("JobPersistence upsert_job(%s) failed: %s", job_id, exc)

    def patch_status(
        self,
        job_id: str,
        status: str,
        exit_code: int | None = None,
        ended_at_iso: str | None = None,
        duration_s: float | None = None,
        stderr_count: int | None = None,
    ) -> None:
        """Patch terminal-state fields for an existing row.

        ``stderr_count`` is optional: when provided, it overwrites the
        column. Pass ``None`` to leave the existing value untouched
        (e.g. mid-run patches that don't yet know the final count).
        """
        if not self._ensure_schema():
            return
        try:
            with self._connect() as conn:
                if stderr_count is None:
                    conn.execute(
                        """
                        UPDATE jobs
                           SET status=?, exit_code=?, ended_at_iso=?, duration_s=?
                         WHERE job_id=?
                        """,
                        (status, exit_code, ended_at_iso, duration_s, job_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE jobs
                           SET status=?, exit_code=?, ended_at_iso=?, duration_s=?, stderr_count=?
                         WHERE job_id=?
                        """,
                        (status, exit_code, ended_at_iso, duration_s, stderr_count, job_id),
                    )
            # Invalidate after either branch since both change `status`
            # (which the by_status group-by reads). The early-return
            # in the previous shape skipped this; reorganized into
            # if/else so the invalidation lands once for both paths.
            self._invalidate_stats_cache()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("JobPersistence patch_status(%s) failed: %s", job_id, exc)

    def append_output(self, job_id: str, kind: str, line: str) -> None:
        """Append a line to stdout_tail or stderr_tail, rolling at the cap.

        Used for per-line persistence. Most callers prefer
        :meth:`set_output_tails` (terminal-only) for fewer writes.
        """
        if not self._ensure_schema():
            return
        column = "stdout_tail" if kind == "stdout" else "stderr_tail"
        try:
            with self._connect() as conn:
                row = conn.execute(
                    f"SELECT {column} FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                if row is None:
                    return
                current = row[0] or ""
                addition = (line + "\n") if not line.endswith("\n") else line
                combined = current + addition
                if len(combined) > MAX_PERSISTED_OUTPUT:
                    combined = combined[-MAX_PERSISTED_OUTPUT:]
                conn.execute(
                    f"UPDATE jobs SET {column}=? WHERE job_id=?",
                    (combined, job_id),
                )
        except (OSError, sqlite3.Error) as exc:
            logger.warning("JobPersistence append_output(%s) failed: %s", job_id, exc)

    def set_output_tails(
        self, job_id: str, stdout_tail: str, stderr_tail: str
    ) -> None:
        """Set the full output tails in one write (used at terminal status).

        Each stream is truncated to ``MAX_PERSISTED_OUTPUT`` chars (last N).
        """
        if not self._ensure_schema():
            return
        if len(stdout_tail) > MAX_PERSISTED_OUTPUT:
            stdout_tail = stdout_tail[-MAX_PERSISTED_OUTPUT:]
        if len(stderr_tail) > MAX_PERSISTED_OUTPUT:
            stderr_tail = stderr_tail[-MAX_PERSISTED_OUTPUT:]
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET stdout_tail=?, stderr_tail=? WHERE job_id=?",
                    (stdout_tail, stderr_tail, job_id),
                )
        except (OSError, sqlite3.Error) as exc:
            logger.warning("JobPersistence set_output_tails(%s) failed: %s", job_id, exc)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, job_id: str) -> dict | None:
        """Return one job row as a dict, or None if missing."""
        if not self._readable():
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("JobPersistence get(%s) failed: %s", job_id, exc)
            return None
        if row is None:
            return None
        return _row_to_dict(row)

    def list_recent(
        self,
        limit: int = 30,
        cli_id: str | None = None,
        cli_ids: list[str] | None = None,
        status: str | None = None,
        since_iso: str | None = None,
        before_iso: str | None = None,
    ) -> list[dict]:
        """Return up to ``limit`` rows in reverse-chronological order.

        Filters (all pushed down to SQL ``WHERE`` clauses, AND-combined):

        - ``cli_id`` matches the catalog id passed at spawn time.
        - ``cli_ids`` matches any of the listed catalog ids (SQL IN
          clause). Mutually exclusive with ``cli_id`` in practice;
          when both are set the more-general ``cli_ids`` wins because
          a single id is just a one-element list of itself.
        - ``status`` matches one of the JobState values.
        - ``since_iso`` keeps only rows whose ``started_at_iso > since_iso``.
          Pass an ISO-8601 string; callers polling for "what's new"
          should hand the previous response's max ``started_at`` back.
        - ``before_iso`` keeps only rows whose ``started_at_iso < before_iso``.
          Combine with ``since_iso`` to scope a windowed query
          ("jobs between yesterday 9am and yesterday 5pm"). Strict
          less-than so a paginating caller can pass the oldest row's
          ``started_at`` from the previous page without re-receiving it.
        """
        if not self._readable():
            return []
        # Build the WHERE clause incrementally so any combination of
        # filters is supported.
        where_parts: list[str] = []
        params: list[object] = []
        if cli_ids is not None and len(cli_ids) > 0:
            placeholders = ", ".join("?" for _ in cli_ids)
            where_parts.append(f"cli_id IN ({placeholders})")
            params.extend(cli_ids)
        elif cli_id is not None:
            where_parts.append("cli_id=?")
            params.append(cli_id)
        if status is not None:
            where_parts.append("status=?")
            params.append(status)
        if since_iso is not None:
            where_parts.append("started_at_iso>?")
            params.append(since_iso)
        if before_iso is not None:
            where_parts.append("started_at_iso<?")
            params.append(before_iso)
        sql = "SELECT * FROM jobs"
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
        sql += " ORDER BY started_at_iso DESC LIMIT ?"
        params.append(int(limit))
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, tuple(params)).fetchall()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("JobPersistence list_recent failed: %s", exc)
            return []
        return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # GC
    # ------------------------------------------------------------------

    _STATS_CACHE_TTL_S = 5.0

    def _invalidate_stats_cache(self) -> None:
        """Drop all cached stats() entries. Called from row-count-
        changing writes so the next stats() reads fresh from disk."""
        self._stats_cache.clear()

    def stats(self, recent_failure_window_seconds: int = 86400) -> dict:
        """Aggregate counts over the persisted jobs table.

        Returns a dict shaped::

            {
              total: int,                 # all rows
              by_status: dict[str, int],  # COUNT(*) GROUP BY status
              total_stderr: int,          # SUM(stderr_count)
              recent_failures: int,       # failed jobs in the last
                                          # ``recent_failure_window_seconds``
            }

        Empty dict shape on read failure (logged at WARN). Each query
        is a single aggregate so the underlying cost is cheap even
        with many rows; cached for ``_STATS_CACHE_TTL_S`` seconds to
        collapse multi-tab health-polling bursts (see __init__
        commentary on the cache).
        """
        # Cache hit path. Returning a shared dict reference is safe
        # because callers MUST treat the result as read-only (the
        # docstring shape commits to dict-of-immutables; mutating
        # would be a contract violation regardless of caching).
        cached = self._stats_cache.get(recent_failure_window_seconds)
        if cached is not None:
            cached_at, value = cached
            if time.time() - cached_at < self._STATS_CACHE_TTL_S:
                return value

        empty = {
            "total": 0,
            "by_status": {},
            "total_stderr": 0,
            "recent_failures": 0,
        }
        if not self._readable():
            return empty
        try:
            from datetime import datetime, timedelta, timezone

            cutoff = datetime.now(timezone.utc) - timedelta(
                seconds=recent_failure_window_seconds
            )
            cutoff_iso = cutoff.isoformat()
            with self._connect() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM jobs"
                ).fetchone()[0] or 0
                by_status: dict[str, int] = {}
                for row in conn.execute(
                    "SELECT status, COUNT(*) FROM jobs GROUP BY status"
                ).fetchall():
                    by_status[row[0]] = row[1]
                total_stderr = conn.execute(
                    "SELECT COALESCE(SUM(stderr_count), 0) FROM jobs"
                ).fetchone()[0] or 0
                recent_failures = conn.execute(
                    "SELECT COUNT(*) FROM jobs "
                    "WHERE status=? AND started_at_iso > ?",
                    ("failed", cutoff_iso),
                ).fetchone()[0] or 0
            result = {
                "total": int(total),
                "by_status": by_status,
                "total_stderr": int(total_stderr),
                "recent_failures": int(recent_failures),
            }
            self._stats_cache[recent_failure_window_seconds] = (
                time.time(),
                result,
            )
            return result
        except (OSError, sqlite3.Error) as exc:
            logger.warning("JobPersistence stats failed: %s", exc)
            return empty

    def delete(self, job_id: str) -> bool:
        """Delete a single row by job_id. Returns True if a row was
        removed, False otherwise (unknown id, or persistence
        unavailable). Idempotent."""
        if not self._readable():
            return False
        try:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
                removed = (cur.rowcount or 0) > 0
            if removed:
                self._invalidate_stats_cache()
            return removed
        except (OSError, sqlite3.Error) as exc:
            logger.warning("JobPersistence delete(%s) failed: %s", job_id, exc)
            return False

    def delete_old(self, keep: int = 100) -> int:
        """Keep the ``keep`` most recent rows; delete the rest. Returns count deleted."""
        if not self._readable():
            return 0
        try:
            with self._connect() as conn:
                # Compute the cutoff started_at_iso. Anything strictly older
                # than the keep-th newest row is deleted.
                cutoff_row = conn.execute(
                    "SELECT started_at_iso FROM jobs "
                    "ORDER BY started_at_iso DESC LIMIT 1 OFFSET ?",
                    (int(keep),),
                ).fetchone()
                if cutoff_row is None:
                    return 0
                cutoff = cutoff_row[0]
                cur = conn.execute(
                    "DELETE FROM jobs WHERE started_at_iso <= ?",
                    (cutoff,),
                )
                deleted = cur.rowcount or 0
            if deleted:
                self._invalidate_stats_cache()
            return deleted
        except (OSError, sqlite3.Error) as exc:
            logger.warning("JobPersistence delete_old failed: %s", exc)
            return 0

    def db_size_bytes(self) -> int:
        """Disk footprint of the persistence db, in bytes.

        Sums the main file plus the ``-wal`` and ``-shm`` sidecars
        when present. Important: in WAL mode, deleted rows show up
        as growth in the WAL file rather than the main file until
        a checkpoint runs, so summing both gives the actual on-disk
        cost a user pays.

        Useful for ``/api/health`` snapshots: an SRE watching the
        dashboard long-term can spot bloat trending toward a vacuum
        before it becomes an incident. Returns 0 when the file
        doesn't exist (e.g. fresh install pre-first-write) or stat
        fails (filesystem permissions surprise) - caller should
        treat 0 as "no signal" rather than "file is empty".
        """
        try:
            total = 0
            if self.db_path.exists():
                total += self.db_path.stat().st_size
            for ext in ("-wal", "-shm"):
                sidecar = self.db_path.with_name(self.db_path.name + ext)
                if sidecar.exists():
                    total += sidecar.stat().st_size
            return total
        except OSError:
            return 0

    def vacuum(self) -> bool:
        """Reclaim disk space from previously-deleted rows.

        SQLite's ``DELETE`` only marks pages as free for reuse; the
        file itself never shrinks. After hours/days of running, the
        ``.evalyn/data/jobs.sqlite`` file accumulates fragmentation
        and grows beyond the working-set size implied by ``keep``.
        Periodic ``VACUUM`` rewrites the file as a tightly-packed
        copy, plus runs an implicit WAL checkpoint as a side effect.

        Best-effort: any I/O or sqlite error is logged at WARN and
        swallowed - callers (typically the FastAPI shutdown hook) MUST
        NOT block process exit on this. Returns True on success.

        VACUUM is heavy (rewrites the whole file) so this is intended
        for shutdown only, not for the steady-state delete_old loop.
        """
        if not self._readable():
            return False
        try:
            with self._connect() as conn:
                # In WAL mode, writes accumulate in jobs.sqlite-wal until
                # a checkpoint folds them back into the main DB file. A
                # plain VACUUM without this would rewrite a stale main
                # file and the freed pages would not actually be reclaimed
                # on disk - we'd end up with a vacuum_ok=True log line and
                # zero size delta, which is the worst kind of "succeed
                # silently" footgun. TRUNCATE checkpoints all uncommitted
                # WAL pages back to the main file AND truncates the WAL
                # file itself, so the subsequent VACUUM operates on the
                # complete dataset.
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("VACUUM")
            # Record the timestamp ONLY on success so the
            # last_vacuum_at metric reports the most recent
            # compaction that actually happened. Failed vacuums
            # leave the previous timestamp in place (or None if
            # there's never been one).
            self._last_vacuum_at = time.time()
            # VACUUM doesn't change row counts, but it DOES change
            # everything stat()-able about the file (size, mtime).
            # Invalidating ensures the post-vacuum stats() call
            # doesn't return a value held over from before the
            # compaction that's misleading in any way.
            self._invalidate_stats_cache()
            return True
        except (OSError, sqlite3.Error) as exc:
            logger.warning("JobPersistence vacuum failed: %s", exc)
            return False

    def last_vacuum_at(self) -> float | None:
        """Unix epoch float of the last successful VACUUM in this process,
        or None when no vacuum has run yet. Process-local by design
        (see __init__ commentary)."""
        return self._last_vacuum_at


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Project a sqlite row to a plain dict (decode args_json)."""
    d = dict(row)
    raw_args = d.pop("args_json", "{}")
    try:
        d["args"] = json.loads(raw_args) if raw_args else {}
    except (TypeError, ValueError):
        d["args"] = {}
    return d


__all__ = [
    "JobPersistence",
    "MAX_PERSISTED_OUTPUT",
    "DEFAULT_DB_PATH",
]
