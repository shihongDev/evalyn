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
                    return
                conn.execute(
                    """
                    UPDATE jobs
                       SET status=?, exit_code=?, ended_at_iso=?, duration_s=?, stderr_count=?
                     WHERE job_id=?
                    """,
                    (status, exit_code, ended_at_iso, duration_s, stderr_count, job_id),
                )
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
        status: str | None = None,
    ) -> list[dict]:
        """Return up to ``limit`` rows in reverse-chronological order.

        When ``cli_id`` and/or ``status`` are set, the filters are
        pushed down to SQL ``WHERE`` clauses so we never project the
        full set into Python just to drop most of it. Saves work on
        installations with a large persisted history.
        """
        if not self._readable():
            return []
        # Build the WHERE clause incrementally so any combination of
        # filters is supported (cli_id only, status only, both, neither).
        where_parts: list[str] = []
        params: list[object] = []
        if cli_id is not None:
            where_parts.append("cli_id=?")
            params.append(cli_id)
        if status is not None:
            where_parts.append("status=?")
            params.append(status)
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
                return cur.rowcount or 0
        except (OSError, sqlite3.Error) as exc:
            logger.warning("JobPersistence delete_old failed: %s", exc)
            return 0


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
