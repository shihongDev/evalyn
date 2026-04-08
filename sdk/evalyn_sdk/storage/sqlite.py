from __future__ import annotations

import functools
import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, List, Optional

from ..models import (
    Annotation, EvalRun, FunctionCall, JudgeConfig, MetricSpec,
    SpanMetricLink, _parse_datetime, now_utc,
)
from .migrations import run_migrations

# Default paths for prod/test separation
DEFAULT_PROD_DB = "data/prod/traces.sqlite"
DEFAULT_TEST_DB = "data/test/traces.sqlite"
DB_PATHS = {"prod": DEFAULT_PROD_DB, "test": DEFAULT_TEST_DB}


@functools.lru_cache(maxsize=1)
def _find_project_root() -> Path:
    """Find project root by looking for .git (preferred) or pyproject.toml."""
    cwd = Path.cwd()
    # First pass: look for .git (most reliable indicator of repo root)
    for parent in [cwd, *cwd.parents]:
        if (parent / ".git").exists():
            return parent
    # Fallback: look for pyproject.toml if no .git found
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd  # fallback to cwd if no project markers found


def _get_default_db_path() -> str:
    """Get default DB path, respecting EVALYN_DB env var."""
    if env_path := os.getenv("EVALYN_DB"):
        return env_path
    return str(_find_project_root() / DEFAULT_PROD_DB)


def _dumps(data: object) -> str:
    return json.dumps(data, default=repr)


class SQLiteStorage:
    """Lightweight SQLite backend for local development."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path if path is not None else _get_default_db_path())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.conn = self._make_connection()
        self._init_tables()

    def _make_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with standard pragmas."""
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Get a thread-local connection. Creates one if needed.

        The main thread uses self.conn. Worker threads get their own
        connection via threading.local() to avoid SQLite mutex contention
        under WAL mode.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        # Main thread: use the primary connection
        if threading.current_thread() is threading.main_thread():
            return self.conn
        # Worker thread: create a new connection
        conn = self._make_connection()
        self._local.conn = conn
        return conn

    def _init_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS function_calls (
                id TEXT PRIMARY KEY,
                function_name TEXT,
                session_id TEXT,
                started_at TEXT,
                ended_at TEXT,
                duration_ms REAL,
                inputs TEXT,
                output TEXT,
                error TEXT,
                trace TEXT,
                metadata TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS otel_spans (
                trace_id TEXT,
                span_id TEXT PRIMARY KEY,
                parent_span_id TEXT,
                call_id TEXT,
                name TEXT,
                start_time TEXT,
                end_time TEXT,
                status TEXT,
                attributes TEXT,
                events TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_runs (
                id TEXT PRIMARY KEY,
                dataset_name TEXT,
                created_at TEXT,
                metric_results TEXT,
                metrics TEXT,
                judge_configs TEXT,
                summary TEXT,
                usage_summary TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id TEXT PRIMARY KEY,
                target_id TEXT,
                label TEXT,
                rationale TEXT,
                annotator TEXT,
                source TEXT,
                confidence REAL,
                created_at TEXT,
                metric_labels TEXT
            )
            """
        )
        # Relational metric results table (replaces JSON blob in eval_runs)
        #
        #   eval_runs 1──< metric_results_rows (one run has many results)
        #       id              run_id, item_id, metric_id, config_hash
        #
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metric_results_rows (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                call_id TEXT,
                metric_id TEXT NOT NULL,
                score REAL,
                passed INTEGER,
                details TEXT,
                config_hash TEXT,
                unit_id TEXT,
                unit_type TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                model TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS span_metric_links (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                metric_result_id TEXT NOT NULL,
                span_id TEXT NOT NULL,
                relevance REAL NOT NULL,
                reason TEXT DEFAULT '',
                UNIQUE(run_id, metric_result_id, span_id)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sml_run_metric
            ON span_metric_links(run_id, metric_result_id)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sml_run_span
            ON span_metric_links(run_id, span_id)
            """
        )
        # Performance indexes for common query patterns
        for idx_name, idx_def in [
            ("idx_fc_started_at", "function_calls(started_at)"),
            ("idx_er_created_at", "eval_runs(created_at)"),
            ("idx_er_dataset", "eval_runs(dataset_name)"),
            ("idx_ann_target", "annotations(target_id)"),
            ("idx_ann_created_at", "annotations(created_at)"),
            ("idx_mr_run", "metric_results_rows(run_id)"),
            ("idx_mr_run_item", "metric_results_rows(run_id, item_id)"),
            ("idx_mr_config", "metric_results_rows(run_id, metric_id, config_hash)"),
        ]:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_def}"
            )
        self.conn.commit()
        run_migrations(self.conn)

    def store_call(self, call: FunctionCall) -> None:
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO function_calls
            (id, function_name, session_id, started_at, ended_at, duration_ms, inputs, output, error, trace, metadata, parent_call_id, spans)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                call.id,
                call.function_name,
                call.session_id,
                call.started_at.isoformat(),
                call.ended_at.isoformat() if call.ended_at else None,
                call.duration_ms,
                _dumps(call.inputs),
                _dumps(call.output),
                call.error,
                _dumps([t.as_dict() for t in call.trace]) if call.trace else "[]",
                _dumps(call.metadata),
                call.parent_call_id,
                _dumps([s.as_dict() for s in call.spans]) if call.spans else None,
            ),
        )
        conn.commit()

    def get_call(self, call_id: str) -> Optional[FunctionCall]:
        cur = self.get_connection().cursor()
        cur.execute("SELECT * FROM function_calls WHERE id = ?", (call_id,))
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_call(row)

    def get_calls_batch(self, call_ids: List[str]) -> Dict[str, FunctionCall]:
        """Fetch multiple calls by ID in a single query. Returns {id: call}."""
        if not call_ids:
            return {}
        cur = self.get_connection().cursor()
        placeholders = ",".join("?" for _ in call_ids)
        cur.execute(
            f"SELECT * FROM function_calls WHERE id IN ({placeholders})",
            call_ids,
        )
        return {row["id"]: self._row_to_call(row) for row in cur.fetchall()}

    # Columns needed for lightweight listing (skip large blobs)
    _LIGHTWEIGHT_COLS = (
        "id, function_name, session_id, started_at, ended_at, "
        "duration_ms, error, metadata, parent_call_id"
    )

    def list_calls(
        self,
        limit: int = 100,
        project: Optional[str] = None,
        lightweight: bool = False,
        function_name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> List[FunctionCall]:
        cur = self.get_connection().cursor()
        cols = self._LIGHTWEIGHT_COLS if lightweight else "*"

        where_parts: list[str] = []
        params: list = []

        if project:
            where_parts.append(
                "(json_extract(metadata, '$.project_id') = ? "
                "OR json_extract(metadata, '$.project_name') = ?)"
            )
            params.extend([project, project])

        if function_name:
            where_parts.append("function_name LIKE ?")
            params.append(f"%{function_name}%")

        if version:
            where_parts.append("json_extract(metadata, '$.version') = ?")
            params.append(version)

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        params.append(limit)
        cur.execute(
            f"SELECT {cols} FROM function_calls {where_clause} ORDER BY started_at DESC LIMIT ?",
            params,
        )
        return [
            self._row_to_call(row, lightweight=lightweight)
            for row in cur.fetchall()
        ]

    def delete_calls(self, call_ids: List[str]) -> int:
        """Delete calls by IDs. Returns number deleted.

        Also deletes associated otel_spans entries.
        """
        if not call_ids:
            return 0
        conn = self.get_connection()
        cur = conn.cursor()
        placeholders = ",".join("?" * len(call_ids))
        # Delete related spans first
        cur.execute(
            f"DELETE FROM otel_spans WHERE call_id IN ({placeholders})", call_ids
        )
        # Delete calls
        cur.execute(
            f"DELETE FROM function_calls WHERE id IN ({placeholders})", call_ids
        )
        conn.commit()
        return cur.rowcount

    def store_eval_run(self, run: EvalRun) -> None:
        conn = self.get_connection()
        cur = conn.cursor()
        # Store run metadata (no JSON blob for metric_results on new runs)
        cur.execute(
            """
            INSERT OR REPLACE INTO eval_runs
            (id, dataset_name, created_at, metric_results, metrics, judge_configs, summary, usage_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                run.dataset_name,
                run.created_at.isoformat(),
                None,  # New runs skip the JSON blob
                _dumps([m.as_dict() for m in run.metrics]),
                _dumps([j.as_dict() for j in run.judge_configs]),
                _dumps(run.summary),
                _dumps(run.usage_summary),
            ),
        )
        # Store individual metric results in relational table
        if run.metric_results:
            self.batch_insert_metric_results(run.id, run.metric_results)
        else:
            conn.commit()

    def batch_insert_metric_results(
        self, run_id: str, results: List, batch_size: int = 1000
    ) -> None:
        """Insert metric results in batches for efficient writes."""
        conn = self.get_connection()
        cur = conn.cursor()
        for i in range(0, len(results), batch_size):
            batch = results[i : i + batch_size]
            cur.executemany(
                """
                INSERT OR REPLACE INTO metric_results_rows
                (id, run_id, item_id, call_id, metric_id, score, passed,
                 details, config_hash, unit_id, unit_type, input_tokens,
                 output_tokens, model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        hashlib.sha256(f"{run_id}:{r.item_id}:{r.metric_id}:{r.unit_id or ''}".encode()).hexdigest()[:32],
                        run_id,
                        r.item_id,
                        r.call_id,
                        r.metric_id,
                        r.score,
                        1 if r.passed else (0 if r.passed is not None else None),
                        _dumps(r.details) if r.details else None,
                        getattr(r, "config_hash", None),
                        r.unit_id,
                        r.unit_type,
                        r.input_tokens,
                        r.output_tokens,
                        r.model,
                    )
                    for r in batch
                ],
            )
        conn.commit()

    def load_metric_results(self, run_id: str) -> List:
        """Load metric results from relational table for a run."""
        from ..models import MetricResult

        cur = self.get_connection().cursor()
        cur.execute(
            "SELECT * FROM metric_results_rows WHERE run_id = ?", (run_id,)
        )
        return [self._row_to_metric_result(row) for row in cur.fetchall()]

    @staticmethod
    def _row_to_metric_result(row: sqlite3.Row):
        from ..models import MetricResult

        passed_val = row["passed"]
        return MetricResult(
            metric_id=row["metric_id"],
            item_id=row["item_id"],
            call_id=row["call_id"] or "",
            score=row["score"],
            passed=bool(passed_val) if passed_val is not None else None,
            details=({} if not row["details"] or row["details"] == "{}"
                    else json.loads(row["details"])),
            unit_id=row["unit_id"],
            unit_type=row["unit_type"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            model=row["model"],
        )

    def _lightweight_load_runs(self, rows: list) -> List[EvalRun]:
        """Convert rows to EvalRun objects WITHOUT loading metric results.

        Instead of deserializing all MetricResult rows (can be 7300+ per run),
        uses a single SQL COUNT to populate summary["results_count"].
        This makes list-runs ~100x faster for runs with many results.
        """
        if not rows:
            return []

        run_ids = [r["id"] for r in rows]

        # Single query to get result counts for all runs (vs N*M deserialization)
        cur = self.get_connection().cursor()
        counts: dict[str, int] = {}

        # Check relational table counts
        placeholders = ",".join("?" for _ in run_ids)
        cur.execute(
            f"SELECT run_id, COUNT(*) FROM metric_results_rows WHERE run_id IN ({placeholders}) GROUP BY run_id",
            run_ids,
        )
        for cid, cnt in cur.fetchall():
            counts[cid] = cnt

        # For legacy JSON blob runs, count by parsing the JSON length
        for row in rows:
            rid = row["id"]
            if rid not in counts and row["metric_results"]:
                blob = row["metric_results"]
                # Count JSON array elements without full deserialization
                counts[rid] = blob.count('"metric_id"')

        results = []
        for row in rows:
            run_id = row["id"]
            metrics_raw = json.loads(row["metrics"]) if row["metrics"] else []
            summary = json.loads(row["summary"]) if row["summary"] else {}
            summary["results_count"] = counts.get(run_id, 0)

            results.append(
                EvalRun(
                    id=run_id,
                    dataset_name=row["dataset_name"],
                    created_at=_parse_datetime(row["created_at"]) or now_utc(),
                    metric_results=[],  # Not loaded in lightweight mode
                    metrics=[MetricSpec.from_dict(m) for m in metrics_raw],
                    judge_configs=[],  # Not needed for listing
                    summary=summary,
                    usage_summary=json.loads(row["usage_summary"])
                    if row["usage_summary"]
                    else {},
                )
            )
        return results

    def _batch_load_runs(self, rows: list) -> List[EvalRun]:
        """Convert rows to EvalRun objects, batch-loading metric results when needed."""
        # Identify runs that need relational metric result loading
        needs_load = [r["id"] for r in rows if not r["metric_results"]]
        batch_results: dict = {}
        if needs_load:
            placeholders = ",".join("?" for _ in needs_load)
            cur = self.get_connection().cursor()
            cur.execute(
                f"SELECT * FROM metric_results_rows WHERE run_id IN ({placeholders})",
                needs_load,
            )
            for mr_row in cur.fetchall():
                rid = mr_row["run_id"]
                batch_results.setdefault(rid, []).append(mr_row)

        results = []
        for row in rows:
            run_id = row["id"]
            json_blob = row["metric_results"]

            # For runs stored in the relational table, construct MetricResult
            # objects directly from rows. Avoids a wasteful round-trip:
            # row -> MetricResult -> dict -> MetricResult (2x construction).
            if json_blob:
                # Legacy: metric results stored as JSON blob in eval_runs
                from ..models import MetricResult
                metric_results = [
                    MetricResult.from_dict(r) for r in json.loads(json_blob)
                ]
            else:
                mr_rows = batch_results.get(run_id, [])
                metric_results = [
                    self._row_to_metric_result(r) for r in mr_rows
                ]

            metrics_raw = json.loads(row["metrics"]) if row["metrics"] else []
            jc_raw = json.loads(row["judge_configs"]) if row["judge_configs"] else []

            results.append(
                EvalRun(
                    id=run_id,
                    dataset_name=row["dataset_name"],
                    created_at=_parse_datetime(row["created_at"]) or now_utc(),
                    metric_results=metric_results,
                    metrics=[MetricSpec.from_dict(m) for m in metrics_raw],
                    judge_configs=[JudgeConfig.from_dict(j) for j in jc_raw],
                    summary=json.loads(row["summary"]) if row["summary"] else {},
                    usage_summary=json.loads(row["usage_summary"])
                    if row["usage_summary"]
                    else {},
                )
            )
        return results

    def list_eval_runs(self, limit: int = 20, lightweight: bool = False) -> List[EvalRun]:
        cur = self.get_connection().cursor()
        cur.execute(
            """
            SELECT * FROM eval_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        if not lightweight:
            return self._batch_load_runs(rows)
        return self._lightweight_load_runs(rows)

    def list_eval_runs_by_project(
        self, dataset_name: str, limit: int = 20, lightweight: bool = False
    ) -> List[EvalRun]:
        """List eval runs for a specific project (dataset_name)."""
        cur = self.get_connection().cursor()
        cur.execute(
            """
            SELECT * FROM eval_runs
            WHERE dataset_name = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (dataset_name, limit),
        )
        rows = cur.fetchall()
        if not lightweight:
            return self._batch_load_runs(rows)
        return self._lightweight_load_runs(rows)

    def get_eval_run(self, run_id: str) -> Optional[EvalRun]:
        cur = self.get_connection().cursor()
        cur.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        if not row:
            return None
        return self._batch_load_runs([row])[0]

    def resolve_call_id(self, short_id: str) -> Optional[str]:
        """Resolve a short ID prefix to full call ID.

        Returns the full ID if exactly one match found, None otherwise.
        Supports both short prefixes (e.g., '6cf21eb3') and full UUIDs.
        """
        # First try exact match
        cur = self.get_connection().cursor()
        cur.execute("SELECT id FROM function_calls WHERE id = ?", (short_id,))
        row = cur.fetchone()
        if row:
            return row[0]

        # Try prefix match
        cur.execute(
            "SELECT id FROM function_calls WHERE id LIKE ? ORDER BY started_at DESC LIMIT 2",
            (short_id + "%",),
        )
        rows = cur.fetchall()
        if len(rows) == 1:
            return rows[0][0]
        return None  # Ambiguous or not found

    def count_call_prefix_matches(self, short_id: str) -> int:
        """Count how many call IDs match a given prefix."""
        cur = self.get_connection().cursor()
        cur.execute(
            "SELECT COUNT(*) FROM function_calls WHERE id LIKE ?",
            (short_id + "%",),
        )
        return cur.fetchone()[0]

    def resolve_eval_run_id(self, short_id: str) -> Optional[str]:
        """Resolve a short ID prefix to full eval run ID.

        Returns the full ID if exactly one match found, None otherwise.
        Supports both short prefixes and full UUIDs.
        """
        # First try exact match
        cur = self.get_connection().cursor()
        cur.execute("SELECT id FROM eval_runs WHERE id = ?", (short_id,))
        row = cur.fetchone()
        if row:
            return row[0]

        # Try prefix match
        cur.execute(
            "SELECT id FROM eval_runs WHERE id LIKE ? ORDER BY created_at DESC LIMIT 2",
            (short_id + "%",),
        )
        rows = cur.fetchall()
        if len(rows) == 1:
            return rows[0][0]
        return None  # Ambiguous or not found

    def store_annotations(self, annotations: Iterable[Annotation]) -> None:
        conn = self.get_connection()
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO annotations
            (id, target_id, label, rationale, annotator, source, confidence, created_at, metric_labels)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    ann.id,
                    ann.target_id,
                    _dumps(ann.label),
                    ann.rationale,
                    ann.annotator,
                    ann.source,
                    ann.confidence,
                    ann.created_at.isoformat(),
                    _dumps({k: v.as_dict() for k, v in ann.metric_labels.items()})
                    if ann.metric_labels
                    else None,
                )
                for ann in annotations
            ),
        )
        conn.commit()

    def list_annotations(
        self, target_id: Optional[str] = None, limit: int = 100
    ) -> List[Annotation]:
        cur = self.get_connection().cursor()
        if target_id:
            cur.execute(
                """
                SELECT * FROM annotations
                WHERE target_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (target_id, limit),
            )
        else:
            cur.execute(
                """
                SELECT * FROM annotations
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        return [
            Annotation.from_dict(
                {
                    "id": row["id"],
                    "target_id": row["target_id"],
                    "label": json.loads(row["label"]) if row["label"] else None,
                    "rationale": row["rationale"],
                    "annotator": row["annotator"],
                    "source": row["source"],
                    "confidence": row["confidence"],
                    "created_at": row["created_at"],
                    "metric_labels": json.loads(row["metric_labels"])
                    if row["metric_labels"]
                    else {},
                }
            )
            for row in cur.fetchall()
        ]

    def store_span_metric_links(self, links: Iterable[SpanMetricLink]) -> None:
        conn = self.get_connection()
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO span_metric_links
            (id, run_id, metric_result_id, span_id, relevance, reason)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (l.id, l.run_id, l.metric_result_id, l.span_id, l.relevance, l.reason)
                for l in links
            ),
        )
        conn.commit()

    def list_span_metric_links(
        self,
        run_id: str,
        span_id: Optional[str] = None,
        metric_result_id: Optional[str] = None,
    ) -> List[SpanMetricLink]:
        cur = self.get_connection().cursor()
        query = "SELECT * FROM span_metric_links WHERE run_id = ?"
        params: list = [run_id]
        if span_id:
            query += " AND span_id = ?"
            params.append(span_id)
        if metric_result_id:
            query += " AND metric_result_id = ?"
            params.append(metric_result_id)
        cur.execute(query, params)
        rows = cur.fetchall()
        return [
            SpanMetricLink(
                id=r["id"],
                run_id=r["run_id"],
                metric_result_id=r["metric_result_id"],
                span_id=r["span_id"],
                relevance=r["relevance"],
                reason=r["reason"],
            )
            for r in rows
        ]

    def list_spans(self, call_id: str) -> List[dict]:
        cur = self.get_connection().cursor()
        try:
            cur.execute(
                """
                SELECT trace_id FROM otel_spans
                WHERE call_id = ?
                ORDER BY start_time ASC
                LIMIT 1
                """,
                (call_id,),
            )
        except sqlite3.OperationalError:
            return []
        row = cur.fetchone()
        if not row or not row["trace_id"]:
            return []
        trace_id = row["trace_id"]
        cur.execute(
            """
            SELECT * FROM otel_spans
            WHERE trace_id = ?
            ORDER BY start_time ASC
            """,
            (trace_id,),
        )
        return [
            {
                "trace_id": r["trace_id"],
                "span_id": r["span_id"],
                "parent_span_id": r["parent_span_id"],
                "call_id": r["call_id"],
                "name": r["name"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "status": r["status"],
                "attributes": json.loads(r["attributes"])
                if r["attributes"]
                else {},
                "events": json.loads(r["events"]) if r["events"] else [],
            }
            for r in cur.fetchall()
        ]

    def close(self) -> None:
        self.conn.close()

    def _row_to_call(
        self, row: sqlite3.Row, lightweight: bool = False
    ) -> FunctionCall:
        # Handle new columns that may not exist in old databases
        parent_call_id = None
        try:
            parent_call_id = row["parent_call_id"]
        except (KeyError, IndexError):
            pass

        if lightweight:
            # Skip deserializing inputs, output, trace, spans -
            # they are not in the SELECT and not needed for listing.
            return FunctionCall.from_dict(
                {
                    "id": row["id"],
                    "function_name": row["function_name"],
                    "session_id": row["session_id"],
                    "started_at": row["started_at"],
                    "ended_at": row["ended_at"],
                    "duration_ms": row["duration_ms"],
                    "inputs": {},
                    "output": None,
                    "error": row["error"],
                    "trace": [],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "parent_call_id": parent_call_id,
                    "spans": [],
                }
            )

        spans: list = []
        try:
            spans_raw = row["spans"]
            if spans_raw:
                spans = json.loads(spans_raw)
        except (KeyError, IndexError):
            pass

        return FunctionCall.from_dict(
            {
                "id": row["id"],
                "function_name": row["function_name"],
                "session_id": row["session_id"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "duration_ms": row["duration_ms"],
                "inputs": json.loads(row["inputs"]) if row["inputs"] else {},
                "output": json.loads(row["output"]) if row["output"] else None,
                "error": row["error"],
                "trace": json.loads(row["trace"]) if row["trace"] else [],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "parent_call_id": parent_call_id,
                "spans": spans,
            }
        )

