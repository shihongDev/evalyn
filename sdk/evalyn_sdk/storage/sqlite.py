from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, List, Optional

from ..models import Annotation, EvalRun, FunctionCall, SpanMetricLink
from .migrations import run_migrations

# Default paths for prod/test separation
DEFAULT_PROD_DB = "data/prod/traces.sqlite"
DEFAULT_TEST_DB = "data/test/traces.sqlite"


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
    return json.dumps(data, default=lambda o: repr(o))


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
        conn.execute("PRAGMA busy_timeout=5000")
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
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fc_started_at "
            "ON function_calls(started_at)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_er_created_at "
            "ON eval_runs(created_at)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_er_dataset "
            "ON eval_runs(dataset_name)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ann_target "
            "ON annotations(target_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ann_created_at "
            "ON annotations(created_at)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_mr_run "
            "ON metric_results_rows(run_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_mr_run_item "
            "ON metric_results_rows(run_id, item_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_mr_config "
            "ON metric_results_rows(run_id, metric_id, config_hash)"
        )
        self.conn.commit()
        run_migrations(self.conn)

    def store_call(self, call: FunctionCall) -> None:
        cur = self.conn.cursor()
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
                _dumps([t.as_dict() for t in call.trace]),
                _dumps(call.metadata),
                call.parent_call_id,
                _dumps([s.as_dict() for s in call.spans]) if call.spans else None,
            ),
        )
        self.conn.commit()

    def get_call(self, call_id: str) -> Optional[FunctionCall]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM function_calls WHERE id = ?", (call_id,))
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_call(row)

    def list_calls(
        self, limit: int = 100, project: Optional[str] = None
    ) -> List[FunctionCall]:
        cur = self.conn.cursor()
        if project:
            # Filter by project in SQL using JSON metadata fields
            cur.execute(
                """
                SELECT * FROM function_calls
                WHERE (
                    json_extract(metadata, '$.project_id') = ?
                    OR json_extract(metadata, '$.project_name') = ?
                )
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (project, project, limit),
            )
        else:
            cur.execute(
                """
                SELECT * FROM function_calls
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        rows = cur.fetchall()
        calls: List[FunctionCall] = []
        for row in rows:
            calls.append(self._row_to_call(row))
        return calls

    def delete_calls(self, call_ids: List[str]) -> int:
        """Delete calls by IDs. Returns number deleted.

        Also deletes associated otel_spans entries.
        """
        if not call_ids:
            return 0
        cur = self.conn.cursor()
        placeholders = ",".join("?" * len(call_ids))
        # Delete related spans first
        cur.execute(
            f"DELETE FROM otel_spans WHERE call_id IN ({placeholders})", call_ids
        )
        # Delete calls
        cur.execute(
            f"DELETE FROM function_calls WHERE id IN ({placeholders})", call_ids
        )
        self.conn.commit()
        return cur.rowcount

    def store_eval_run(self, run: EvalRun) -> None:
        cur = self.conn.cursor()
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
            self.conn.commit()

    def batch_insert_metric_results(
        self, run_id: str, results: List, batch_size: int = 1000
    ) -> None:
        """Insert metric results in batches for efficient writes."""
        from ..models import MetricResult

        cur = self.conn.cursor()
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
                        f"{run_id}-{r.item_id}-{r.metric_id}"[:64],
                        run_id,
                        r.item_id,
                        r.call_id,
                        r.metric_id,
                        r.score,
                        1 if r.passed else (0 if r.passed is not None else None),
                        _dumps(r.details) if r.details else None,
                        getattr(r, "config_hash", None),
                        getattr(r, "unit_id", None),
                        getattr(r, "unit_type", None),
                        getattr(r, "input_tokens", None),
                        getattr(r, "output_tokens", None),
                        getattr(r, "model", None),
                    )
                    for r in batch
                ],
            )
            self.conn.commit()

    def load_metric_results(self, run_id: str) -> List:
        """Load metric results from relational table for a run."""
        from ..models import MetricResult

        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM metric_results_rows WHERE run_id = ?", (run_id,)
        )
        rows = cur.fetchall()
        results = []
        for row in rows:
            details = json.loads(row["details"]) if row["details"] else {}
            passed_val = row["passed"]
            passed = bool(passed_val) if passed_val is not None else None
            results.append(
                MetricResult(
                    metric_id=row["metric_id"],
                    item_id=row["item_id"],
                    call_id=row["call_id"] or "",
                    score=row["score"],
                    passed=passed,
                    details=details,
                    unit_id=row["unit_id"],
                    unit_type=row["unit_type"],
                    input_tokens=row["input_tokens"],
                    output_tokens=row["output_tokens"],
                    model=row["model"],
                )
            )
        return results

    def list_eval_runs(self, limit: int = 20) -> List[EvalRun]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM eval_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [self._row_to_eval_run(r) for r in rows]

    def list_eval_runs_by_project(
        self, dataset_name: str, limit: int = 20
    ) -> List[EvalRun]:
        """List eval runs for a specific project (dataset_name)."""
        cur = self.conn.cursor()
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
        return [self._row_to_eval_run(r) for r in rows]

    def get_eval_run(self, run_id: str) -> Optional[EvalRun]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        return self._row_to_eval_run(row) if row else None

    def resolve_call_id(self, short_id: str) -> Optional[str]:
        """Resolve a short ID prefix to full call ID.

        Returns the full ID if exactly one match found, None otherwise.
        Supports both short prefixes (e.g., '6cf21eb3') and full UUIDs.
        """
        # First try exact match
        cur = self.conn.cursor()
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

    def resolve_eval_run_id(self, short_id: str) -> Optional[str]:
        """Resolve a short ID prefix to full eval run ID.

        Returns the full ID if exactly one match found, None otherwise.
        Supports both short prefixes and full UUIDs.
        """
        # First try exact match
        cur = self.conn.cursor()
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
        cur = self.conn.cursor()
        for ann in annotations:
            cur.execute(
                """
                INSERT OR REPLACE INTO annotations
                (id, target_id, label, rationale, annotator, source, confidence, created_at, metric_labels)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
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
                ),
            )
        self.conn.commit()

    def list_annotations(
        self, target_id: Optional[str] = None, limit: int = 100
    ) -> List[Annotation]:
        cur = self.conn.cursor()
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
        rows = cur.fetchall()
        anns: List[Annotation] = []
        for row in rows:
            anns.append(
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
            )
        return anns

    def store_span_metric_links(self, links: Iterable[SpanMetricLink]) -> None:
        cur = self.conn.cursor()
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
        self.conn.commit()

    def list_span_metric_links(
        self,
        run_id: str,
        span_id: Optional[str] = None,
        metric_result_id: Optional[str] = None,
    ) -> List[SpanMetricLink]:
        cur = self.conn.cursor()
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
        cur = self.conn.cursor()
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
        rows = cur.fetchall()
        spans = []
        for r in rows:
            spans.append(
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
            )
        return spans

    def close(self) -> None:
        self.conn.close()

    def _row_to_call(self, row: sqlite3.Row) -> FunctionCall:
        # Handle new columns that may not exist in old databases
        parent_call_id = None
        spans = []
        try:
            parent_call_id = row["parent_call_id"]
            spans_raw = row["spans"]
            if spans_raw:
                spans = json.loads(spans_raw)
        except (KeyError, IndexError):
            # Old database without new columns
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

    def _row_to_eval_run(self, row: sqlite3.Row) -> EvalRun:
        # Load metric results from relational table first, fall back to JSON blob
        run_id = row["id"]
        json_blob = row["metric_results"]
        if json_blob:
            metric_results_raw = json.loads(json_blob)
        else:
            metric_results_raw = [
                r.as_dict() for r in self.load_metric_results(run_id)
            ]

        return EvalRun.from_dict(
            {
                "id": run_id,
                "dataset_name": row["dataset_name"],
                "created_at": row["created_at"],
                "metric_results": metric_results_raw,
                "metrics": json.loads(row["metrics"]) if row["metrics"] else [],
                "judge_configs": json.loads(row["judge_configs"])
                if row["judge_configs"]
                else [],
                "summary": json.loads(row["summary"]) if row["summary"] else {},
                "usage_summary": json.loads(row["usage_summary"])
                if row["usage_summary"]
                else {},
            }
        )
