from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from .base import StorageBackend
from ..models import Annotation, EvalRun, FunctionCall


def _dumps(data: object) -> str:
    return json.dumps(data, default=lambda o: repr(o))


class SQLiteStorage(StorageBackend):
    """Lightweight SQLite backend for local development."""

    def __init__(self, path: str | Path = "data/evalyn.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

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
                summary TEXT
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
                created_at TEXT
            )
            """
        )
        self.conn.commit()
        self._ensure_otel_columns()
        self._ensure_span_columns()

    def _ensure_otel_columns(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(otel_spans)")
        cols = {row[1] for row in cur.fetchall()}
        for col, col_type in [
            ("trace_id", "TEXT"),
            ("parent_span_id", "TEXT"),
        ]:
            if col not in cols:
                cur.execute(f"ALTER TABLE otel_spans ADD COLUMN {col} {col_type}")
        self.conn.commit()

    def _ensure_span_columns(self) -> None:
        """Add hierarchical span columns to function_calls table."""
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(function_calls)")
        cols = {row[1] for row in cur.fetchall()}
        for col, col_type in [
            ("parent_call_id", "TEXT"),  # Parent @eval call
            ("spans", "TEXT"),  # JSON array of Span objects
        ]:
            if col not in cols:
                cur.execute(f"ALTER TABLE function_calls ADD COLUMN {col} {col_type}")
        self.conn.commit()

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

    def list_calls(self, limit: int = 100) -> List[FunctionCall]:
        cur = self.conn.cursor()
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

    def store_eval_run(self, run: EvalRun) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO eval_runs
            (id, dataset_name, created_at, metric_results, metrics, judge_configs, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                run.dataset_name,
                run.created_at.isoformat(),
                _dumps([r.as_dict() for r in run.metric_results]),
                _dumps([m.__dict__ for m in run.metrics]),
                _dumps([j.__dict__ for j in run.judge_configs]),
                _dumps(run.summary),
            ),
        )
        self.conn.commit()

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
                (id, target_id, label, rationale, annotator, source, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                    }
                )
            )
        return anns

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
        return EvalRun.from_dict(
            {
                "id": row["id"],
                "dataset_name": row["dataset_name"],
                "created_at": row["created_at"],
                "metric_results": json.loads(row["metric_results"])
                if row["metric_results"]
                else [],
                "metrics": json.loads(row["metrics"]) if row["metrics"] else [],
                "judge_configs": json.loads(row["judge_configs"])
                if row["judge_configs"]
                else [],
                "summary": json.loads(row["summary"]) if row["summary"] else {},
            }
        )
