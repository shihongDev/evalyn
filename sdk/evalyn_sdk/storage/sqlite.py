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

    def __init__(self, path: str | Path = "evalyn.sqlite"):
        self.path = Path(path)
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

    def store_call(self, call: FunctionCall) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO function_calls
            (id, function_name, session_id, started_at, ended_at, duration_ms, inputs, output, error, trace, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def get_eval_run(self, run_id: str) -> Optional[EvalRun]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        return self._row_to_eval_run(row) if row else None

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

    def list_annotations(self, target_id: Optional[str] = None, limit: int = 100) -> List[Annotation]:
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

    def close(self) -> None:
        self.conn.close()

    def _row_to_call(self, row: sqlite3.Row) -> FunctionCall:
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
            }
        )

    def _row_to_eval_run(self, row: sqlite3.Row) -> EvalRun:
        return EvalRun.from_dict(
            {
                "id": row["id"],
                "dataset_name": row["dataset_name"],
                "created_at": row["created_at"],
                "metric_results": json.loads(row["metric_results"]) if row["metric_results"] else [],
                "metrics": json.loads(row["metrics"]) if row["metrics"] else [],
                "judge_configs": json.loads(row["judge_configs"]) if row["judge_configs"] else [],
                "summary": json.loads(row["summary"]) if row["summary"] else {},
            }
        )
