"""Schema migrations for SQLite storage.

Each migration checks whether a column exists before adding it,
making every migration idempotent and safe to re-run.
"""

from __future__ import annotations

import sqlite3


def _get_columns(cur: sqlite3.Cursor, table: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _add_missing_columns(
    conn: sqlite3.Connection,
    table: str,
    columns: list[tuple[str, str]],
) -> None:
    cur = conn.cursor()
    existing = _get_columns(cur, table)
    for col, col_type in columns:
        if col not in existing:
            try:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass  # concurrent process already added the column
    conn.commit()


def run_migrations(conn: sqlite3.Connection) -> None:
    """Run all schema migrations against the given connection."""
    # otel_spans: trace correlation columns
    _add_missing_columns(conn, "otel_spans", [
        ("trace_id", "TEXT"),
        ("parent_span_id", "TEXT"),
    ])

    # function_calls: hierarchical span columns
    _add_missing_columns(conn, "function_calls", [
        ("parent_call_id", "TEXT"),
        ("spans", "TEXT"),
    ])

    # eval_runs: token/cost tracking
    _add_missing_columns(conn, "eval_runs", [
        ("usage_summary", "TEXT"),
    ])

    # annotations: per-metric labels
    _add_missing_columns(conn, "annotations", [
        ("metric_labels", "TEXT"),
    ])
