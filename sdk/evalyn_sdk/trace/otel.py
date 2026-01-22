from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Default path for prod/test separation
DEFAULT_PROD_DB = "data/prod/traces.sqlite"


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

# OTLP exporter import path differs by version; try modern path first.
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
except ImportError:
    from opentelemetry.sdk.trace.export import OTLPSpanExporter  # type: ignore

# Kept for backward compatibility - always True now
OTEL_AVAILABLE = True


class SQLiteSpanExporter:
    """
    Minimal OTEL span exporter that writes spans to a SQLite database.
    Spans are keyed by evalyn.call_id (if present in attributes) for easy lookup.
    """

    def __init__(self, path: str | None = None):
        import sqlite3

        self.path = path or _get_default_db_path()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_table()

    def _init_table(self) -> None:
        cur = self.conn.cursor()
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
        self.conn.commit()

        # Ensure new columns exist when table was created by an older version
        cur.execute("PRAGMA table_info(otel_spans)")
        existing = {row[1] for row in cur.fetchall()}
        for col, col_type in [
            ("trace_id", "TEXT"),
            ("parent_span_id", "TEXT"),
        ]:
            if col not in existing:
                cur.execute(f"ALTER TABLE otel_spans ADD COLUMN {col} {col_type}")
        self.conn.commit()

    def export(self, spans) -> None:
        import json

        cur = self.conn.cursor()
        for span in spans:

            def _format_id(value, width: int) -> str | None:
                if value is None:
                    return None
                if hasattr(value, "hex"):
                    try:
                        return value.hex
                    except Exception:
                        pass
                if isinstance(value, int):
                    return f"{value:0{width}x}"
                if isinstance(value, str):
                    raw = value.strip().lower()
                    if raw.startswith("0x"):
                        raw = raw[2:]
                    if raw and all(c in "0123456789abcdef" for c in raw):
                        return raw.zfill(width)
                    return value
                try:
                    return value.hex()
                except Exception:
                    return str(value)

            attrs = dict(span.attributes) if getattr(span, "attributes", None) else {}
            events = [
                {
                    "name": ev.name,
                    "attributes": dict(ev.attributes),
                    "timestamp": getattr(ev, "timestamp", None),
                }
                for ev in getattr(span, "events", []) or []
            ]
            call_id = attrs.get("evalyn.call_id")
            parent_span_id = None
            if getattr(span, "parent", None):
                try:
                    parent_span_id = _format_id(span.parent.span_id, 16)
                except Exception:
                    parent_span_id = None
            trace_id = None
            span_id = None
            if getattr(span, "context", None):
                trace_id = _format_id(span.context.trace_id, 32)
                span_id = _format_id(span.context.span_id, 16)
            status = getattr(getattr(span, "status", None), "status_code", None)
            if status is not None and hasattr(status, "name"):
                status = status.name
            cur.execute(
                """
                INSERT OR REPLACE INTO otel_spans
                (trace_id, span_id, parent_span_id, call_id, name, start_time, end_time, status, attributes, events)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    span_id,
                    parent_span_id,
                    call_id,
                    span.name,
                    getattr(span, "start_time", None),
                    getattr(span, "end_time", None),
                    status,
                    json.dumps(attrs, default=str),
                    json.dumps(events, default=str),
                ),
            )
        self.conn.commit()
        return None

    def shutdown(self) -> None:
        self.conn.close()


def configure_otel(
    service_name: str = "evalyn",
    exporter: str = "console",
    endpoint: Optional[str] = None,
    sqlite_path: Optional[str] = None,
):
    """
    Configure an OpenTelemetry tracer provider and return a tracer for Evalyn to use.
    exporter: "console", "otlp", or "sqlite"
    endpoint: OTLP endpoint when exporter="otlp" (grpc/http depending on installed exporter)
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if exporter == "console":
        processor = BatchSpanProcessor(ConsoleSpanExporter())
    elif exporter == "otlp":
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    elif exporter == "sqlite":
        processor = BatchSpanProcessor(SQLiteSpanExporter(sqlite_path))
    else:
        raise ValueError("Unsupported exporter; use 'console', 'otlp', or 'sqlite'")

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(service_name)
    return tracer


def configure_default_otel(
    service_name: str = "evalyn", exporter: str = "sqlite", endpoint: str | None = None
):
    """
    Convenience wrapper that configures OTEL with default settings.
    Returns None on any configuration error.
    """
    try:
        return configure_otel(
            service_name=service_name, exporter=exporter, endpoint=endpoint
        )
    except Exception:
        return None
