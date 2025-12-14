from __future__ import annotations

from typing import Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    # OTLP exporter import path differs by version; try modern path first.
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except ImportError:
        from opentelemetry.sdk.trace.export import OTLPSpanExporter  # type: ignore
    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    OTEL_AVAILABLE = False
    trace = None  # type: ignore


class SQLiteSpanExporter:
    """
    Minimal OTEL span exporter that writes spans to a SQLite database.
    Spans are keyed by evalyn.call_id (if present in attributes) for easy lookup.
    """

    def __init__(self, path: str = "evalyn.sqlite"):
        import sqlite3

        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_table()

    def _init_table(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS otel_spans (
                span_id TEXT PRIMARY KEY,
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

    def export(self, spans) -> None:
        import json

        cur = self.conn.cursor()
        for span in spans:
            attrs = dict(span.attributes) if getattr(span, "attributes", None) else {}
            events = [
                {"name": ev.name, "attributes": dict(ev.attributes), "timestamp": getattr(ev, "timestamp", None)}
                for ev in getattr(span, "events", []) or []
            ]
            call_id = attrs.get("evalyn.call_id")
            cur.execute(
                """
                INSERT OR REPLACE INTO otel_spans
                (span_id, call_id, name, start_time, end_time, status, attributes, events)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    span.context.span_id.hex,
                    call_id,
                    span.name,
                    getattr(span, "start_time", None),
                    getattr(span, "end_time", None),
                    getattr(getattr(span, "status", None), "status_code", None),
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
    if not OTEL_AVAILABLE:
        raise RuntimeError("opentelemetry-sdk not installed. Install with extras: pip install -e '.[otel]'")

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if exporter == "console":
        processor = BatchSpanProcessor(ConsoleSpanExporter())
    elif exporter == "otlp":
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    elif exporter == "sqlite":
        processor = BatchSpanProcessor(SQLiteSpanExporter(sqlite_path or "evalyn.sqlite"))
    else:
        raise ValueError("Unsupported exporter; use 'console', 'otlp', or 'sqlite'")

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(service_name)
    return tracer


def configure_default_otel(service_name: str = "evalyn", exporter: str = "console", endpoint: str | None = None):
    """
    Convenience wrapper that only acts if opentelemetry is installed; otherwise returns None.
    """
    if not OTEL_AVAILABLE:
        return None
    try:
        return configure_otel(service_name=service_name, exporter=exporter, endpoint=endpoint)
    except Exception:
        return None
