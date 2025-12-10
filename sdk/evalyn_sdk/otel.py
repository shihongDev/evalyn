from __future__ import annotations

from typing import Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, OTLPSpanExporter

    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    OTEL_AVAILABLE = False
    trace = None  # type: ignore


def configure_otel(
    service_name: str = "evalyn",
    exporter: str = "console",
    endpoint: Optional[str] = None,
):
    """
    Configure an OpenTelemetry tracer provider and return a tracer for Evalyn to use.
    exporter: "console" or "otlp"
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
    else:
        raise ValueError("Unsupported exporter; use 'console' or 'otlp'")

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
