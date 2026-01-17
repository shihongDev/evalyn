"""
EvalynSpanProcessor for OTEL-native SDKs.

Intercepts OpenTelemetry spans and converts them to Evalyn spans,
adding them to the span collector.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

from .span_converter import SpanConverter
from .. import context as span_context


class EvalynSpanProcessor:
    """
    OpenTelemetry SpanProcessor that captures spans for Evalyn.

    Converts OTEL ReadableSpan objects to Evalyn Span objects and
    adds them to the span collector for the current FunctionCall.

    Usage:
        from opentelemetry.sdk.trace import TracerProvider
        from evalyn_sdk.trace.instrumentation import EvalynSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(EvalynSpanProcessor())
    """

    def __init__(self):
        self._parent_id_map: Dict[str, str] = {}  # OTEL span_id -> Evalyn span_id

    def on_start(self, span: Any, parent_context: Optional[Any] = None) -> None:
        """Called when a span starts."""
        # We mainly process on_end, but can use on_start for parent tracking
        pass

    def on_end(self, span: "ReadableSpan") -> None:
        """Called when a span ends. Convert and record the span."""
        # Check if we're in an active Evalyn trace
        if span_context.get_current_call() is None:
            return

        # Convert OTEL span to Evalyn span
        # Look up parent in our ID map for proper hierarchy
        parent_evalyn_id = None
        if span.parent:
            otel_parent_id = format(span.parent.span_id, "016x")
            parent_evalyn_id = self._parent_id_map.get(otel_parent_id)

        # Fall back to current Evalyn span if no mapped parent
        if parent_evalyn_id is None:
            parent_evalyn_id = span_context.get_current_span_id()

        evalyn_span = SpanConverter.from_otel_span(span, parent_evalyn_id)

        # Store mapping for future child spans
        otel_span_id = format(span.context.span_id, "016x")
        self._parent_id_map[otel_span_id] = evalyn_span.id

        # Add to Evalyn collector
        span_context._add_span_to_collector(evalyn_span)

    def shutdown(self) -> None:
        """Shutdown the processor."""
        self._parent_id_map.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending spans."""
        return True


def create_evalyn_tracer_provider() -> Any:
    """
    Create an OpenTelemetry TracerProvider configured for Evalyn.

    Returns a TracerProvider with EvalynSpanProcessor attached.
    """
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    except ImportError:
        raise ImportError(
            "opentelemetry-sdk is required for OTEL-native instrumentation. "
            "Install with: pip install opentelemetry-sdk"
        )

    provider = TracerProvider()
    provider.add_span_processor(EvalynSpanProcessor())

    return provider


def get_or_create_tracer_provider() -> Any:
    """
    Get existing TracerProvider or create a new one for Evalyn.

    If OpenTelemetry already has a global provider set, returns that.
    Otherwise creates a new provider with EvalynSpanProcessor.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError:
        raise ImportError(
            "opentelemetry-sdk is required for OTEL-native instrumentation. "
            "Install with: pip install opentelemetry-sdk"
        )

    current = trace.get_tracer_provider()

    # Check if it's already a TracerProvider (not NoOpTracerProvider)
    if isinstance(current, TracerProvider):
        # Add our processor if not already present
        for processor in getattr(current, "_active_span_processor", {}).get("_span_processors", []):
            if isinstance(processor, EvalynSpanProcessor):
                return current

        # Add our processor
        current.add_span_processor(EvalynSpanProcessor())
        return current

    # Create new provider
    provider = create_evalyn_tracer_provider()
    trace.set_tracer_provider(provider)
    return provider
