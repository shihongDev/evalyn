"""
EvalynSpanProcessor for OTEL-native SDKs.

Intercepts OpenTelemetry spans and converts them to Evalyn spans,
adding them to the span collector.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

from .. import context as span_context
from .span_converter import SpanConverter

# Cap on the parent-ID lookup map. Bound prevents linear memory growth
# in long-running processes; the value should comfortably exceed the
# deepest typical span tree in a single Evalyn trace.
_DEFAULT_PARENT_MAP_MAX = 10_000


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

    def __init__(self, parent_map_max: int = _DEFAULT_PARENT_MAP_MAX):
        # OrderedDict keyed by OTEL span_id -> Evalyn span_id, with LRU
        # eviction at `parent_map_max` to keep memory bounded over long
        # running tracers (the original unbounded dict leaked linearly).
        self._parent_id_map: OrderedDict[str, str] = OrderedDict()
        self._parent_map_max = parent_map_max

    def on_start(self, span: Any, parent_context: Any | None = None) -> None:
        """Called when a span starts."""
        # We mainly process on_end, but can use on_start for parent tracking

    def on_end(self, span: ReadableSpan) -> None:
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
            if parent_evalyn_id is not None:
                # Refresh LRU recency: parent was just referenced.
                self._parent_id_map.move_to_end(otel_parent_id)

        # Fall back to current Evalyn span if no mapped parent
        if parent_evalyn_id is None:
            parent_evalyn_id = span_context.get_current_span_id()

        evalyn_span = SpanConverter.from_otel_span(span, parent_evalyn_id)

        # Store mapping for future child spans, evicting the oldest entry
        # if we'd exceed the cap.
        otel_span_id = format(span.context.span_id, "016x")
        self._parent_id_map[otel_span_id] = evalyn_span.id
        if len(self._parent_id_map) > self._parent_map_max:
            self._parent_id_map.popitem(last=False)

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
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    provider.add_span_processor(EvalynSpanProcessor())

    return provider


def get_or_create_tracer_provider() -> Any:
    """
    Get existing TracerProvider or create a new one for Evalyn.

    If OpenTelemetry already has a global provider set, returns that.
    Otherwise creates a new provider with EvalynSpanProcessor.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    current = trace.get_tracer_provider()

    # Check if it's already a TracerProvider (not NoOpTracerProvider)
    if isinstance(current, TracerProvider):
        # Add our processor if not already present
        active = getattr(current, "_active_span_processor", None)
        processors = getattr(active, "_span_processors", []) if active else []
        for processor in processors:
            if isinstance(processor, EvalynSpanProcessor):
                return current

        # Add our processor
        current.add_span_processor(EvalynSpanProcessor())
        return current

    # Create new provider
    provider = create_evalyn_tracer_provider()
    trace.set_tracer_provider(provider)
    return provider
