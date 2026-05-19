"""
Haystack framework instrumentor (v2+).

Uses Haystack's official Tracer protocol to capture:
- Pipeline execution (run)
- Component execution (generators, retrievers, builders)
- LLM calls (via component input/output)
- Retrieval operations
- Proper parent-child span hierarchy

Span hierarchy:
    pipeline:run
      -> component:{name} (type=generator)
        -> (LLM calls captured via content tags)
      -> component:{name} (type=retriever)
"""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from ....models import Span
from ... import context as span_context
from ..base import Instrumentor, InstrumentorType


class HaystackInstrumentor(Instrumentor):
    """Instrumentor for Haystack v2+ via its Tracer protocol."""

    _instrumented = False

    @property
    def name(self) -> str:
        return "haystack"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.HOOK_BASED

    def is_available(self) -> bool:
        return importlib.util.find_spec("haystack") is not None

    def is_instrumented(self) -> bool:
        return self._instrumented

    def instrument(self) -> bool:
        if self._instrumented:
            return True

        try:
            from haystack.tracing import tracer as proxy_tracer

            evalyn_tracer = EvalynHaystackTracer()
            proxy_tracer.actual_tracer = evalyn_tracer
            proxy_tracer.is_content_tracing_enabled = True
            self._instrumented = True
            return True
        except (ImportError, AttributeError):
            return False

    def uninstrument(self) -> bool:
        if not self._instrumented:
            return True

        try:
            from haystack.tracing import NullTracer
            from haystack.tracing import tracer as proxy_tracer

            proxy_tracer.actual_tracer = NullTracer()
            proxy_tracer.is_content_tracing_enabled = False
        except (ImportError, AttributeError):
            pass

        self._instrumented = False
        return True


class EvalynHaystackSpan:
    """Span adapter implementing Haystack's Span protocol."""

    def __init__(
        self,
        operation_name: str,
        tags: Optional[Dict[str, Any]] = None,
        parent: Optional[EvalynHaystackSpan] = None,
    ) -> None:
        self.operation_name = operation_name
        self.tags: Dict[str, Any] = tags or {}
        self.parent = parent

        span_type = _map_operation_to_span_type(operation_name)
        parent_span_id = span_context.get_current_span_id()
        self._span = Span.new(
            name=operation_name,
            span_type=span_type,
            parent_id=parent_span_id,
        )

        self._prev_stack = span_context._span_stack.get()
        span_context._span_stack.set(self._prev_stack + [self._span.id])

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value
        safe_value = _safe_tag_value(value)
        if safe_value is not None:
            self._span.attributes[key] = safe_value

    def set_tags(self, tags: Dict[str, Any]) -> None:
        for key, value in tags.items():
            self.set_tag(key, value)

    def set_content_tag(self, key: str, value: Any) -> None:
        """Store content data (input/output) with extended truncation limit."""
        self.tags[key] = value
        safe_value = _safe_tag_value(value, max_length=2000)
        if safe_value is not None:
            self._span.attributes[key] = safe_value

    def raw_span(self) -> Any:
        return self._span

    def get_correlation_data_for_logs(self) -> Dict[str, Any]:
        return {
            "span_id": self._span.id,
            "operation_name": self.operation_name,
        }

    def finish(self, status: str = "ok") -> None:
        _enrich_span_from_tags(self._span, self.tags)
        span_context._span_stack.set(self._prev_stack)
        self._span.finish(status=status)
        span_context._add_span_to_collector(self._span)


class EvalynHaystackTracer:
    """Tracer implementing Haystack's Tracer protocol."""

    @contextmanager
    def trace(
        self,
        operation_name: str,
        tags: Optional[Dict[str, Any]] = None,
        parent_span: Optional[EvalynHaystackSpan] = None,
    ) -> Iterator[EvalynHaystackSpan]:
        span = EvalynHaystackSpan(
            operation_name=operation_name,
            tags=tags,
            parent=parent_span,
        )

        if tags:
            span.set_tags(tags)

        try:
            yield span
            span.finish(status="ok")
        except Exception as e:
            span._span.attributes["error"] = str(e)
            span.finish(status="error")
            raise

    def current_span(self) -> Optional[EvalynHaystackSpan]:
        return None


def _map_operation_to_span_type(operation_name: str) -> str:
    """Map Haystack operation names to evalyn span types."""
    if operation_name == "haystack.pipeline.run":
        return "graph"
    if operation_name == "haystack.component.run":
        return "agent"
    return "custom"


def _enrich_span_from_tags(span: Span, tags: Dict[str, Any]) -> None:
    """Enrich span with structured info extracted from Haystack tags."""
    component_name = tags.get("haystack.component.name")
    component_type = tags.get("haystack.component.type", "")

    if component_name:
        span.name = f"component:{component_name}"

        type_lower = component_type.lower() if component_type else ""
        if "generator" in type_lower:
            span.span_type = "llm_call"
        elif "retriever" in type_lower:
            span.span_type = "retrieval"
        elif "embedder" in type_lower:
            span.span_type = "custom"
            span.attributes["haystack.operation"] = "embedding"

    pipeline_metadata = tags.get("haystack.pipeline.metadata")
    if pipeline_metadata:
        span.attributes["haystack.pipeline.metadata"] = _safe_tag_value(
            pipeline_metadata, max_length=500
        )


def _safe_tag_value(value: Any, max_length: int = 1000) -> Any:
    """Convert a tag value to something safe for storage."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:max_length]
    return str(value)[:max_length]
