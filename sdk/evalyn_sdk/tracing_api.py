"""Public tracing API for manual span creation.

Provides a clean, user-friendly API for creating custom spans
without needing to understand the internal tracing machinery.

Usage:
    from evalyn_sdk.tracing_api import trace_span

    with trace_span("my_operation", "custom") as s:
        result = do_work()
        s.attributes["result_size"] = len(result)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from .trace.context import span as _internal_span


@contextmanager
def trace_span(
    name: str,
    span_type: str = "custom",
    **attributes: Any,
) -> Generator:
    """Create a traced span for manual instrumentation.

    Use this when you want to trace a specific operation that isn't
    automatically captured by auto-instrumentation. The span will be
    attached to the current active trace (if any).

    Args:
        name: Human-readable name for this span.
        span_type: Span type. Common values:
            "custom" (default), "llm_call", "tool_call", "retrieval",
            "agent", "node", "graph"
        **attributes: Key-value pairs to attach to the span.

    Yields:
        Span object. You can set additional attributes on it:
            span.attributes["key"] = "value"

    Example:
        with trace_span("search_docs", "retrieval", query="hello") as s:
            results = search(query)
            s.attributes["result_count"] = len(results)
    """
    with _internal_span(name, span_type, **attributes) as s:
        yield s


@contextmanager
def trace_llm_call(
    model: str,
    **attributes: Any,
) -> Generator:
    """Convenience wrapper for tracing LLM calls.

    Args:
        model: Model name/ID being called.
        **attributes: Additional attributes (temperature, max_tokens, etc.).

    Yields:
        Span object for the LLM call.
    """
    attrs = {"model": model, **attributes}
    with _internal_span(f"llm_call:{model}", "llm_call", **attrs) as s:
        yield s


@contextmanager
def trace_tool_call(
    tool_name: str,
    **attributes: Any,
) -> Generator:
    """Convenience wrapper for tracing tool/function calls.

    Args:
        tool_name: Name of the tool being called.
        **attributes: Additional attributes (arguments, etc.).

    Yields:
        Span object for the tool call.
    """
    attrs = {"tool.name": tool_name, **attributes}
    with _internal_span(f"tool_call:{tool_name}", "tool_call", **attrs) as s:
        yield s


@contextmanager
def trace_retrieval(
    query: str,
    source: str = "unknown",
    **attributes: Any,
) -> Generator:
    """Convenience wrapper for tracing retrieval operations.

    Args:
        query: The retrieval query.
        source: Data source name (e.g., "pinecone", "chroma").
        **attributes: Additional attributes.

    Yields:
        Span object for the retrieval operation.
    """
    attrs = {"retrieval.query": query, "retrieval.source": source, **attributes}
    with _internal_span(f"retrieval:{source}", "retrieval", **attrs) as s:
        yield s
