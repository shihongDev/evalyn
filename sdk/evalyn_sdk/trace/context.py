"""
Span context propagation for hierarchical tracing.

This module provides context variables and utilities for tracking parent-child
span relationships automatically during function execution. It enables
Phoenix/LangSmith-style trace trees.

Usage:
    from evalyn_sdk.trace import span, get_current_span_id

    with span("my_operation", "custom") as span_obj:
        # Any nested spans created here will have this span as parent
        do_work()
"""

from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Span, FunctionCall

# Context variables for span hierarchy
_span_stack: ContextVar[List[str]] = ContextVar("_span_stack", default=[])
_active_call: ContextVar[Optional["FunctionCall"]] = ContextVar("_active_call", default=None)

# Sentinel to detect uninitialized collector (for context propagation detection)
_UNSET_COLLECTOR: List["Span"] = []
_span_collector: ContextVar[List["Span"]] = ContextVar("_span_collector", default=_UNSET_COLLECTOR)

# Thread-safe global collector fallback (for threads that don't inherit ContextVar)
_global_lock = threading.Lock()
_global_collectors: Dict[str, List["Span"]] = {}
_global_call_id: Optional[str] = None


def _generate_span_id() -> str:
    """Generate a unique span ID."""
    return str(uuid.uuid4())


def get_current_span_id() -> Optional[str]:
    """Get the ID of the current parent span.

    Returns None if no span is active (we're at the root level).
    """
    stack = _span_stack.get()
    return stack[-1] if stack else None


def get_current_call() -> Optional["FunctionCall"]:
    """Get the currently active FunctionCall being traced."""
    return _active_call.get()


def set_current_call(call: Optional["FunctionCall"]) -> None:
    """Set the currently active FunctionCall and manage global collector."""
    global _global_call_id
    _active_call.set(call)

    with _global_lock:
        if call is not None:
            _global_collectors[call.id] = []
            _global_call_id = call.id
        elif _global_call_id:
            _global_collectors.pop(_global_call_id, None)
            _global_call_id = None


def get_span_collector() -> List["Span"]:
    """Get the context-local span collector."""
    return _span_collector.get()


def get_global_spans(call_id: str) -> List["Span"]:
    """Pop and return spans from the global collector for a call."""
    with _global_lock:
        return _global_collectors.pop(call_id, [])


def _add_span_to_collector(span: "Span") -> None:
    """Add span to collector. Falls back to global collector for threads."""
    collector = _span_collector.get()

    # Use context-local collector if initialized
    if collector is not _UNSET_COLLECTOR:
        collector.append(span)
        return

    # Fallback: use global collector (for threads without ContextVar)
    with _global_lock:
        if _global_call_id and _global_call_id in _global_collectors:
            _global_collectors[_global_call_id].append(span)


@contextmanager
def span(
    name: str,
    span_type: str,
    **attributes: Any,
) -> Generator["Span", None, None]:
    """
    Create a child span under the current parent.

    This context manager:
    1. Creates a new Span with the current span as parent
    2. Pushes it onto the span stack (so nested spans have correct parent)
    3. Records start time
    4. On exit, records end time and status
    5. Adds the span to the collector for the current FunctionCall

    Args:
        name: Display name for the span (e.g., "generate_query", "llm_call")
        span_type: Type of span (llm_call, tool_call, node, etc.)
        **attributes: Additional attributes to attach to the span

    Yields:
        The Span object (can be used to add more attributes)

    Example:
        with span("my_llm_call", "llm_call", model="gpt-4") as s:
            response = client.chat.completions.create(...)
            s.attributes["tokens"] = response.usage.total_tokens
    """
    from ..models import Span as SpanModel

    parent_id = get_current_span_id()
    span_id = _generate_span_id()

    # Create the span
    span_obj = SpanModel(
        id=span_id,
        name=name,
        span_type=span_type,
        parent_id=parent_id,
        start_time=datetime.now(timezone.utc),
        status="running",
        attributes=dict(attributes),
    )

    # Push onto stack
    stack = _span_stack.get()
    new_stack = stack + [span_id]
    token = _span_stack.set(new_stack)

    try:
        yield span_obj
        span_obj.finish(status="ok")
    except Exception as e:
        span_obj.finish(status="error", error=str(e))
        raise
    finally:
        # Pop from stack
        _span_stack.set(stack)
        # Add to collector
        _add_span_to_collector(span_obj)


@contextmanager
def root_span(
    name: str,
    call: "FunctionCall",
) -> Generator["Span", None, None]:
    """
    Create a root span for a FunctionCall.

    This is used by the @eval decorator to create the top-level span
    for a traced function. All nested spans will be children of this.

    Args:
        name: Function name
        call: The FunctionCall being traced

    Yields:
        The root Span object
    """
    from ..models import Span as SpanModel

    # Reset the collector for this call
    _span_collector.set([])

    # Set the active call
    set_current_call(call)

    span_id = _generate_span_id()

    # Create root span
    root = SpanModel(
        id=span_id,
        name=name,
        span_type="session",
        parent_id=None,  # Root has no parent
        start_time=datetime.now(timezone.utc),
        status="running",
        attributes={"call_id": call.id},
    )

    # Push onto stack
    stack = _span_stack.get()
    new_stack = stack + [span_id]
    token = _span_stack.set(new_stack)

    try:
        yield root
        root.finish(status="ok")
    except Exception as e:
        root.finish(status="error", error=str(e))
        raise
    finally:
        # Pop from stack
        _span_stack.set(stack)
        # Add root span to collector
        _add_span_to_collector(root)
        # Clear active call
        set_current_call(None)


def create_span(
    name: str,
    span_type: str,
    parent_id: Optional[str] = None,
    **attributes: Any,
) -> "Span":
    """
    Create a span without context management.

    Use this for programmatic span creation where you manage
    start/finish manually. Prefer the `span()` context manager
    for most use cases.

    Args:
        name: Display name
        span_type: Type of span
        parent_id: Parent span ID (uses current if None)
        **attributes: Additional attributes

    Returns:
        New Span object (not yet finished)
    """
    from ..models import Span as SpanModel

    if parent_id is None:
        parent_id = get_current_span_id()

    return SpanModel.new(
        name=name,
        span_type=span_type,
        parent_id=parent_id,
        **attributes,
    )


def record_span(span: "Span") -> None:
    """
    Record a manually-created span to the collector.

    Call this after finishing a span created with create_span().
    """
    _add_span_to_collector(span)
