from __future__ import annotations

import functools
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from collections.abc import Callable

from ..models import Span

# ---------------------------------------------------------------------------
# SpanStack: thread-local parent-child hierarchy
# ---------------------------------------------------------------------------

class SpanStack:
    """Thread-local stack for managing parent-child span hierarchy."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _stack(self) -> list[str]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def push(self, span_id: str) -> None:
        """Push a span ID onto the stack."""
        self._stack().append(span_id)

    def pop(self) -> str | None:
        """Pop and return the top span ID. Returns None if empty."""
        stack = self._stack()
        if not stack:
            return None
        return stack.pop()

    def current(self) -> str | None:
        """Peek at the top span ID without popping."""
        stack = self._stack()
        if not stack:
            return None
        return stack[-1]

    def depth(self) -> int:
        """Current stack depth."""
        return len(self._stack())

    def clear(self) -> None:
        """Clear the stack."""
        self._local.stack = []


# Module-level span stack instance
_STACK = SpanStack()


# ---------------------------------------------------------------------------
# SpanRecord / DecoratorConfig
# ---------------------------------------------------------------------------

@dataclass
class SpanRecord:
    """Record of a span created by the trace_span decorator."""

    span: Span
    function_name: str = ""
    args_summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "span": self.span.as_dict(),
            "function_name": self.function_name,
            "args_summary": self.args_summary,
        }


@dataclass
class DecoratorConfig:
    """Configuration for the trace_span decorator."""

    capture_args: bool = True
    capture_return: bool = False
    max_arg_length: int = 200

    def as_dict(self) -> dict[str, Any]:
        return {
            "capture_args": self.capture_args,
            "capture_return": self.capture_return,
            "max_arg_length": self.max_arg_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecoratorConfig:
        return cls(
            capture_args=data.get("capture_args", True),
            capture_return=data.get("capture_return", False),
            max_arg_length=data.get("max_arg_length", 200),
        )


# ---------------------------------------------------------------------------
# Global span record collection
# ---------------------------------------------------------------------------

_SPAN_RECORDS: list[SpanRecord] = []


def get_recorded_spans() -> list[SpanRecord]:
    """Return all recorded SpanRecords."""
    return list(_SPAN_RECORDS)


def clear_recorded_spans() -> None:
    """Clear all recorded SpanRecords."""
    _SPAN_RECORDS.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize_args(args: tuple, kwargs: dict, max_length: int = 200) -> str:
    """Build a short string summary of function arguments."""
    parts: list[str] = []
    for a in args:
        parts.append(repr(a))
    for k, v in kwargs.items():
        parts.append(f"{k}={v!r}")
    summary = ", ".join(parts)
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    return summary


def _make_span(
    name: str,
    span_type: str,
    parent_id: str | None,
    attributes: dict[str, Any] | None = None,
) -> Span:
    """Create a new Span with a generated ID and current timestamp."""
    return Span(
        id=str(uuid.uuid4()),
        name=name,
        span_type=span_type,
        parent_id=parent_id,
        start_time=datetime.now(timezone.utc),
        status="running",
        attributes=attributes or {},
    )


# ---------------------------------------------------------------------------
# trace_span: decorator and context manager
# ---------------------------------------------------------------------------

class trace_span:
    """Decorator and context manager that creates a proper Span object.

    As decorator:
        @trace_span("my_operation")
        def do_work():
            ...

    As context manager:
        with trace_span("op") as span:
            ...

    Automatically sets parent_id from the SpanStack, records function
    args/kwargs in span attributes, and catches exceptions to set
    status="error".
    """

    def __init__(
        self,
        name: str = "",
        span_type: str = "custom",
        config: DecoratorConfig | None = None,
    ) -> None:
        self._name = name
        self._span_type = span_type
        self._config = config or DecoratorConfig()
        self._span: Span | None = None

    # -- decorator usage ----------------------------------------------------

    def __call__(self, func: Callable) -> Callable:
        span_name = self._name or func.__name__
        span_type = self._span_type
        config = self._config

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            parent_id = _STACK.current()
            attrs: dict[str, Any] = {}
            args_summary = ""
            if config.capture_args:
                args_summary = _summarize_args(args, kwargs, config.max_arg_length)
                attrs["args"] = args_summary

            span = _make_span(span_name, span_type, parent_id, attrs)
            record = SpanRecord(
                span=span,
                function_name=func.__name__,
                args_summary=args_summary,
            )

            _STACK.push(span.id)
            try:
                result = func(*args, **kwargs)
                span.end_time = datetime.now(timezone.utc)
                span.status = "ok"
                if config.capture_return:
                    ret_str = repr(result)
                    if len(ret_str) > config.max_arg_length:
                        ret_str = ret_str[: config.max_arg_length] + "..."
                    span.attributes["return"] = ret_str
                return result
            except Exception as exc:
                span.end_time = datetime.now(timezone.utc)
                span.status = "error"
                span.attributes["error"] = str(exc)
                raise
            finally:
                _STACK.pop()
                _SPAN_RECORDS.append(record)

        return wrapper

    # -- context manager usage ----------------------------------------------

    def __enter__(self) -> Span:
        parent_id = _STACK.current()
        name = self._name or "context_span"
        self._span = _make_span(name, self._span_type, parent_id)
        _STACK.push(self._span.id)
        record = SpanRecord(span=self._span, function_name="", args_summary="")
        _SPAN_RECORDS.append(record)
        return self._span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._span is not None:
            self._span.end_time = datetime.now(timezone.utc)
            if exc_type is not None:
                self._span.status = "error"
                self._span.attributes["error"] = str(exc_val)
            else:
                self._span.status = "ok"
        _STACK.pop()
        return None
