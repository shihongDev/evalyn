from __future__ import annotations

from typing import Any, Callable, Optional

from .tracing import EvalTracer
from .storage.sqlite import SQLiteStorage

_default_tracer: Optional[EvalTracer] = None


def get_default_tracer() -> EvalTracer:
    """Lazily create a tracer backed by local SQLite storage."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = EvalTracer(SQLiteStorage())
    return _default_tracer


def configure_tracer(tracer: EvalTracer) -> None:
    """Override the module-level tracer (useful for tests or custom backends)."""
    global _default_tracer
    _default_tracer = tracer


def eval(func: Optional[Callable[..., Any]] = None, *, tracer: Optional[EvalTracer] = None, name: Optional[str] = None):
    """
    Decorator to trace sync/async functions. Example:

    @eval
    def my_agent(user_input: str) -> str:
        ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tracer_obj = tracer or get_default_tracer()
        return tracer_obj.instrument(fn, name=name)

    if func is not None:
        return decorator(func)
    return decorator
