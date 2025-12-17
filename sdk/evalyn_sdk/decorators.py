from __future__ import annotations

import os
from typing import Any, Callable, Optional

from .tracing import EvalTracer
from .storage.sqlite import SQLiteStorage
from .otel import configure_default_otel, OTEL_AVAILABLE

_default_tracer: Optional[EvalTracer] = None


def get_default_tracer() -> EvalTracer:
    """Lazily create a tracer backed by local SQLite storage."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = EvalTracer(SQLiteStorage())
        if not OTEL_AVAILABLE:
            raise RuntimeError(
                "OpenTelemetry dependencies are not installed. Install with: pip install -e \"sdk[otel]\""
            )
        # Auto-enable OpenTelemetry spans if not explicitly disabled.
        otel_flag = os.getenv("EVALYN_OTEL", "on").lower()
        if otel_flag not in {"0", "off", "false"}:
            otel_tracer = configure_default_otel(
                service_name=os.getenv("EVALYN_OTEL_SERVICE", "evalyn"),
                exporter=os.getenv("EVALYN_OTEL_EXPORTER", "sqlite"),
                endpoint=os.getenv("EVALYN_OTEL_ENDPOINT"),
            )
            if otel_tracer:
                _default_tracer.attach_otel_tracer(otel_tracer)
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
