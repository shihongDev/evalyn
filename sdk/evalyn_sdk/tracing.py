from __future__ import annotations

import contextvars
import functools
import inspect
import json
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple

from .models import FunctionCall, TraceEvent, now_utc
from .storage.base import StorageBackend

_current_session: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "evalyn_session", default=None
)
_active_call: contextvars.ContextVar[Optional[FunctionCall]] = contextvars.ContextVar(
    "evalyn_active_call", default=None
)


def _safe_value(value: Any) -> Any:
    """Convert values to something JSON serializable; fallback to repr."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _normalize_inputs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "args": [_safe_value(a) for a in args],
        "kwargs": {k: _safe_value(v) for k, v in kwargs.items()},
    }


@contextmanager
def eval_session(session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """Context manager to group calls under a shared session id."""
    session_payload = {"id": session_id or str(uuid.uuid4()), "metadata": metadata or {}}
    token = _current_session.set(session_payload)
    try:
        yield session_payload["id"]
    finally:
        _current_session.reset(token)


class EvalTracer:
    def __init__(self, storage: Optional[StorageBackend] = None):
        self.storage = storage
        self._last_call: Optional[FunctionCall] = None

    @property
    def last_call(self) -> Optional[FunctionCall]:
        return self._last_call

    def attach_storage(self, storage: StorageBackend) -> None:
        self.storage = storage

    def log_event(self, kind: str, detail: Optional[Dict[str, Any]] = None) -> None:
        call = _active_call.get()
        if call is None:
            return
        call.trace.append(TraceEvent(kind=kind, timestamp=now_utc(), detail=detail or {}))

    def start_call(
        self, function_name: str, inputs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[FunctionCall, contextvars.Token]:
        session = _current_session.get()
        call = FunctionCall.new(
            function_name=function_name,
            inputs=inputs,
            session_id=session["id"] if session else None,
            metadata=metadata or {},
        )
        token = _active_call.set(call)
        return call, token

    def finish_call(
        self,
        call: FunctionCall,
        token: contextvars.Token,
        *,
        output: Any = None,
        error: Optional[str] = None,
    ) -> FunctionCall:
        call.output = output
        call.error = error
        call.ended_at = now_utc()
        call.duration_ms = (call.ended_at - call.started_at).total_seconds() * 1000
        _active_call.reset(token)
        self._last_call = call
        if self.storage:
            self.storage.store_call(call)
        return call

    def instrument(self, func: Callable[..., Any], name: Optional[str] = None) -> Callable[..., Any]:
        """Wrap any callable to record inputs/outputs/errors via this tracer."""
        function_name = name or getattr(func, "__name__", "anonymous")
        tracer = self

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                inputs = _normalize_inputs(args, kwargs)
                call, token = tracer.start_call(function_name, inputs)
                try:
                    result = await func(*args, **kwargs)
                    tracer.finish_call(call, token, output=result)
                    return result
                except Exception as exc:  # pragma: no cover - re-raises
                    tracer.finish_call(call, token, error=str(exc))
                    raise

            async_wrapper._evalyn_instrumented = True  # type: ignore[attr-defined]
            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            inputs = _normalize_inputs(args, kwargs)
            call, token = tracer.start_call(function_name, inputs)
            try:
                result = func(*args, **kwargs)
                tracer.finish_call(call, token, output=result)
                return result
            except Exception as exc:  # pragma: no cover - re-raises
                tracer.finish_call(call, token, error=str(exc))
                raise

        sync_wrapper._evalyn_instrumented = True  # type: ignore[attr-defined]
        return sync_wrapper
