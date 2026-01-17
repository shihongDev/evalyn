"""
Auto-instrumentation for LLM libraries.

Automatically captures:
- LLM API calls (OpenAI, Anthropic, Google Gemini)
- Token usage and cost
- Nested call hierarchies

Auto-instrumentation is ENABLED BY DEFAULT and runs lazily when the first
trace starts (not at import time). This keeps CLI commands fast.

To disable:
    export EVALYN_AUTO_INSTRUMENT=off

Or in code (before starting a trace):
    import os
    os.environ["EVALYN_AUTO_INSTRUMENT"] = "off"

This module now delegates to the unified instrumentation registry while
maintaining backward compatibility with the existing API.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

# Import from the new instrumentation module
from .instrumentation.registry import get_registry


def _get_tracer():
    """Lazy import to avoid circular dependency."""
    from ..decorators import get_default_tracer

    return get_default_tracer()


# Track nested calls (for backwards compat)
_call_stack: contextvars.ContextVar[List[str]] = contextvars.ContextVar(
    "evalyn_call_stack", default=[]
)


def _get_parent_call_id() -> Optional[str]:
    """Get the current parent call ID from the stack."""
    stack = _call_stack.get()
    return stack[-1] if stack else None


@contextmanager
def _push_call(call_id: str):
    """Push a call ID onto the stack for nested tracking."""
    stack = _call_stack.get().copy()
    stack.append(call_id)
    token = _call_stack.set(stack)
    try:
        yield
    finally:
        _call_stack.reset(token)


# =============================================================================
# Backward-compatible patching functions
# These now delegate to the registry
# =============================================================================


def patch_openai() -> bool:
    """Patch OpenAI library to auto-capture LLM calls."""
    registry = get_registry()
    return registry.instrument("openai").get("openai", False)


def patch_anthropic() -> bool:
    """Patch Anthropic library to auto-capture LLM calls."""
    registry = get_registry()
    return registry.instrument("anthropic").get("anthropic", False)


def patch_gemini() -> bool:
    """Patch Google Gemini library to auto-capture LLM calls."""
    registry = get_registry()
    return registry.instrument("gemini").get("gemini", False)


def patch_langchain() -> bool:
    """Patch LangChain to auto-capture LLM and tool calls."""
    registry = get_registry()
    result = registry.instrument("langchain").get("langchain", False)

    # For backwards compat, expose langchain_handler on this module
    if result:
        inst = registry.get_instrumentor("langchain")
        if inst and hasattr(inst, "get_handler"):
            global langchain_handler
            langchain_handler = inst.get_handler()

    return result


def patch_langgraph() -> bool:
    """Patch LangGraph to auto-capture node execution spans."""
    registry = get_registry()
    return registry.instrument("langgraph").get("langgraph", False)


# Track patched state (for backwards compat, now wraps registry)
class _PatchedStateProxy:
    """Proxy to the registry's instrumented state for backwards compat."""

    def __getitem__(self, key: str) -> bool:
        return get_registry().is_instrumented(key)

    def __setitem__(self, key: str, value: bool) -> None:
        # Setting is not supported through the proxy
        pass

    def get(self, key: str, default: bool = False) -> bool:
        return get_registry().is_instrumented(key)

    def copy(self) -> Dict[str, bool]:
        registry = get_registry()
        return {
            name: registry.is_instrumented(name)
            for name in registry.list_instrumentors()
        }


_patched = _PatchedStateProxy()


def patch_all() -> Dict[str, bool]:
    """
    Patch all supported LLM libraries.

    Returns a dict showing which libraries were successfully patched.

    Usage:
        from evalyn_sdk import auto_instrument

        # At startup, before any LLM calls
        results = auto_instrument.patch_all()
        print(results)  # {'openai': True, 'anthropic': False, 'gemini': True, ...}
    """
    registry = get_registry()
    return registry.instrument_all()


def is_patched(library: str) -> bool:
    """Check if a library has been patched."""
    return get_registry().is_instrumented(library)


# =============================================================================
# Trace Decorator (unchanged)
# =============================================================================


def trace(name: Optional[str] = None):
    """
    Lightweight decorator for tracing internal function calls.

    Unlike @eval, this doesn't create a new top-level trace.
    Instead, it logs as an event within the current trace.

    Usage:
        @trace()
        def process_data(data):
            return transform(data)

        @trace("custom_name")
        def another_function():
            pass
    """

    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = _get_tracer()
                start = time.time()
                call_id = str(uuid.uuid4())[:8]

                tracer.log_event(
                    f"trace.{func_name}.start",
                    {
                        "call_id": call_id,
                        "args": str(args)[:200],
                        "kwargs": str(kwargs)[:200],
                        "parent_call_id": _get_parent_call_id(),
                    },
                )

                with _push_call(call_id):
                    try:
                        result = await func(*args, **kwargs)
                        duration_ms = (time.time() - start) * 1000
                        tracer.log_event(
                            f"trace.{func_name}.end",
                            {
                                "call_id": call_id,
                                "duration_ms": duration_ms,
                                "success": True,
                                "result": str(result)[:200],
                            },
                        )
                        return result
                    except Exception as e:
                        duration_ms = (time.time() - start) * 1000
                        tracer.log_event(
                            f"trace.{func_name}.error",
                            {
                                "call_id": call_id,
                                "duration_ms": duration_ms,
                                "success": False,
                                "error": str(e),
                            },
                        )
                        raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                tracer = _get_tracer()
                start = time.time()
                call_id = str(uuid.uuid4())[:8]

                tracer.log_event(
                    f"trace.{func_name}.start",
                    {
                        "call_id": call_id,
                        "args": str(args)[:200],
                        "kwargs": str(kwargs)[:200],
                        "parent_call_id": _get_parent_call_id(),
                    },
                )

                with _push_call(call_id):
                    try:
                        result = func(*args, **kwargs)
                        duration_ms = (time.time() - start) * 1000
                        tracer.log_event(
                            f"trace.{func_name}.end",
                            {
                                "call_id": call_id,
                                "duration_ms": duration_ms,
                                "success": True,
                                "result": str(result)[:200],
                            },
                        )
                        return result
                    except Exception as e:
                        duration_ms = (time.time() - start) * 1000
                        tracer.log_event(
                            f"trace.{func_name}.error",
                            {
                                "call_id": call_id,
                                "duration_ms": duration_ms,
                                "success": False,
                                "error": str(e),
                            },
                        )
                        raise

            return sync_wrapper

    # Allow @trace without parentheses
    if callable(name):
        func = name
        name = None
        return decorator(func)

    return decorator


# =============================================================================
# Lazy auto-patch (on first trace, not at import)
# =============================================================================


# Track auto-patched state for backwards compat
_auto_patched = False


def ensure_patched() -> Dict[str, bool]:
    """Lazily patch all libraries when first needed (not at import time).

    This is called automatically when starting a trace. Call this explicitly
    if you want to patch before the first trace.
    """
    global _auto_patched
    registry = get_registry()
    result = registry.ensure_instrumented()
    _auto_patched = True
    return result


# Placeholder for langchain handler (populated on patch_langchain)
langchain_handler: Optional[Any] = None
