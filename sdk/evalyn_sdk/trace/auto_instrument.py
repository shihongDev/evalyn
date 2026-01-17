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
"""

from __future__ import annotations

import contextvars
import functools
import importlib.util
import inspect
import os
import time
import uuid
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional

from . import context as span_context
from ..models import Span


def _get_tracer():
    """Lazy import to avoid circular dependency."""
    from ..decorators import get_default_tracer

    return get_default_tracer()


# Track nested calls
_call_stack: contextvars.ContextVar[List[str]] = contextvars.ContextVar(
    "evalyn_call_stack", default=[]
)

# Cost per 1M tokens (as of Jan 2025, approximate)
COST_PER_1M_TOKENS = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
}

# Track patched state
_patched = {
    "openai": False,
    "anthropic": False,
    "gemini": False,
    "langchain": False,
    "langgraph": False,
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a given model and token counts."""
    # Normalize model name
    model_lower = model.lower()

    # Find matching cost entry
    for model_key, costs in COST_PER_1M_TOKENS.items():
        if model_key in model_lower:
            input_cost = (input_tokens / 1_000_000) * costs["input"]
            output_cost = (output_tokens / 1_000_000) * costs["output"]
            return input_cost + output_cost

    # Default estimate if model not found
    return (input_tokens + output_tokens) / 1_000_000 * 1.0  # $1 per 1M tokens default


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


def _log_llm_call(
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    duration_ms: float = 0,
    success: bool = True,
    error: Optional[str] = None,
    request: Optional[Dict[str, Any]] = None,
    response: Optional[Dict[str, Any]] = None,
    tool_tokens: int = 0,
    search_queries: Optional[List[str]] = None,
    sources: Optional[List[Dict[str, str]]] = None,
) -> None:
    """Log an LLM call to the tracer and create a span."""
    tracer = _get_tracer()

    cost = calculate_cost(model, input_tokens, output_tokens)

    detail = {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost,
        "duration_ms": duration_ms,
        "success": success,
        "parent_call_id": _get_parent_call_id(),
    }

    if error:
        detail["error"] = error
    if request:
        detail["request"] = request
    if response:
        detail["response"] = response
    if tool_tokens:
        detail["tool_tokens"] = tool_tokens
    if search_queries:
        detail["search_queries"] = search_queries
    if sources:
        detail["sources"] = sources

    # Create span for hierarchy
    parent_span_id = span_context.get_current_span_id()
    span = Span.new(
        name=f"{provider}:{model}",
        span_type="llm_call",
        parent_id=parent_span_id,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )

    # Add tool/grounding info to span
    if tool_tokens:
        span.attributes["tool_tokens"] = tool_tokens
    if search_queries:
        span.attributes["search_queries"] = search_queries
    if sources:
        span.attributes["sources"] = sources

    # Set duration retroactively (span was created after the call)
    span.start_time = span.start_time - timedelta(milliseconds=duration_ms)
    span.finish(status="error" if error else "ok")
    if error:
        span.attributes["error"] = error

    # Add span to collector
    span_context._add_span_to_collector(span)

    # Also log as trace event for backwards compatibility
    detail["span_id"] = span.id
    tracer.log_event(f"{provider}.completion", detail)


def _log_tool_call(
    tool_name: str,
    tool_input: Any,
    tool_output: Any = None,
    duration_ms: float = 0,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    """Log a tool call to the tracer and create a span."""
    tracer = _get_tracer()

    detail = {
        "tool_name": tool_name,
        "input": str(tool_input)[:1000],  # Truncate for storage
        "output": str(tool_output)[:1000] if tool_output else None,
        "duration_ms": duration_ms,
        "success": success,
        "parent_call_id": _get_parent_call_id(),
    }

    if error:
        detail["error"] = error

    # Create span for hierarchy
    parent_span_id = span_context.get_current_span_id()
    span = Span.new(
        name=tool_name,
        span_type="tool_call",
        parent_id=parent_span_id,
        tool_name=tool_name,
    )
    # Set duration retroactively
    span.start_time = span.start_time - timedelta(milliseconds=duration_ms)
    span.finish(status="error" if error else "ok")
    if error:
        span.attributes["error"] = error

    # Add span to collector
    span_context._add_span_to_collector(span)

    # Also log as trace event for backwards compatibility
    detail["span_id"] = span.id
    tracer.log_event("tool.call", detail)


# =============================================================================
# OpenAI Patching
# =============================================================================


def patch_openai() -> bool:
    """Patch OpenAI library to auto-capture LLM calls."""
    if _patched["openai"]:
        return True

    if importlib.util.find_spec("openai") is None:
        return False

    _patch_openai_client_class()

    _patched["openai"] = True
    return True


def _patch_openai_client_class():
    """Patch OpenAI client class to intercept completions."""
    try:
        from openai.resources.chat import completions as chat_completions
    except ImportError:
        return

    # Patch sync completions.create
    if hasattr(chat_completions, "Completions"):
        original_create = chat_completions.Completions.create

        @functools.wraps(original_create)
        def patched_create(self, *args, **kwargs):
            start = time.time()
            model = kwargs.get("model", "unknown")
            messages = kwargs.get("messages", [])

            try:
                response = original_create(self, *args, **kwargs)
                duration_ms = (time.time() - start) * 1000

                # Extract token usage
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                _log_llm_call(
                    provider="openai",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    success=True,
                    request={"messages": str(messages)[:500]},
                    response={
                        "content": str(
                            getattr(response.choices[0].message, "content", "")
                        )[:500]
                        if response.choices
                        else None
                    },
                )

                return response
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                _log_llm_call(
                    provider="openai",
                    model=model,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                )
                raise

        chat_completions.Completions.create = patched_create

    # Patch async completions.create
    if hasattr(chat_completions, "AsyncCompletions"):
        original_acreate = chat_completions.AsyncCompletions.create

        @functools.wraps(original_acreate)
        async def patched_acreate(self, *args, **kwargs):
            start = time.time()
            model = kwargs.get("model", "unknown")

            try:
                response = await original_acreate(self, *args, **kwargs)
                duration_ms = (time.time() - start) * 1000

                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                _log_llm_call(
                    provider="openai",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    success=True,
                )

                return response
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                _log_llm_call(
                    provider="openai",
                    model=model,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                )
                raise

        chat_completions.AsyncCompletions.create = patched_acreate


# =============================================================================
# Anthropic Patching
# =============================================================================


def patch_anthropic() -> bool:
    """Patch Anthropic library to auto-capture LLM calls."""
    if _patched["anthropic"]:
        return True

    try:
        from anthropic.resources import messages as messages_module
    except ImportError:
        return False

    # Patch sync messages.create
    if hasattr(messages_module, "Messages"):
        original_create = messages_module.Messages.create

        @functools.wraps(original_create)
        def patched_create(self, *args, **kwargs):
            start = time.time()
            model = kwargs.get("model", "unknown")
            messages = kwargs.get("messages", [])

            try:
                response = original_create(self, *args, **kwargs)
                duration_ms = (time.time() - start) * 1000

                # Extract token usage from Anthropic response
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

                _log_llm_call(
                    provider="anthropic",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    success=True,
                    request={"messages": str(messages)[:500]},
                )

                return response
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                _log_llm_call(
                    provider="anthropic",
                    model=model,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                )
                raise

        messages_module.Messages.create = patched_create

    # Patch async messages.create
    if hasattr(messages_module, "AsyncMessages"):
        original_acreate = messages_module.AsyncMessages.create

        @functools.wraps(original_acreate)
        async def patched_acreate(self, *args, **kwargs):
            start = time.time()
            model = kwargs.get("model", "unknown")

            try:
                response = await original_acreate(self, *args, **kwargs)
                duration_ms = (time.time() - start) * 1000

                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

                _log_llm_call(
                    provider="anthropic",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    success=True,
                )

                return response
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                _log_llm_call(
                    provider="anthropic",
                    model=model,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                )
                raise

        messages_module.AsyncMessages.create = patched_acreate

    _patched["anthropic"] = True
    return True


# =============================================================================
# Google Gemini Patching
# =============================================================================


def patch_gemini() -> bool:
    """Patch Google Gemini library to auto-capture LLM calls."""
    if _patched["gemini"]:
        return True

    try:
        from google import genai
        from google.genai import models as models_module
    except ImportError:
        try:
            # Try older google-generativeai package
            import google.generativeai as genai

            _patch_gemini_legacy(genai)
            _patched["gemini"] = True
            return True
        except ImportError:
            return False

    # Patch the new google-genai client
    if hasattr(models_module, "Models"):
        original_generate = models_module.Models.generate_content

        @functools.wraps(original_generate)
        def patched_generate(self, *args, **kwargs):
            start = time.time()
            model = kwargs.get("model", args[0] if args else "unknown")
            if hasattr(model, "name"):
                model = model.name

            try:
                response = original_generate(self, *args, **kwargs)
                duration_ms = (time.time() - start) * 1000

                # Extract token usage
                usage = getattr(response, "usage_metadata", None)
                input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                output_tokens = (
                    getattr(usage, "candidates_token_count", 0) if usage else 0
                )
                tool_tokens = (
                    getattr(usage, "tool_use_prompt_token_count", 0) if usage else 0
                )

                # Extract grounding metadata (search queries, sources)
                search_queries = None
                sources = None
                if response.candidates:
                    gm = getattr(response.candidates[0], "grounding_metadata", None)
                    if gm:
                        search_queries = getattr(gm, "web_search_queries", None)
                        chunks = getattr(gm, "grounding_chunks", None)
                        if chunks:
                            sources = []
                            for chunk in chunks[:5]:  # Limit to 5 sources
                                web = getattr(chunk, "web", None)
                                if web:
                                    sources.append(
                                        {
                                            "title": getattr(web, "title", "")[:100],
                                            "uri": getattr(web, "uri", "")[:200],
                                        }
                                    )

                _log_llm_call(
                    provider="gemini",
                    model=str(model),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    success=True,
                    response={"text": str(getattr(response, "text", ""))[:500]},
                    tool_tokens=tool_tokens,
                    search_queries=search_queries,
                    sources=sources,
                )

                return response
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                _log_llm_call(
                    provider="gemini",
                    model=str(model),
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                )
                raise

        models_module.Models.generate_content = patched_generate

    _patched["gemini"] = True
    return True


def _patch_gemini_legacy(genai):
    """Patch older google-generativeai package."""
    if hasattr(genai, "GenerativeModel"):
        original_generate = genai.GenerativeModel.generate_content

        @functools.wraps(original_generate)
        def patched_generate(self, *args, **kwargs):
            start = time.time()
            model = getattr(self, "model_name", "unknown")

            try:
                response = original_generate(self, *args, **kwargs)
                duration_ms = (time.time() - start) * 1000

                # Extract token usage
                usage = getattr(response, "usage_metadata", None)
                input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                output_tokens = (
                    getattr(usage, "candidates_token_count", 0) if usage else 0
                )

                _log_llm_call(
                    provider="gemini",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    success=True,
                )

                return response
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                _log_llm_call(
                    provider="gemini",
                    model=model,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                )
                raise

        genai.GenerativeModel.generate_content = patched_generate


# =============================================================================
# LangChain Patching
# =============================================================================


def patch_langchain() -> bool:
    """Patch LangChain to auto-capture LLM and tool calls."""
    if _patched["langchain"]:
        return True

    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError:
        return False

    # Create a callback handler that logs to Evalyn
    class EvalynCallbackHandler(BaseCallbackHandler):
        def __init__(self):
            self._start_times: Dict[str, float] = {}

        def on_llm_start(self, serialized, prompts, **kwargs):
            run_id = str(kwargs.get("run_id", ""))
            self._start_times[run_id] = time.time()

        def on_llm_end(self, response, **kwargs):
            run_id = str(kwargs.get("run_id", ""))
            start = self._start_times.pop(run_id, time.time())
            duration_ms = (time.time() - start) * 1000

            # Extract info from response
            llm_output = getattr(response, "llm_output", {}) or {}
            token_usage = llm_output.get("token_usage", {})
            model = llm_output.get("model_name", "unknown")

            _log_llm_call(
                provider="langchain",
                model=model,
                input_tokens=token_usage.get("prompt_tokens", 0),
                output_tokens=token_usage.get("completion_tokens", 0),
                duration_ms=duration_ms,
                success=True,
            )

        def on_llm_error(self, error, **kwargs):
            run_id = str(kwargs.get("run_id", ""))
            start = self._start_times.pop(run_id, time.time())
            duration_ms = (time.time() - start) * 1000

            _log_llm_call(
                provider="langchain",
                model="unknown",
                duration_ms=duration_ms,
                success=False,
                error=str(error),
            )

        def on_tool_start(self, serialized, input_str, **kwargs):
            run_id = str(kwargs.get("run_id", ""))
            self._start_times[f"tool_{run_id}"] = time.time()

        def on_tool_end(self, output, **kwargs):
            run_id = str(kwargs.get("run_id", ""))
            start = self._start_times.pop(f"tool_{run_id}", time.time())
            duration_ms = (time.time() - start) * 1000

            tool_name = kwargs.get("name", "unknown")
            _log_tool_call(
                tool_name=tool_name,
                tool_input=kwargs.get("input", ""),
                tool_output=output,
                duration_ms=duration_ms,
                success=True,
            )

        def on_tool_error(self, error, **kwargs):
            run_id = str(kwargs.get("run_id", ""))
            start = self._start_times.pop(f"tool_{run_id}", time.time())
            duration_ms = (time.time() - start) * 1000

            tool_name = kwargs.get("name", "unknown")
            _log_tool_call(
                tool_name=tool_name,
                tool_input=kwargs.get("input", ""),
                duration_ms=duration_ms,
                success=False,
                error=str(error),
            )

    # Store handler for later use
    _langchain_handler = EvalynCallbackHandler()

    # Make handler available
    import evalyn_sdk.trace.auto_instrument as self_module

    self_module.langchain_handler = _langchain_handler

    _patched["langchain"] = True
    return True


# =============================================================================
# Trace Decorator
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
# Main Entry Point
# =============================================================================


def patch_langgraph() -> bool:
    """Patch LangGraph to auto-capture node execution spans."""
    if _patched["langgraph"]:
        return True

    try:
        from .langgraph import _do_patch_langgraph

        result = _do_patch_langgraph()
        _patched["langgraph"] = result
        return result
    except ImportError:
        return False


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
    return {
        "openai": patch_openai(),
        "anthropic": patch_anthropic(),
        "gemini": patch_gemini(),
        "langchain": patch_langchain(),
        "langgraph": patch_langgraph(),
    }


def is_patched(library: str) -> bool:
    """Check if a library has been patched."""
    return _patched.get(library, False)


# =============================================================================
# Lazy auto-patch (on first trace, not at import)
# =============================================================================

_auto_patched = False


def ensure_patched() -> Dict[str, bool]:
    """Lazily patch all libraries when first needed (not at import time).

    This is called automatically when starting a trace. Call this explicitly
    if you want to patch before the first trace.
    """
    global _auto_patched
    if _auto_patched:
        return _patched.copy()
    _auto_patched = True

    if os.environ.get("EVALYN_AUTO_INSTRUMENT", "").lower() in (
        "off",
        "false",
        "0",
        "no",
    ):
        return {}
    return patch_all()
