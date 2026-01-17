"""
Tracing module - hierarchical span tracking for LLM agents.

This module provides Phoenix/LangSmith-style trace visualization with:
- Hierarchical spans with parent-child relationships
- Auto-instrumentation for OpenAI, Anthropic, Gemini, LangChain, LangGraph
- OTEL-native instrumentation for Google ADK
- Hook-based instrumentation for Anthropic Agent SDK
- Context propagation for nested spans

Usage:
    from evalyn_sdk.trace import span, EvalTracer

    @eval(project='myproj')
    def my_agent(query):
        with span("reasoning", "custom"):
            result = llm.generate(...)
        return result

    # View trace tree:
    # evalyn show-trace --id <call_id>

For advanced instrumentation:
    from evalyn_sdk.trace.instrumentation import (
        instrument,       # Instrument specific SDKs
        instrument_all,   # Instrument all available SDKs
        list_available,   # List installed SDKs
        create_agent_hooks,  # Create hooks for Anthropic Agent SDK
    )
"""

from .context import (
    span,
    root_span,
    get_current_span_id,
    get_current_call,
    set_current_call,
    create_span,
    record_span,
)
from .tracer import EvalTracer, eval_session
from .auto_instrument import (
    patch_all,
    patch_openai,
    patch_anthropic,
    patch_gemini,
    patch_langchain,
    patch_langgraph,
    is_patched,
    trace,
    ensure_patched,
)

# New instrumentation API
from .instrumentation import (
    instrument,
    instrument_all,
    uninstrument,
    uninstrument_all,
    is_instrumented,
    list_available,
    list_instrumented,
    get_hooks,
    create_agent_hooks,
)

__all__ = [
    # Context
    "span",
    "root_span",
    "get_current_span_id",
    "get_current_call",
    "set_current_call",
    "create_span",
    "record_span",
    # Tracer
    "EvalTracer",
    "eval_session",
    # Auto-instrumentation (legacy API - backwards compat)
    "patch_all",
    "patch_openai",
    "patch_anthropic",
    "patch_gemini",
    "patch_langchain",
    "patch_langgraph",
    "is_patched",
    "trace",
    "ensure_patched",
    # New instrumentation API
    "instrument",
    "instrument_all",
    "uninstrument",
    "uninstrument_all",
    "is_instrumented",
    "list_available",
    "list_instrumented",
    "get_hooks",
    "create_agent_hooks",
]
