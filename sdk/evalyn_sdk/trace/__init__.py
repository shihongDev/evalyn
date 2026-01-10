"""
Tracing module - hierarchical span tracking for LLM agents.

This module provides Phoenix/LangSmith-style trace visualization with:
- Hierarchical spans with parent-child relationships
- Auto-instrumentation for OpenAI, Anthropic, Gemini, LangChain, LangGraph
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
    # Auto-instrumentation
    "patch_all",
    "patch_openai",
    "patch_anthropic",
    "patch_gemini",
    "patch_langchain",
    "patch_langgraph",
    "is_patched",
    "trace",
]
