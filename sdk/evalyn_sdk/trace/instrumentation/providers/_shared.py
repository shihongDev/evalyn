"""
Shared utilities for instrumentation providers.

Contains common functionality used across monkey-patch instrumentors.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Optional

from ....models import Span
from ... import context as span_context


# Cost per 1M tokens (as of Jan 2025)
# Format: {"input": X, "output": Y, "cache_write": Z, "cache_read": W}
# cache_write is typically 1.25x input, cache_read is typically 0.1x input
# Sources:
#   - Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
#   - Google: https://ai.google.dev/gemini-api/docs/pricing
#   - OpenAI: https://openai.com/api/pricing
COST_PER_1M_TOKENS = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic Claude 4.5 models
    "claude-opus-4-5": {
        "input": 5.00,
        "output": 25.00,
        "cache_write": 6.25,
        "cache_read": 0.50,
    },
    "claude-sonnet-4-5": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-haiku-4-5": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
    # Anthropic Claude 4.1 models
    "claude-opus-4-1": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    # Anthropic Claude 4 models
    "claude-sonnet-4": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-opus-4": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    # Anthropic Claude 3.5 models
    "claude-3-5-sonnet": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-3-5-haiku": {
        "input": 0.80,
        "output": 4.00,
        "cache_write": 1.00,
        "cache_read": 0.08,
    },
    # Anthropic Claude 3 models
    "claude-3-opus": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    "claude-3-sonnet": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-3-haiku": {
        "input": 0.25,
        "output": 1.25,
        "cache_write": 0.3125,
        "cache_read": 0.025,
    },
    # Google Gemini 2.5 models
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    # Google Gemini 2.0 models
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    # Google Gemini 1.5 models
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # xAI Grok models (more specific names must come first for substring matching)
    # Source: https://docs.x.ai/docs/models
    "grok-4-1-fast-reasoning": {"input": 0.20, "output": 0.50},
    "grok-4-1-fast-non-reasoning": {"input": 0.20, "output": 0.50},
    "grok-4-fast-reasoning": {"input": 0.20, "output": 0.50},
    "grok-4-fast-non-reasoning": {"input": 0.20, "output": 0.50},
    "grok-code-fast-1": {"input": 0.20, "output": 1.50},
    "grok-4-0709": {"input": 3.00, "output": 15.00},
    "grok-4": {"input": 3.00, "output": 15.00},  # grok-4-latest maps here
}


def is_model_pricing_known(model: str) -> bool:
    """Check if a model's pricing is in our registry."""
    model_lower = model.lower()
    return any(model_key in model_lower for model_key in COST_PER_1M_TOKENS)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a given model and token counts."""
    model_lower = model.lower()

    for model_key, costs in COST_PER_1M_TOKENS.items():
        if model_key in model_lower:
            input_cost = (input_tokens / 1_000_000) * costs["input"]
            output_cost = (output_tokens / 1_000_000) * costs["output"]
            return input_cost + output_cost

    # Default estimate if model not found: $1/1M tokens
    return (input_tokens + output_tokens) / 1_000_000 * 1.0


def calculate_cost_with_cache(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """
    Calculate cost in USD including cache token costs.

    Args:
        model: Model name/ID
        input_tokens: Regular input tokens (non-cached)
        output_tokens: Output tokens
        cache_creation_tokens: Tokens written to cache (charged at cache_write rate)
        cache_read_tokens: Tokens read from cache (charged at cache_read rate)

    Returns:
        Total cost in USD
    """
    model_lower = model.lower()

    for model_key, costs in COST_PER_1M_TOKENS.items():
        if model_key in model_lower:
            input_cost = (input_tokens / 1_000_000) * costs["input"]
            output_cost = (output_tokens / 1_000_000) * costs["output"]

            # Cache costs (use input rate as fallback if not specified)
            cache_write_rate = costs.get("cache_write", costs["input"] * 1.25)
            cache_read_rate = costs.get("cache_read", costs["input"] * 0.1)

            cache_write_cost = (cache_creation_tokens / 1_000_000) * cache_write_rate
            cache_read_cost = (cache_read_tokens / 1_000_000) * cache_read_rate

            return input_cost + output_cost + cache_write_cost + cache_read_cost

    # Default estimate if model not found
    total_tokens = input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens
    return total_tokens / 1_000_000 * 1.0


def _get_tracer():
    """Lazy import to avoid circular dependency."""
    from ....decorators import get_default_tracer

    return get_default_tracer()


def log_llm_call(
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
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> None:
    """Log an LLM call to the tracer and create a span."""
    tracer = _get_tracer()

    # Use cache-aware cost calculation
    cost = calculate_cost_with_cache(
        model, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens
    )

    detail = {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost,
        "duration_ms": duration_ms,
        "success": success,
    }

    # Add cache token info if present
    if cache_creation_tokens:
        detail["cache_creation_tokens"] = cache_creation_tokens
    if cache_read_tokens:
        detail["cache_read_tokens"] = cache_read_tokens

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

    # Add cache token info to span
    if cache_creation_tokens:
        span.attributes["cache_creation_input_tokens"] = cache_creation_tokens
    if cache_read_tokens:
        span.attributes["cache_read_input_tokens"] = cache_read_tokens

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


def log_tool_call(
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
