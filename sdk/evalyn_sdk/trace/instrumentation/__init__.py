"""
Unified instrumentation module for Evalyn.

Provides a consistent interface for instrumenting various LLM SDKs:
- Monkey-patch: OpenAI, Anthropic, Gemini, LangChain, LangGraph
- OTEL-native: Google ADK
- Hook-based: Anthropic Agent SDK

Usage:
    # Auto-instrument all available SDKs (default)
    from evalyn_sdk import eval
    @eval
    def my_agent(): ...

    # Selective instrumentation
    from evalyn_sdk.trace.instrumentation import instrument
    instrument("openai", "google_adk")

    # Hook-based (explicit)
    from evalyn_sdk.trace.instrumentation import create_agent_hooks
    agent = Agent(hooks=create_agent_hooks())
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Instrumentor, InstrumentorType
from .registry import InstrumentorRegistry, get_registry
from .span_converter import SpanConverter
from .span_processor import EvalynSpanProcessor, create_evalyn_tracer_provider

# Import providers and register them
from .providers.openai import OpenAIInstrumentor
from .providers.anthropic import AnthropicInstrumentor
from .providers.gemini import GeminiInstrumentor
from .providers.langchain import LangChainInstrumentor
from .providers.langgraph import LangGraphInstrumentor

# Optional providers
try:
    from .providers.google_adk import (
        GoogleADKInstrumentor,
        EvalynADKCallbacks,
        ADKStreamAdapter,
        create_adk_callbacks,
        create_stream_adapter as create_adk_stream_adapter,
    )

    _has_google_adk = True
except ImportError:
    _has_google_adk = False

    def create_adk_callbacks(*args, **kwargs):  # type: ignore
        raise ImportError("google-adk not installed")

    def create_adk_stream_adapter(*args, **kwargs):  # type: ignore
        raise ImportError("google-adk not installed")

try:
    from .providers.claude_agent_sdk import (
        AnthropicAgentsInstrumentor,  # Backwards compat alias
        ClaudeAgentSDKInstrumentor,
        EvalynAgentHooks,
        MessageStreamAdapter,
        SubagentContext,
        create_agent_hooks,
        create_stream_adapter,
    )

    _has_claude_agent_sdk = True
except ImportError:
    _has_claude_agent_sdk = False

    def create_agent_hooks(*args, **kwargs):  # type: ignore
        raise ImportError("claude_agent_sdk not installed")

    def create_stream_adapter(*args, **kwargs):  # type: ignore
        raise ImportError("claude_agent_sdk not installed")


def _setup_registry() -> None:
    """Register all instrumentors with the global registry."""
    registry = get_registry()

    # Core instrumentors (monkey-patch)
    registry.register(OpenAIInstrumentor())
    registry.register(AnthropicInstrumentor())
    registry.register(GeminiInstrumentor())
    registry.register(LangChainInstrumentor())
    registry.register(LangGraphInstrumentor())

    # Optional OTEL-native instrumentors
    if _has_google_adk:
        registry.register(GoogleADKInstrumentor())

    # Optional hook-based instrumentors
    if _has_claude_agent_sdk:
        registry.register(ClaudeAgentSDKInstrumentor())


# Initialize registry on import
_setup_registry()


# Public API
def instrument(*names: str) -> Dict[str, bool]:
    """
    Instrument specific SDKs by name.

    Args:
        *names: Names of SDKs to instrument (e.g., "openai", "google_adk")

    Returns:
        Dict mapping name to success status

    Example:
        instrument("openai", "anthropic")
    """
    registry = get_registry()
    return registry.instrument(*names)


def instrument_all() -> Dict[str, bool]:
    """
    Instrument all available SDKs.

    Returns:
        Dict mapping name to success status
    """
    registry = get_registry()
    return registry.instrument_all()


def uninstrument(*names: str) -> Dict[str, bool]:
    """
    Remove instrumentation from specific SDKs.

    Args:
        *names: Names of SDKs to uninstrument

    Returns:
        Dict mapping name to success status
    """
    registry = get_registry()
    return registry.uninstrument(*names)


def uninstrument_all() -> Dict[str, bool]:
    """Remove instrumentation from all SDKs."""
    registry = get_registry()
    return registry.uninstrument_all()


def is_instrumented(name: str) -> bool:
    """Check if a specific SDK is instrumented."""
    registry = get_registry()
    return registry.is_instrumented(name)


def list_available() -> List[str]:
    """List all available (installed) SDKs that can be instrumented."""
    registry = get_registry()
    return registry.list_available()


def list_instrumented() -> List[str]:
    """List currently instrumented SDKs."""
    registry = get_registry()
    return registry.list_instrumented()


def get_hooks(name: str) -> Optional[Any]:
    """
    Get hook adapter for a hook-based instrumentor.

    Args:
        name: Instrumentor name (e.g., "anthropic_agents")

    Returns:
        Hook object to pass to the agent, or None if not applicable
    """
    registry = get_registry()
    return registry.get_hooks(name)


__all__ = [
    # Base classes
    "Instrumentor",
    "InstrumentorType",
    "InstrumentorRegistry",
    "get_registry",
    # Span handling
    "SpanConverter",
    "EvalynSpanProcessor",
    "create_evalyn_tracer_provider",
    # Public API
    "instrument",
    "instrument_all",
    "uninstrument",
    "uninstrument_all",
    "is_instrumented",
    "list_available",
    "list_instrumented",
    "get_hooks",
    # Claude Agent SDK hooks
    "create_agent_hooks",
    "create_stream_adapter",
    "EvalynAgentHooks",
    "MessageStreamAdapter",
    "SubagentContext",
    "ClaudeAgentSDKInstrumentor",
    "AnthropicAgentsInstrumentor",  # Backwards compat alias
    # Google ADK hooks
    "create_adk_callbacks",
    "create_adk_stream_adapter",
    "EvalynADKCallbacks",
    "ADKStreamAdapter",
    "GoogleADKInstrumentor",
    # Instrumentors
    "OpenAIInstrumentor",
    "AnthropicInstrumentor",
    "GeminiInstrumentor",
    "LangChainInstrumentor",
    "LangGraphInstrumentor",
]
