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

import importlib
from typing import Any

from .base import Instrumentor, InstrumentorType
from .registry import InstrumentorRegistry, get_registry
from .span_converter import SpanConverter
from .span_processor import EvalynSpanProcessor, create_evalyn_tracer_provider

# ---------------------------------------------------------------------------
# Lazy provider registration: providers are imported and registered on first
# use (instrument() or ensure_instrumented()) rather than at module import.
# This saves ~300ms by deferring the import of 14 provider modules.
# ---------------------------------------------------------------------------

_registry_initialized = False


def _setup_registry() -> None:
    """Register all instrumentors with the global registry (called lazily)."""
    global _registry_initialized
    if _registry_initialized:
        return
    _registry_initialized = True

    registry = get_registry()

    # Core instrumentors (monkey-patch) - imported here, not at module level
    from .providers.anthropic import AnthropicInstrumentor
    from .providers.autogen import AutoGenInstrumentor
    from .providers.crewai import CrewAIInstrumentor
    from .providers.dspy import DSPyInstrumentor
    from .providers.gemini import GeminiInstrumentor
    from .providers.haystack import HaystackInstrumentor
    from .providers.langchain import LangChainInstrumentor
    from .providers.langgraph import LangGraphInstrumentor
    from .providers.llamaindex import LlamaIndexInstrumentor
    from .providers.openai import OpenAIInstrumentor
    from .providers.semantic_kernel import SemanticKernelInstrumentor
    from .providers.xai import XAIInstrumentor

    registry.register(OpenAIInstrumentor())
    registry.register(AnthropicInstrumentor())
    registry.register(GeminiInstrumentor())
    registry.register(LangChainInstrumentor())
    registry.register(LangGraphInstrumentor())
    registry.register(XAIInstrumentor())
    registry.register(CrewAIInstrumentor())
    registry.register(AutoGenInstrumentor())
    registry.register(DSPyInstrumentor())
    registry.register(HaystackInstrumentor())
    registry.register(LlamaIndexInstrumentor())
    registry.register(SemanticKernelInstrumentor())

    # Optional OTEL-native instrumentors
    try:
        from .providers.google_adk import GoogleADKInstrumentor

        registry.register(GoogleADKInstrumentor())
    except ImportError:
        pass

    # Optional hook-based instrumentors
    try:
        from .providers.claude_agent_sdk import ClaudeAgentSDKInstrumentor

        registry.register(ClaudeAgentSDKInstrumentor())
    except ImportError:
        pass


def create_adk_callbacks(*args, **kwargs):
    """Create Google ADK callbacks (imports on first call)."""
    from .providers.google_adk import create_adk_callbacks as _create

    return _create(*args, **kwargs)


def create_adk_stream_adapter(*args, **kwargs):
    """Create ADK stream adapter (imports on first call)."""
    from .providers.google_adk import create_stream_adapter as _create

    return _create(*args, **kwargs)


def create_agent_hooks(*args, **kwargs):
    """Create Claude Agent SDK hooks (imports on first call)."""
    from .providers.claude_agent_sdk import create_agent_hooks as _create

    return _create(*args, **kwargs)


def create_stream_adapter(*args, **kwargs):
    """Create Claude Agent SDK stream adapter (imports on first call)."""
    from .providers.claude_agent_sdk import create_stream_adapter as _create

    return _create(*args, **kwargs)


# Public API
def instrument(*names: str) -> dict[str, bool]:
    """
    Instrument specific SDKs by name.

    Args:
        *names: Names of SDKs to instrument (e.g., "openai", "google_adk")

    Returns:
        Dict mapping name to success status

    Example:
        instrument("openai", "anthropic")
    """
    _setup_registry()
    registry = get_registry()
    return registry.instrument(*names)


def instrument_all() -> dict[str, bool]:
    """
    Instrument all available SDKs.

    Returns:
        Dict mapping name to success status
    """
    _setup_registry()
    registry = get_registry()
    return registry.instrument_all()


def uninstrument(*names: str) -> dict[str, bool]:
    """
    Remove instrumentation from specific SDKs.

    Args:
        *names: Names of SDKs to uninstrument

    Returns:
        Dict mapping name to success status
    """
    _setup_registry()
    registry = get_registry()
    return registry.uninstrument(*names)


def uninstrument_all() -> dict[str, bool]:
    """Remove instrumentation from all SDKs."""
    _setup_registry()
    registry = get_registry()
    return registry.uninstrument_all()


def is_instrumented(name: str) -> bool:
    """Check if a specific SDK is instrumented."""
    _setup_registry()
    registry = get_registry()
    return registry.is_instrumented(name)


def list_available() -> list[str]:
    """List all available (installed) SDKs that can be instrumented."""
    _setup_registry()
    registry = get_registry()
    return registry.list_available()


def list_instrumented() -> list[str]:
    """List currently instrumented SDKs."""
    _setup_registry()
    registry = get_registry()
    return registry.list_instrumented()


def get_hooks(name: str) -> Any | None:
    """
    Get hook adapter for a hook-based instrumentor.

    Args:
        name: Instrumentor name (e.g., "anthropic_agents")

    Returns:
        Hook object to pass to the agent, or None if not applicable
    """
    _setup_registry()
    registry = get_registry()
    return registry.get_hooks(name)


# Lazy attribute access for provider classes (PEP 562)
_LAZY_PROVIDER_IMPORTS = {
    "OpenAIInstrumentor": ".providers.openai",
    "AnthropicInstrumentor": ".providers.anthropic",
    "GeminiInstrumentor": ".providers.gemini",
    "LangChainInstrumentor": ".providers.langchain",
    "LangGraphInstrumentor": ".providers.langgraph",
    "XAIInstrumentor": ".providers.xai",
    "CrewAIInstrumentor": ".providers.crewai",
    "AutoGenInstrumentor": ".providers.autogen",
    "DSPyInstrumentor": ".providers.dspy",
    "HaystackInstrumentor": ".providers.haystack",
    "LlamaIndexInstrumentor": ".providers.llamaindex",
    "SemanticKernelInstrumentor": ".providers.semantic_kernel",
    "GoogleADKInstrumentor": ".providers.google_adk",
    "EvalynADKCallbacks": ".providers.google_adk",
    "ADKStreamAdapter": ".providers.google_adk",
    "ClaudeAgentSDKInstrumentor": ".providers.claude_agent_sdk",
    "AnthropicAgentsInstrumentor": ".providers.claude_agent_sdk",
    "EvalynAgentHooks": ".providers.claude_agent_sdk",
    "MessageStreamAdapter": ".providers.claude_agent_sdk",
    "SubagentContext": ".providers.claude_agent_sdk",
}


def __getattr__(name: str):
    if name in _LAZY_PROVIDER_IMPORTS:
        module = importlib.import_module(_LAZY_PROVIDER_IMPORTS[name], __package__)
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "XAIInstrumentor",
    "CrewAIInstrumentor",
    "AutoGenInstrumentor",
    "DSPyInstrumentor",
    "HaystackInstrumentor",
    "LlamaIndexInstrumentor",
    "SemanticKernelInstrumentor",
]
