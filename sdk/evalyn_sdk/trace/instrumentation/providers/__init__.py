"""
Instrumentation providers for various LLM SDKs.

Each provider implements the Instrumentor protocol for a specific SDK.
"""

from __future__ import annotations

from .openai import OpenAIInstrumentor
from .anthropic import AnthropicInstrumentor
from .gemini import GeminiInstrumentor
from .langchain import LangChainInstrumentor
from .langgraph import LangGraphInstrumentor

# These are imported conditionally to avoid import errors if deps not installed
try:
    from .google_adk import GoogleADKInstrumentor
except ImportError:
    GoogleADKInstrumentor = None  # type: ignore

try:
    from .anthropic_agents import AnthropicAgentsInstrumentor
except ImportError:
    AnthropicAgentsInstrumentor = None  # type: ignore


__all__ = [
    "OpenAIInstrumentor",
    "AnthropicInstrumentor",
    "GeminiInstrumentor",
    "LangChainInstrumentor",
    "LangGraphInstrumentor",
    "GoogleADKInstrumentor",
    "AnthropicAgentsInstrumentor",
]
