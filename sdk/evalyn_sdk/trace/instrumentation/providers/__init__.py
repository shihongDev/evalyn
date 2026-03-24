"""
Instrumentation providers for various LLM SDKs and agent frameworks.

Each provider implements the Instrumentor protocol for a specific SDK.
"""

from __future__ import annotations

from .openai import OpenAIInstrumentor
from .anthropic import AnthropicInstrumentor
from .gemini import GeminiInstrumentor
from .langchain import LangChainInstrumentor
from .langgraph import LangGraphInstrumentor
from .crewai import CrewAIInstrumentor
from .autogen import AutoGenInstrumentor
from .dspy import DSPyInstrumentor
from .haystack import HaystackInstrumentor
from .llamaindex import LlamaIndexInstrumentor
from .semantic_kernel import SemanticKernelInstrumentor

# These are imported conditionally to avoid import errors if deps not installed
try:
    from .google_adk import GoogleADKInstrumentor
except ImportError:
    GoogleADKInstrumentor = None  # type: ignore

try:
    from .claude_agent_sdk import ClaudeAgentSDKInstrumentor
except ImportError:
    ClaudeAgentSDKInstrumentor = None  # type: ignore


__all__ = [
    "OpenAIInstrumentor",
    "AnthropicInstrumentor",
    "GeminiInstrumentor",
    "LangChainInstrumentor",
    "LangGraphInstrumentor",
    "CrewAIInstrumentor",
    "AutoGenInstrumentor",
    "DSPyInstrumentor",
    "HaystackInstrumentor",
    "LlamaIndexInstrumentor",
    "SemanticKernelInstrumentor",
    "GoogleADKInstrumentor",
    "ClaudeAgentSDKInstrumentor",
]
