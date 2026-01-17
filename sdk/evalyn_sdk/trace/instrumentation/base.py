"""
Base protocol for SDK instrumentors.

Instrumentors wrap LLM SDKs to capture spans automatically.
Three instrumentation patterns are supported:

1. MONKEY_PATCH: Wrap SDK methods directly (OpenAI, Anthropic client, Gemini)
2. OTEL_NATIVE: Use SDK's built-in OTEL support with custom SpanProcessor (Google ADK)
3. HOOK_BASED: Use SDK's hook/callback system (Anthropic Agent SDK)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional


class InstrumentorType(Enum):
    """Types of instrumentation strategies."""

    MONKEY_PATCH = "monkey_patch"  # Wrap SDK methods directly
    OTEL_NATIVE = "otel_native"  # Use OTEL SpanProcessor
    HOOK_BASED = "hook_based"  # Use SDK's hook/callback system


class Instrumentor(ABC):
    """
    Base class for SDK instrumentors.

    Each instrumentor is responsible for:
    - Detecting if its SDK is available
    - Instrumenting the SDK to capture spans
    - Providing cleanup/uninstrumentation if needed
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this instrumentor (e.g., 'openai', 'google_adk')."""
        ...

    @property
    @abstractmethod
    def instrumentor_type(self) -> InstrumentorType:
        """The instrumentation strategy used by this instrumentor."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the SDK is installed and available for instrumentation."""
        ...

    @abstractmethod
    def instrument(self) -> bool:
        """
        Apply instrumentation to the SDK.

        Returns True if instrumentation was successful, False otherwise.
        Should be idempotent: calling multiple times has no additional effect.
        """
        ...

    @abstractmethod
    def uninstrument(self) -> bool:
        """
        Remove instrumentation from the SDK.

        Returns True if uninstrumentation was successful, False otherwise.
        Should be idempotent.
        """
        ...

    @abstractmethod
    def is_instrumented(self) -> bool:
        """Check if the SDK is currently instrumented."""
        ...

    def get_hooks(self) -> Optional[Any]:
        """
        Get hook adapter for hook-based instrumentors.

        Only applicable for HOOK_BASED instrumentors.
        Returns the hook object that users should pass to their agent.
        """
        return None
