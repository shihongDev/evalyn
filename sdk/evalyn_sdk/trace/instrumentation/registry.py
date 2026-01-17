"""
Instrumentor registry for managing SDK instrumentors.

Provides a central place to register, discover, and invoke instrumentors.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base import Instrumentor, InstrumentorType


class InstrumentorRegistry:
    """
    Singleton registry for SDK instrumentors.

    Usage:
        registry = InstrumentorRegistry.get()
        registry.register(OpenAIInstrumentor())
        registry.instrument_all()  # Instrument all available SDKs
        registry.instrument("openai")  # Instrument specific SDK
    """

    _instance: Optional["InstrumentorRegistry"] = None

    def __init__(self):
        self._instrumentors: Dict[str, Instrumentor] = {}
        self._auto_instrumented = False

    @classmethod
    def get(cls) -> "InstrumentorRegistry":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. Useful for testing."""
        cls._instance = None

    def register(self, instrumentor: Instrumentor) -> None:
        """Register an instrumentor."""
        self._instrumentors[instrumentor.name] = instrumentor

    def get_instrumentor(self, name: str) -> Optional[Instrumentor]:
        """Get an instrumentor by name."""
        return self._instrumentors.get(name)

    def list_instrumentors(self) -> List[str]:
        """List all registered instrumentor names."""
        return list(self._instrumentors.keys())

    def list_available(self) -> List[str]:
        """List instrumentors whose SDKs are available."""
        return [
            name for name, inst in self._instrumentors.items() if inst.is_available()
        ]

    def list_instrumented(self) -> List[str]:
        """List currently instrumented SDKs."""
        return [
            name for name, inst in self._instrumentors.items() if inst.is_instrumented()
        ]

    def is_instrumented(self, name: str) -> bool:
        """Check if a specific SDK is instrumented."""
        inst = self._instrumentors.get(name)
        return inst.is_instrumented() if inst else False

    def instrument(self, *names: str) -> Dict[str, bool]:
        """
        Instrument specific SDKs by name.

        Args:
            *names: Names of instrumentors to activate

        Returns:
            Dict mapping name to success status
        """
        results = {}
        for name in names:
            inst = self._instrumentors.get(name)
            if inst and inst.is_available():
                results[name] = inst.instrument()
            else:
                results[name] = False
        return results

    def instrument_all(self) -> Dict[str, bool]:
        """
        Instrument all available SDKs.

        Returns:
            Dict mapping name to success status
        """
        results = {}
        for name, inst in self._instrumentors.items():
            if inst.is_available():
                results[name] = inst.instrument()
            else:
                results[name] = False
        return results

    def uninstrument(self, *names: str) -> Dict[str, bool]:
        """Uninstrument specific SDKs."""
        results = {}
        for name in names:
            inst = self._instrumentors.get(name)
            if inst:
                results[name] = inst.uninstrument()
            else:
                results[name] = False
        return results

    def uninstrument_all(self) -> Dict[str, bool]:
        """Uninstrument all SDKs."""
        results = {}
        for name, inst in self._instrumentors.items():
            results[name] = inst.uninstrument()
        return results

    def ensure_instrumented(self) -> Dict[str, bool]:
        """
        Lazily instrument all SDKs on first trace.

        Called automatically by the tracer. Respects EVALYN_AUTO_INSTRUMENT env var.
        """
        if self._auto_instrumented:
            return {
                name: inst.is_instrumented()
                for name, inst in self._instrumentors.items()
            }

        self._auto_instrumented = True

        if os.environ.get("EVALYN_AUTO_INSTRUMENT", "").lower() in (
            "off",
            "false",
            "0",
            "no",
        ):
            return {}

        return self.instrument_all()

    def get_hooks(self, name: str) -> Optional[Any]:
        """
        Get hook adapter for a hook-based instrumentor.

        Args:
            name: Instrumentor name (e.g., 'anthropic_agents')

        Returns:
            Hook object to pass to the agent, or None if not applicable
        """
        inst = self._instrumentors.get(name)
        if inst and inst.instrumentor_type == InstrumentorType.HOOK_BASED:
            return inst.get_hooks()
        return None


def get_registry() -> InstrumentorRegistry:
    """Get the global instrumentor registry."""
    return InstrumentorRegistry.get()
