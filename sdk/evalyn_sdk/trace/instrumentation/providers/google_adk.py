"""
Google ADK (Agent Development Kit) instrumentor.

Uses OpenInference instrumentation for Google ADK with EvalynSpanProcessor.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Optional

from ..base import Instrumentor, InstrumentorType
from ..span_processor import get_or_create_tracer_provider


class GoogleADKInstrumentor(Instrumentor):
    """
    Instrumentor for Google ADK (Agent Development Kit).

    Uses the OTEL-native approach with openinference-instrumentation-google-adk.
    The EvalynSpanProcessor captures OTEL spans and converts them to Evalyn spans.
    """

    _instrumented = False
    _otel_instrumentor: Optional[Any] = None

    @property
    def name(self) -> str:
        return "google_adk"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.OTEL_NATIVE

    def is_available(self) -> bool:
        # Check for google-adk (OTEL is always available as required dep)
        try:
            return importlib.util.find_spec("google.adk") is not None
        except (ModuleNotFoundError, ImportError):
            return False

    def is_instrumented(self) -> bool:
        return self._instrumented

    def instrument(self) -> bool:
        if self._instrumented:
            return True

        if not self.is_available():
            return False

        try:
            # Set up Evalyn's TracerProvider first
            get_or_create_tracer_provider()

            # Try to use openinference instrumentation if available
            try:
                from openinference.instrumentation.google_adk import (
                    GoogleADKInstrumentor as OIInstrumentor,
                )

                self._otel_instrumentor = OIInstrumentor()
                self._otel_instrumentor.instrument()
            except ImportError:
                # Fall back to manual OTEL instrumentation
                self._instrument_manually()

            self._instrumented = True
            return True

        except Exception:
            return False

    def _instrument_manually(self) -> None:
        """
        Manual instrumentation when openinference is not available.

        Patches google.adk to emit OTEL spans directly.
        """
        try:
            from opentelemetry import trace

            tracer = trace.get_tracer("evalyn.google_adk")

            # Import and patch google.adk Agent class
            from google.adk import Agent

            original_run = Agent.run

            def patched_run(self, *args, **kwargs):
                with tracer.start_as_current_span(
                    f"agent:{getattr(self, 'name', 'agent')}",
                    attributes={
                        "openinference.span.kind": "AGENT",
                        "agent.name": getattr(self, "name", "agent"),
                    },
                ):
                    return original_run(self, *args, **kwargs)

            Agent.run = patched_run

            # Store for uninstrument
            self._original_run = original_run

        except ImportError:
            pass

    def uninstrument(self) -> bool:
        if not self._instrumented:
            return True

        try:
            if self._otel_instrumentor:
                self._otel_instrumentor.uninstrument()
                self._otel_instrumentor = None
            elif hasattr(self, "_original_run"):
                from google.adk import Agent

                Agent.run = self._original_run
                del self._original_run

            self._instrumented = False
            return True
        except Exception:
            return False
