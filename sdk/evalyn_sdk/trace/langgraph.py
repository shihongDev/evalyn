"""
LangGraph auto-instrumentation for hierarchical span tracing.

This module is kept for backward compatibility.
The actual implementation is now in evalyn_sdk.trace.instrumentation.providers.langgraph.

Usage:
    # Auto-enabled when evalyn_sdk is imported
    # Or manually:
    from evalyn_sdk.trace.langgraph import patch_langgraph
    patch_langgraph()
"""

from __future__ import annotations

from .instrumentation.providers.langgraph import LangGraphInstrumentor

# Singleton instrumentor for backward compatibility
_instrumentor = LangGraphInstrumentor()


def _do_patch_langgraph() -> bool:
    """
    Internal: Patch LangGraph to emit spans for graph and node execution.

    Called by auto_instrument.py which manages the patched state.
    Returns True if patching succeeded, False if LangGraph is not installed.

    Note: This function is deprecated. Use LangGraphInstrumentor.instrument() instead.
    """
    return _instrumentor.instrument()
