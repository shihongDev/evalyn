"""
Instrumentation toggle API: hot-toggle instrumentation on/off at runtime.

Provides global state and a context manager for pausing/resuming
instrumentation without restarting the process.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List

_INSTRUMENTATION_ENABLED: bool = True
_TOGGLE_HISTORY: List[Dict[str, Any]] = []


def toggle_instrumentation(enabled: bool = True) -> bool:
    """Set the instrumentation enabled state.

    Returns the previous state. Records the change in history.
    """
    global _INSTRUMENTATION_ENABLED
    previous = _INSTRUMENTATION_ENABLED
    _INSTRUMENTATION_ENABLED = enabled
    _TOGGLE_HISTORY.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "enabled": enabled,
            "previous": previous,
        }
    )
    return previous


def is_instrumentation_enabled() -> bool:
    """Check whether instrumentation is currently enabled."""
    return _INSTRUMENTATION_ENABLED


def pause_instrumentation() -> bool:
    """Shortcut to disable instrumentation. Returns previous state."""
    return toggle_instrumentation(False)


def resume_instrumentation() -> bool:
    """Shortcut to enable instrumentation. Returns previous state."""
    return toggle_instrumentation(True)


def get_toggle_history() -> List[Dict[str, Any]]:
    """Return the history of toggle operations.

    Each entry is {"timestamp": str, "enabled": bool, "previous": bool}.
    """
    return list(_TOGGLE_HISTORY)


def clear_toggle_history() -> None:
    """Clear the toggle history."""
    _TOGGLE_HISTORY.clear()


def reset_instrumentation() -> None:
    """Reset to default state (enabled=True) and clear history."""
    global _INSTRUMENTATION_ENABLED
    _INSTRUMENTATION_ENABLED = True
    _TOGGLE_HISTORY.clear()


@contextmanager
def instrumentation_paused() -> Generator[None, None, None]:
    """Context manager that pauses instrumentation on enter and restores on exit."""
    previous = pause_instrumentation()
    try:
        yield
    finally:
        toggle_instrumentation(previous)
