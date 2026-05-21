"""ANSI color utilities for terminal output.

Respects NO_COLOR env var (https://no-color.org/) and --no-color flag.
Falls back to no-op when colors are disabled or output is not a terminal.
"""

from __future__ import annotations

import os
import sys


def _colors_enabled() -> bool:
    """Check if ANSI colors should be used."""
    # NO_COLOR env var (standard: https://no-color.org/)
    if os.environ.get("NO_COLOR") is not None:
        return False
    # EVALYN_NO_COLOR for evalyn-specific override
    if os.environ.get("EVALYN_NO_COLOR") is not None:
        return False
    # Check if stdout is a terminal (not piped)
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# ANSI escape codes
_RESET = "\033[0m"
_DIM = "\033[2m"

_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"


def _wrap(text: str, code: str) -> str:
    """Wrap text with ANSI code if colors enabled."""
    if not _colors_enabled():
        return text
    return f"{code}{text}{_RESET}"


def green(text: str) -> str:
    """Green text (pass, success)."""
    return _wrap(text, _GREEN)


def red(text: str) -> str:
    """Red text (fail, error)."""
    return _wrap(text, _RED)


def yellow(text: str) -> str:
    """Yellow text (warning, partial)."""
    return _wrap(text, _YELLOW)


def blue(text: str) -> str:
    """Blue text (info, headers)."""
    return _wrap(text, _BLUE)


def dim(text: str) -> str:
    """Dimmed text (less important)."""
    return _wrap(text, _DIM)


def delta_color(delta: float) -> str:
    """Color-code a delta value.

    Green for positive (improvement), red for negative (regression),
    yellow for zero/near-zero.
    """
    sign = "+" if delta >= 0 else ""
    text = f"{sign}{delta * 100:.1f}%"
    if delta > 0.01:
        return green(text)
    if delta < -0.01:
        return red(text)
    return yellow(text)
