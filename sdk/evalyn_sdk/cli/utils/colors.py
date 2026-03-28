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
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    return True


# ANSI escape codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"

_BG_RED = "\033[41m"
_BG_GREEN = "\033[42m"
_BG_YELLOW = "\033[43m"


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


def cyan(text: str) -> str:
    """Cyan text (secondary info)."""
    return _wrap(text, _CYAN)


def dim(text: str) -> str:
    """Dimmed text (less important)."""
    return _wrap(text, _DIM)


def bold(text: str) -> str:
    """Bold text (emphasis)."""
    return _wrap(text, _BOLD)


def pass_fail(passed: bool) -> str:
    """Color-coded PASS/FAIL label."""
    if passed:
        return green("PASS")
    return red("FAIL")


def score_color(score: float) -> str:
    """Color-code a numeric score (0.0-1.0).

    Green >= 0.8, yellow >= 0.5, red < 0.5.
    """
    text = f"{score:.2f}"
    if score >= 0.8:
        return green(text)
    if score >= 0.5:
        return yellow(text)
    return red(text)


def pass_rate_color(rate: float) -> str:
    """Color-code a pass rate (0.0-1.0) as percentage.

    Green >= 80%, yellow >= 50%, red < 50%.
    """
    pct = f"{rate * 100:.1f}%"
    if rate >= 0.8:
        return green(pct)
    if rate >= 0.5:
        return yellow(pct)
    return red(pct)


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


def status_badge(status: str) -> str:
    """Color-coded status badge."""
    s = status.upper()
    if s in ("PASS", "SUCCESS", "OK", "GOOD"):
        return green(f"[{s}]")
    if s in ("FAIL", "ERROR", "CRITICAL"):
        return red(f"[{s}]")
    if s in ("WARN", "WARNING", "MODERATE", "NEEDS_ATTENTION"):
        return yellow(f"[{s}]")
    return f"[{s}]"
