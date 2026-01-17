"""Error handling utilities for CLI commands."""

from __future__ import annotations

import sys
from typing import NoReturn


def fatal_error(message: str, hint: str = "") -> NoReturn:
    """Print error message and exit with code 1.

    Args:
        message: Error message to display
        hint: Optional hint for how to fix the error

    Example:
        fatal_error("No dataset specified", "Use --dataset <path> or --latest")
    """
    print(f"Error: {message}")
    if hint:
        print(f"Hint: {hint}")
    sys.exit(1)


def warning(message: str) -> None:
    """Print warning message to stderr."""
    print(f"Warning: {message}", file=sys.stderr)


__all__ = ["fatal_error", "warning"]
