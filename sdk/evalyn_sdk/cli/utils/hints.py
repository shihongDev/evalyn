"""CLI hint utilities for guiding users to next steps."""

from __future__ import annotations


def print_hint(message: str, quiet: bool = False, format: str = "table") -> None:
    """Print a hint message to guide users to the next step.

    Args:
        message: The hint message to display
        quiet: If True, suppress the hint
        format: Output format - hints are suppressed for 'json'
    """
    if quiet or format == "json":
        return
    print(f"\nHint: {message}")
