"""CLI hint utilities for guiding users to next steps."""

from __future__ import annotations

import os


def print_hint(message: str, quiet: bool = False, format: str = "table") -> None:
    """Print a hint message to guide users to the next step.

    Args:
        message: The hint message to display
        quiet: If True, suppress the hint
        format: Output format - hints are suppressed for 'json'

    Environment:
        EVALYN_NO_HINTS: Set to '1' or 'true' to suppress all hints globally
    """
    # Check global quiet setting from environment
    env_quiet = os.environ.get("EVALYN_NO_HINTS", "").lower() in ("1", "true")
    if quiet or env_quiet or format == "json":
        return
    print(f"\nHint: {message}")
