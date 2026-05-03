"""CLI hint utilities for guiding users to next steps."""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple


def _is_suppressed(quiet: bool = False, format: str = "table") -> bool:
    """Check whether hints should be suppressed."""
    env_quiet = os.environ.get("EVALYN_NO_HINTS", "").lower() in ("1", "true")
    return quiet or env_quiet or format == "json"


# Type alias: each option is (flag_text, description)
HintOption = Tuple[str, str]


class HintCollector:
    """Collects hints during command execution, renders them as a block."""

    def __init__(self, quiet: bool = False, format: str = "table"):
        self._hints: list[tuple[str, str, list[HintOption]]] = []
        self._quiet = quiet
        self._format = format

    @property
    def has_hints(self) -> bool:
        """Return True if any hints have been collected."""
        return bool(self._hints)

    def add(
        self,
        command: str,
        description: str,
        options: Optional[Sequence[HintOption]] = None,
    ) -> None:
        """Add a hint with a command, description, and optional key flags."""
        self._hints.append((command, description, list(options or [])))

    def render(self, max_hints: int = 3) -> None:
        """Print collected hints as a formatted block."""
        if _is_suppressed(self._quiet, self._format) or not self._hints:
            return

        to_show = self._hints[:max_hints]

        # Compute column width for alignment
        max_cmd_len = max(len(cmd) for cmd, _, _ in to_show)
        pad = max_cmd_len + 4  # 4 chars gap between command and description

        print("\nNext steps:")
        for cmd, desc, opts in to_show:
            print(f"  {cmd:<{pad}}{desc}")
            for flag, flag_desc in opts:
                print(f"    {flag:<{pad}}{flag_desc}")
