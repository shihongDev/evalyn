"""CLI output pagination: built-in pager for long terminal outputs.

Pure Python, no external dependencies. Provides page splitting logic,
terminal height detection, and pager command resolution.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class PagerConfig:
    """Configuration for output pagination.

    Attributes:
        enabled: Whether pagination is active.
        page_size: Lines per page. 0 means auto-detect from terminal height.
        pager_command: External pager command. Empty string means use built-in
            or respect PAGER env var.
    """

    enabled: bool = True
    page_size: int = 0
    pager_command: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "page_size": self.page_size,
            "pager_command": self.pager_command,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PagerConfig:
        return cls(
            enabled=data.get("enabled", True),
            page_size=data.get("page_size", 0),
            pager_command=data.get("pager_command", ""),
        )


def get_terminal_height() -> int:
    """Return the terminal height in lines.

    Tries os.get_terminal_size().lines, falls back to 24.
    """
    try:
        return os.get_terminal_size().lines
    except (ValueError, OSError):
        return 24


def _effective_page_size(config: PagerConfig) -> int:
    """Resolve page size: use config value or auto-detect from terminal."""
    if config.page_size > 0:
        return config.page_size
    # Reserve one line for the footer
    height = get_terminal_height()
    return max(1, height - 1)


def should_paginate(text: str, config: PagerConfig) -> bool:
    """Return True if the text exceeds the page size and paging is enabled."""
    if not config.enabled:
        return False
    page_size = _effective_page_size(config)
    line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    return line_count > page_size


def paginate_builtin(text: str, page_size: int) -> list[str]:
    """Split text into pages of page_size lines each.

    Returns a list of page strings. Each page contains up to page_size lines,
    preserving original line endings.
    """
    if page_size < 1:
        page_size = 1
    lines = text.splitlines(keepends=True)
    # If text is empty, return one empty page
    if not lines:
        return [""]
    pages: list[str] = []
    for i in range(0, len(lines), page_size):
        pages.append("".join(lines[i : i + page_size]))
    return pages


def format_page(page: str, page_num: int, total_pages: int) -> str:
    """Append a page footer to the given page text.

    Footer format: "-- Page N/M --"
    """
    footer = f"-- Page {page_num}/{total_pages} --"
    # Ensure there is a newline before the footer
    if page and not page.endswith("\n"):
        return page + "\n" + footer
    return page + footer


def get_pager_command(config: PagerConfig) -> str:
    """Determine the pager command to use.

    Priority:
    1. config.pager_command (if non-empty)
    2. PAGER environment variable (if set)
    3. "less" on Unix, "more" on Windows
    """
    if config.pager_command:
        return config.pager_command
    env_pager = os.environ.get("PAGER", "")
    if env_pager:
        return env_pager
    if sys.platform == "win32":
        return "more"
    return "less"


def paginate_text(text: str, config: PagerConfig | None = None) -> str:
    """Paginate text if it exceeds the page size.

    If pagination is needed, returns the first page with a footer.
    Otherwise returns the text unchanged.

    Note: actual interactive paging requires a terminal - this module
    provides the page splitting logic.
    """
    if config is None:
        config = PagerConfig()
    if not should_paginate(text, config):
        return text
    page_size = _effective_page_size(config)
    pages = paginate_builtin(text, page_size)
    total = len(pages)
    return format_page(pages[0], 1, total)


def create_page_iterator(text: str, page_size: int) -> list[tuple[int, str]]:
    """Return a list of (page_num, page_text) tuples for iteration.

    Page numbers are 1-based.
    """
    if page_size < 1:
        page_size = 1
    pages = paginate_builtin(text, page_size)
    return [(i + 1, page) for i, page in enumerate(pages)]
