"""CLI output width control: respect terminal width for tables and charts."""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class WidthConfig:
    """Configuration for terminal output width.

    max_width of 0 means auto-detect from the terminal.
    """

    max_width: int = 0
    min_width: int = 40
    default_width: int = 80

    def as_dict(self) -> dict[str, object]:
        return {
            "max_width": self.max_width,
            "min_width": self.min_width,
            "default_width": self.default_width,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> WidthConfig:
        return cls(
            max_width=int(data.get("max_width", 0)),
            min_width=int(data.get("min_width", 40)),
            default_width=int(data.get("default_width", 80)),
        )


def get_terminal_width() -> int:
    """Detect terminal width, falling back to 80 columns."""
    try:
        return os.get_terminal_size().columns
    except (OSError, ValueError):
        return 80


def get_effective_width(config: WidthConfig | None = None) -> int:
    """Return the effective output width, clamped to min_width.

    If config.max_width is 0 (or config is None), auto-detect from terminal.
    """
    if config is None:
        config = WidthConfig()

    if config.max_width > 0:
        width = config.max_width
    else:
        width = get_terminal_width()

    return max(width, config.min_width)


def truncate_table_row(
    columns: list[str], widths: list[int], separator: str = " | "
) -> str:
    """Truncate each column to its designated width and join with separator.

    Columns wider than their width get an ellipsis suffix.
    """
    parts: list[str] = []
    for col, w in zip(columns, widths):
        if w < 1:
            w = 1
        if len(col) > w:
            if w >= 4:
                truncated = col[: w - 3] + "..."
            else:
                truncated = col[:w]
            parts.append(truncated)
        else:
            parts.append(col.ljust(w))
    return separator.join(parts)


def compute_column_widths(
    headers: list[str],
    rows: list[list[str]],
    max_width: int,
    separator: str = " | ",
) -> list[int]:
    """Distribute available width proportionally based on content.

    Each column gets at least 5 characters.
    """
    if not headers:
        return []

    n_cols = len(headers)
    min_per_col = 5

    # Compute natural (max content) width for each column
    natural: list[int] = []
    for i in range(n_cols):
        col_max = len(headers[i])
        for row in rows:
            if i < len(row):
                col_max = max(col_max, len(row[i]))
        natural.append(col_max)

    # Available width after separators
    sep_total = len(separator) * (n_cols - 1) if n_cols > 1 else 0
    available = max_width - sep_total

    # Ensure at least min_per_col * n_cols
    min_total = min_per_col * n_cols
    if available < min_total:
        available = min_total

    total_natural = sum(natural)
    if total_natural == 0:
        # All empty columns - split evenly
        base = available // n_cols
        widths = [base] * n_cols
        # Distribute remainder
        remainder = available - base * n_cols
        for i in range(remainder):
            widths[i] += 1
        return widths

    if total_natural <= available:
        # Everything fits
        return natural

    # Distribute proportionally, ensuring minimum
    widths: list[int] = []
    for nat in natural:
        w = max(min_per_col, int(available * nat / total_natural))
        widths.append(w)

    # Adjust to match available exactly
    diff = available - sum(widths)
    # Distribute diff to largest columns first
    if diff > 0:
        indices = sorted(range(n_cols), key=lambda i: -natural[i])
        for i in indices:
            if diff <= 0:
                break
            widths[i] += 1
            diff -= 1
    elif diff < 0:
        indices = sorted(range(n_cols), key=lambda i: -widths[i])
        for i in indices:
            if diff >= 0:
                break
            if widths[i] > min_per_col:
                widths[i] -= 1
                diff += 1

    return widths


def format_table(
    headers: list[str],
    rows: list[list[str]],
    config: WidthConfig | None = None,
) -> str:
    """Format a full table respecting terminal width.

    Returns header row, separator line, and data rows.
    """
    if not headers:
        return ""

    width = get_effective_width(config)
    separator = " | "
    col_widths = compute_column_widths(headers, rows, width, separator)

    lines: list[str] = []
    # Header
    lines.append(truncate_table_row(headers, col_widths, separator))
    # Separator line
    sep_parts = ["-" * w for w in col_widths]
    lines.append(separator.join(sep_parts))
    # Data rows
    for row in rows:
        # Pad row to match headers length
        padded = row + [""] * (len(headers) - len(row))
        lines.append(truncate_table_row(padded, col_widths, separator))

    return "\n".join(lines)


def wrap_paragraph(text: str, width: int | None = None) -> str:
    """Word-wrap text to the given or terminal width."""
    if width is None:
        width = get_effective_width()
    return textwrap.fill(text, width=width)


def format_progress_bar(
    progress: float, width: int | None = None, label: str = ""
) -> str:
    """Render a progress bar fitting terminal width.

    Format: [=====>     ] 50% label
    """
    if width is None:
        width = get_effective_width()

    progress = max(0.0, min(1.0, progress))
    pct_str = f" {int(progress * 100)}%"
    label_str = f" {label}" if label else ""
    suffix = pct_str + label_str

    # Brackets take 2 chars, suffix takes len(suffix)
    bar_width = width - 2 - len(suffix)
    if bar_width < 1:
        bar_width = 1

    filled = int(bar_width * progress)
    # Place arrow head at the end of filled portion (if not 100%)
    if progress < 1.0 and filled > 0:
        bar = "=" * (filled - 1) + ">" + " " * (bar_width - filled)
    elif progress >= 1.0:
        bar = "=" * bar_width
    else:
        bar = " " * bar_width

    return f"[{bar}]{suffix}"
