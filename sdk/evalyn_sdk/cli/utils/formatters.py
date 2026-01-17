"""Output formatters for CLI commands."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional


def is_json_format(args) -> bool:
    """Check if output format is JSON."""
    return getattr(args, "format", "table") == "json"


def print_if_table(args, message: str, file=None) -> None:
    """Print message only if not JSON mode."""
    if not is_json_format(args):
        print(message, file=file or sys.stdout)


def print_error_if_table(args, message: str) -> None:
    """Print error message only if not JSON mode."""
    if not is_json_format(args):
        print(message, file=sys.stderr)


def output_json(data: Dict[str, Any]) -> None:
    """Output JSON data."""
    print(json.dumps(data, indent=2, default=str))


def print_table(
    headers: List[str],
    rows: List[List[str]],
    col_widths: Optional[List[int]] = None,
) -> None:
    """Print a simple table."""
    # Calculate column widths if not provided
    if not col_widths:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

    # Print header
    header_row = " | ".join(
        str(h).ljust(col_widths[i]) for i, h in enumerate(headers)
    )
    print(header_row)
    print("-" * len(header_row))

    # Print rows
    for row in rows:
        print(
            " | ".join(
                str(cell).ljust(col_widths[i]) if i < len(col_widths) else str(cell)
                for i, cell in enumerate(row)
            )
        )


__all__ = [
    "is_json_format",
    "print_if_table",
    "print_error_if_table",
    "output_json",
    "print_table",
]
