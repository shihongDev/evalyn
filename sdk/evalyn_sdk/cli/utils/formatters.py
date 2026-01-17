"""Output formatters for CLI commands."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional


class OutputFormatter:
    """Handle JSON vs table output formatting."""

    def __init__(self, format_type: str = "table"):
        self.format_type = format_type
        self._json_data: Dict[str, Any] = {}

    @property
    def is_json(self) -> bool:
        return self.format_type == "json"

    def print(self, message: str, file=None) -> None:
        """Print message only if not JSON mode."""
        if not self.is_json:
            print(message, file=file or sys.stdout)

    def print_error(self, message: str) -> None:
        """Print error message only if not JSON mode."""
        if not self.is_json:
            print(message, file=sys.stderr)

    def print_warning(self, message: str) -> None:
        """Print warning only if not JSON mode."""
        if not self.is_json:
            print(f"Warning: {message}", file=sys.stderr)

    def set_data(self, key: str, value: Any) -> None:
        """Set data for JSON output."""
        self._json_data[key] = value

    def add_to_list(self, key: str, item: Any) -> None:
        """Add item to a list in JSON data."""
        if key not in self._json_data:
            self._json_data[key] = []
        self._json_data[key].append(item)

    def output_json(self, data: Optional[Dict[str, Any]] = None) -> None:
        """Output JSON data and exit."""
        output = data if data is not None else self._json_data
        print(json.dumps(output, indent=2, default=str))

    def print_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        col_widths: Optional[List[int]] = None,
    ) -> None:
        """Print a simple table."""
        if self.is_json:
            return

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


def get_formatter(args) -> OutputFormatter:
    """Get formatter from args."""
    format_type = getattr(args, "format", "table")
    return OutputFormatter(format_type)


__all__ = ["OutputFormatter", "get_formatter"]
