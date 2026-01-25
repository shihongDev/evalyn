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


def format_cost(cost: float) -> str:
    """Format a cost value as a string with appropriate precision."""
    if cost < 1:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def print_token_usage_summary(usage: Dict[str, Any], verbose: bool = False) -> None:
    """Print token usage and cost summary from a usage_summary dict.

    Args:
        usage: Usage summary dict with keys like total_input_tokens, total_cost_usd, etc.
        verbose: If True, show per-metric cost breakdown.
    """
    if not usage or usage.get("total_tokens", 0) == 0:
        return

    input_tok = usage.get("total_input_tokens", 0)
    output_tok = usage.get("total_output_tokens", 0)
    total_tok = usage.get("total_tokens", 0)
    models = usage.get("models_used", [])
    total_cost = usage.get("total_cost_usd", 0.0)
    has_unknown = usage.get("has_unknown_pricing", False)

    # Format cost string with asterisk if unknown pricing
    cost_str = format_cost(total_cost)
    if has_unknown:
        cost_str = f"~{cost_str}*"

    print(f"\nToken usage: {input_tok:,} input + {output_tok:,} output = {total_tok:,} total ({cost_str})")

    if models:
        print(f"Models: {', '.join(models)}")

    if has_unknown:
        print("  * Cost is approximate - some model pricing not in registry")

    # Show verbose cost breakdown if requested
    cost_by_metric = usage.get("cost_by_metric", {})
    if verbose and cost_by_metric and len(cost_by_metric) > 1:
        print("\nCost breakdown by metric:")
        sorted_metrics = sorted(cost_by_metric.items(), key=lambda x: x[1], reverse=True)
        for metric_id, cost in sorted_metrics:
            pct = (cost / total_cost * 100) if total_cost > 0 else 0
            print(f"  {metric_id:<25} {format_cost(cost):>10}  ({pct:.0f}%)")


__all__ = [
    "is_json_format",
    "print_if_table",
    "print_error_if_table",
    "output_json",
    "print_table",
    "format_cost",
    "print_token_usage_summary",
]
