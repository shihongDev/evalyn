"""Search command: query DSL over captured traces.

Usage examples:
    evalyn search "duration_ms > 5000"
    evalyn search "project = myapp and error = true"
    evalyn search "function contains chat or metadata.is_simulation = true"

Reuses the dataset filter parser (`datasets/filter.py:parse_filter`) for
syntax, with a FunctionCall-aware field resolver.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ...datasets.filter import CompoundFilter, FilterExpression, parse_filter
from ...storage.sqlite import DB_PATHS

# Dashboard catalog group
GROUP = "Tracing"
ESSENTIAL = {"query"}


SUPPORTED_FIELDS = """
Supported fields:
  id                 Call short ID
  function           Function name
  project            Project name (from metadata)
  duration_ms        Wall-clock duration in milliseconds
  error              true/false (call ended with an error)
  status             "ok" | "error"
  started_at         ISO timestamp
  metadata.<key>     Any nested metadata field

Operators: = != > >= < <= contains not_contains matches is_null is_not_null
Combine with: and | or (AND has higher precedence)
"""


def _get_storage(args: argparse.Namespace):
    """Get storage based on --db flag (mirrors traces.py:_get_storage)."""
    from ...storage import SQLiteStorage

    db = getattr(args, "db", None)
    if db and db in DB_PATHS:
        return SQLiteStorage(DB_PATHS[db])
    from ...decorators import get_default_tracer

    tracer = get_default_tracer()
    return tracer.storage


def _resolve_call_field(call, field: str) -> Any:
    """Resolve a FunctionCall field by name. Supports `metadata.<key>` paths."""
    if field in ("id", "call_id"):
        return call.id
    if field in ("function", "function_name"):
        return call.function_name
    if field == "duration_ms":
        return call.duration_ms
    if field == "started_at":
        return call.started_at
    if field == "error":
        return bool(call.error)
    if field == "status":
        return "error" if call.error else "ok"
    if field == "project":
        meta = call.metadata if isinstance(call.metadata, dict) else {}
        return meta.get("project") or meta.get("project_id")
    if field.startswith("metadata."):
        key = field.removeprefix("metadata.")
        meta = call.metadata if isinstance(call.metadata, dict) else {}
        return meta.get(key)
    return None


def _evaluate_expression(expr: FilterExpression, call) -> bool:
    """Evaluate a single parsed expression against a FunctionCall."""
    actual = _resolve_call_field(call, expr.field)
    # Special-case `error`: the resolver coerces it to bool, but users
    # may want `error is_null` to mean "no error recorded". Pass through
    # the raw call.error (which is None / str) to honor is_null semantics.
    if expr.field == "error" and expr.operator in ("is_null", "is_not_null"):
        actual = call.error
    elif isinstance(actual, bool):
        # For =/!= operators, coerce bool to canonical strings so the
        # existing string-equality comparator works.
        actual = "true" if actual else "false"
    return FilterExpression._compare(actual, expr.operator, expr.value)


def _evaluate_filter(filt: CompoundFilter, call) -> bool:
    """Evaluate a CompoundFilter against a FunctionCall."""
    if not filt.expressions:
        return True
    results = (_evaluate_expression(e, call) for e in filt.expressions)
    return all(results) if filt.logic == "and" else any(results)


def _format_ts(ts) -> str:
    """Render a started_at value to a string. Tolerates datetime or str or None."""
    if ts is None:
        return ""
    if hasattr(ts, "isoformat"):
        return ts.isoformat(sep=" ")
    return str(ts)


def cmd_search(args: argparse.Namespace) -> None:
    """Execute a query DSL over stored traces."""
    try:
        filt = parse_filter(args.query)
    except ValueError as e:
        print(f"Error: {e}")
        print(SUPPORTED_FIELDS)
        raise SystemExit(2)

    storage = _get_storage(args)
    if storage is None:
        print("No storage available.")
        return

    # Pull a generous batch then filter client-side. Storage doesn't support
    # arbitrary predicates; we rely on client-side filtering after fetch.
    raw_calls = storage.list_calls(limit=args.scan_limit, lightweight=True)
    matched = [c for c in raw_calls if _evaluate_filter(filt, c)][: args.limit]

    if args.format == "json":
        records = [
            {
                "id": c.id,
                "function": c.function_name,
                "duration_ms": c.duration_ms,
                "started_at": _format_ts(c.started_at),
                "error": bool(c.error),
                "project": _resolve_call_field(c, "project"),
            }
            for c in matched
        ]
        print(json.dumps({"matches": len(matched), "calls": records}, indent=2))
        return

    if not matched:
        print(f"No calls matched query: {args.query}")
        print(f"(Scanned {len(raw_calls)} calls. Try --scan-limit to widen.)")
        return

    print(f"Matched {len(matched)} of {len(raw_calls)} scanned calls.\n")
    print(f"{'ID':<10}  {'Function':<24}  {'Status':<6}  {'Duration':>10}  Started")
    print("-" * 78)
    for c in matched:
        duration = f"{c.duration_ms or 0:.0f}ms" if c.duration_ms else "-"
        status = "ERROR" if c.error else "ok"
        started = _format_ts(c.started_at)[:19]
        fn = (c.function_name or "")[:24]
        print(f"{c.id[:8]:<10}  {fn:<24}  {status:<6}  {duration:>10}  {started}")


def register_commands(subparsers) -> None:
    """Register the `search` subcommand."""
    parser = subparsers.add_parser(
        "search",
        help="Search captured traces using a query DSL",
        description=(
            "Search traces by field expressions. Example:\n"
            '  evalyn search "duration_ms > 5000 and project = myapp"\n'
            + SUPPORTED_FIELDS
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="Filter query, e.g. 'duration_ms > 5000'")
    parser.add_argument(
        "--db",
        choices=list(DB_PATHS.keys()),
        help="Storage to query (prod or test)",
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Max calls to display (default 50)"
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=1000,
        help="Max calls to scan from storage before filtering (default 1000)",
    )
    parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )
    parser.set_defaults(func=cmd_search)
