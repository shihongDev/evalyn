"""Watch command: real-time tail of incoming traced calls.

Prints the last `--limit` calls (most-recent-last), then polls storage every
`--interval` seconds and prints any newly-arrived calls one line at a time.

Useful during development: run your agent in one terminal while
`evalyn watch` streams new traces in another.

Usage:
    evalyn watch
    evalyn watch --project my_project
    evalyn watch --function my_fn --error-only
    evalyn watch --interval 1.0 --limit 10 --db test
"""

from __future__ import annotations

# Dashboard catalog group (used by evalyn_dashboard.introspect.build_catalog).
GROUP = "Tracing"

# Params worth surfacing in the dashboard form alongside required ones.
ESSENTIAL = {"interval", "limit"}

# Per-param numeric range hints for sliders in the dashboard form.
RANGES = {
    "interval": (0.5, 30.0, 0.5),
    "limit": (1, 100, 1),
}
UNITS = {"interval": "seconds"}

import argparse
import sys
import time
from datetime import datetime

from ...storage.sqlite import DB_PATHS
from ..utils.colors import green, red
from ..utils.formatters import format_duration
from ..utils.validation import extract_project_id


def _get_storage(args: argparse.Namespace):
    """Get storage based on --db flag.

    Mirrors traces.py._get_storage so watch uses the same DB resolution.
    """
    from ...storage import SQLiteStorage

    db = getattr(args, "db", None)
    if db and db in DB_PATHS:
        return SQLiteStorage(DB_PATHS[db])
    from ...decorators import get_default_tracer

    tracer = get_default_tracer()
    return tracer.storage


def _format_call_line(call) -> str:
    """Format a single call as a compact one-line summary.

    Format: ``HH:MM:SS  <short_id>  <function>  <project>  <status>  <duration_ms>ms``
    Uses green for OK, red for ERROR (if colors enabled).
    """
    ts = ""
    if call.started_at:
        if hasattr(call.started_at, "strftime"):
            ts = call.started_at.strftime("%H:%M:%S")
        else:
            ts = str(call.started_at)[11:19]

    short_id = (call.id or "")[:8]
    func = call.function_name or ""
    project = extract_project_id(call.metadata) or "-"
    status = red("ERROR") if call.error else green("OK")
    dur = format_duration(call.duration_ms)

    return f"{ts}  {short_id}  {func}  {project}  {status}  {dur}"


def _fetch_calls(storage, *, project: str | None, function: str | None, limit: int, error_only: bool):
    """Fetch calls from storage. ``error_only`` is filtered client-side."""
    if storage is None:
        return []
    # Overfetch when error_only is set since it's filtered client-side.
    fetch_limit = limit * 3 if error_only else limit
    calls = list(storage.list_calls(
        limit=fetch_limit,
        project=project,
        function_name=function,
        lightweight=True,
    ))
    if error_only:
        calls = [c for c in calls if c.error]
    return calls


def _latest_started_at(calls) -> datetime | None:
    """Return the max started_at across calls, or None if empty."""
    timestamps = [c.started_at for c in calls if c.started_at is not None]
    return max(timestamps) if timestamps else None


def _print_initial(calls, limit: int) -> None:
    """Print the initial backlog (oldest first so newest is at the bottom).

    storage.list_calls returns DESC by started_at, so we reverse to get
    chronological order. Then slice to ``limit`` from the tail.
    """
    # Take the most recent `limit` (DESC -> first `limit`), then reverse for display.
    head = calls[:limit]
    chrono = list(reversed(head))
    for call in chrono:
        print(_format_call_line(call))
    sys.stdout.flush()


def _poll_loop(
    storage,
    *,
    project: str | None,
    function: str | None,
    error_only: bool,
    interval: float,
    limit: int,
    last_seen: datetime | None,
    sleep_fn=time.sleep,
    iterations: int | None = None,
) -> datetime | None:
    """Poll storage and print new calls since ``last_seen``.

    Runs forever until KeyboardInterrupt unless ``iterations`` is set
    (used by tests to bound the loop).

    Returns the final ``last_seen`` for inspection by tests.
    """
    count = 0
    while True:
        if iterations is not None and count >= iterations:
            return last_seen

        calls = _fetch_calls(
            storage,
            project=project,
            function=function,
            limit=max(limit * 4, 20),
            error_only=error_only,
        )

        # Filter to strictly-newer-than last_seen, then sort chronologically.
        if last_seen is not None:
            new_calls = [c for c in calls if c.started_at and c.started_at > last_seen]
        else:
            new_calls = [c for c in calls if c.started_at is not None]

        new_calls.sort(key=lambda c: c.started_at)

        for call in new_calls:
            print(_format_call_line(call))
        if new_calls:
            sys.stdout.flush()
            last_seen = new_calls[-1].started_at

        count += 1
        sleep_fn(interval)


def cmd_watch(args: argparse.Namespace) -> int:
    """Tail incoming traced calls in real time.

    Prints the most recent ``--limit`` calls first, then polls storage every
    ``--interval`` seconds for new ones. Exits cleanly on Ctrl+C.
    """
    storage = _get_storage(args)
    project = getattr(args, "project", None)
    function = getattr(args, "function", None)
    error_only = getattr(args, "error_only", False)
    interval = getattr(args, "interval", 2.0)
    limit = getattr(args, "limit", 5)

    # Sanity: interval must be non-negative; clamp to avoid busy loops.
    if interval < 0:
        interval = 0.0

    # Initial backlog: print the last `limit` calls, oldest-last so the next
    # new call appears below.
    initial = _fetch_calls(
        storage,
        project=project,
        function=function,
        limit=limit,
        error_only=error_only,
    )
    _print_initial(initial, limit)
    last_seen = _latest_started_at(initial)

    try:
        _poll_loop(
            storage,
            project=project,
            function=function,
            error_only=error_only,
            interval=interval,
            limit=limit,
            last_seen=last_seen,
        )
    except KeyboardInterrupt:
        print("watch stopped", file=sys.stderr)
        return 0
    return 0


def register_commands(subparsers) -> None:
    """Register the watch command."""
    p = subparsers.add_parser(
        "watch",
        help="Tail incoming traced calls in real time",
    )
    p.add_argument("--project", help="Filter by project name")
    p.add_argument("--function", help="Filter by function name (substring match)")
    p.add_argument("--error-only", action="store_true", help="Show only calls with errors")
    p.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds (default: 2.0)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of existing calls to show before tailing (default: 5)",
    )
    p.add_argument(
        "--db",
        choices=["prod", "test"],
        help="Database to use: prod (default) or test",
    )
    p.set_defaults(func=cmd_watch)


__all__ = ["cmd_watch", "register_commands"]
