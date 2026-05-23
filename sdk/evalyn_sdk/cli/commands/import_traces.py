"""Import-traces command: ingest external provider logs into evalyn storage.

Supports three input formats:
- openai:    chat completions JSONL log
- anthropic: messages JSONL log
- generic:   minimal JSONL with input/output (+ optional model, tokens)

Each record becomes one FunctionCall containing one llm_call Span,
written to the same SQLite database that the live tracer uses, so the
imported records show up in `evalyn list-calls`, `show-call`, `show-trace`.

Usage:
    evalyn import-traces --file logs.jsonl --format openai
    evalyn import-traces --file logs.jsonl --format anthropic --project my-app
    evalyn import-traces --file logs.jsonl --format generic --db test
"""

from __future__ import annotations

# Dashboard catalog group (used by evalyn_dashboard.introspect.build_catalog).
GROUP = "Tracing"

# Surface these non-required params in the dashboard's default form.
ESSENTIAL = {"project", "function"}

import argparse
from pathlib import Path

from ...integration.trace_importers import FORMAT_IMPORTERS, import_traces
from ...storage.sqlite import DB_PATHS
from ..utils.errors import fatal_error
from ..utils.rich import banner, kv, section


def _get_storage(args: argparse.Namespace):
    """Resolve storage backend based on --db flag, mirroring traces.py."""
    from ...storage import SQLiteStorage

    db = getattr(args, "db", None)
    if db and db in DB_PATHS:
        return SQLiteStorage(DB_PATHS[db])
    from ...decorators import get_default_tracer

    return get_default_tracer().storage


def cmd_import_traces(args: argparse.Namespace) -> None:
    """Import external trace logs into evalyn's SQLite store."""
    file_path = Path(args.file)
    if not file_path.exists():
        fatal_error(
            f"Input file not found: {args.file}",
            "Check the path and try again",
        )
    if not file_path.is_file():
        fatal_error(
            f"Not a regular file: {args.file}",
            "Pass the path to a JSONL log file",
        )

    fmt = args.format
    if fmt not in FORMAT_IMPORTERS:
        fatal_error(
            f"Unsupported format: {fmt!r}",
            f"Choose one of: {', '.join(sorted(FORMAT_IMPORTERS))}",
        )

    print()
    print(banner("EVALYN IMPORT TRACES"))
    print()
    print(section("PARSING"))
    parse_pairs: list[tuple[str, str]] = [
        ("File", str(file_path)),
        ("Format", fmt),
    ]
    if args.project:
        parse_pairs.append(("Project", args.project))
    if args.function:
        parse_pairs.append(("Function name", args.function))
    print(kv(parse_pairs))

    calls = import_traces(
        file_path,
        fmt=fmt,
        project=args.project,
        function_name=args.function,
    )

    if not calls:
        print()
        print("  No records imported. Either the file is empty or all records were skipped.")
        print("  Re-run with `python -W default` to see per-record warnings.")
        return

    storage = _get_storage(args)
    for call in calls:
        storage.store_call(call)

    started = sorted(c.started_at for c in calls)
    first_started = started[0].isoformat() if started else "n/a"
    last_started = started[-1].isoformat() if started else "n/a"
    storage_path = str(storage.path)

    print()
    print(section("RESULT"))
    print(f"  Imported {len(calls)} call(s) into {storage_path}.")
    print(
        kv(
            [
                ("Project", args.project or "(none)"),
                ("First started", first_started),
                ("Last started", last_started),
            ]
        )
    )
    print()
    print("  Next steps:")
    print("    evalyn list-calls" + (f" --project {args.project}" if args.project else ""))
    print("    evalyn show-call --last")


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the import-traces command."""
    p = subparsers.add_parser(
        "import-traces",
        help="Import external trace logs (OpenAI / Anthropic / generic JSONL) into evalyn storage",
    )
    p.add_argument(
        "--file",
        required=True,
        help="Path to the JSONL log file to import",
    )
    p.add_argument(
        "--format",
        required=True,
        choices=sorted(FORMAT_IMPORTERS),
        help="Input log format",
    )
    p.add_argument(
        "--project",
        help="Project name to tag imported calls with (shows up in list-calls --project)",
    )
    p.add_argument(
        "--function",
        help="Override the function_name field for imported calls",
    )
    p.add_argument(
        "--db",
        choices=["prod", "test"],
        help="Database to write to: prod (default) or test",
    )
    p.set_defaults(func=cmd_import_traces)


__all__ = ["cmd_import_traces", "register_commands"]
