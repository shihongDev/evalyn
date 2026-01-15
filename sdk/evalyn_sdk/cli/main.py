"""Evalyn CLI main entry point."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

# Import command modules
from .commands import (
    analysis,
    annotation,
    calibration,
    dataset,
    evaluation,
    export,
    infrastructure,
    runs,
    simulate,
    traces,
)


def _print_ascii_help(parser: argparse.ArgumentParser) -> None:
    """Print ASCII art banner and help."""
    # Dont change this ascii art
    art = r"""
         ______  __      __    /\       _     __     __  __   __
        |  ____| \ \    / /   /  \     | |    \ \   / /  | \ | |
        | |__     \ \  / /   / /\ \    | |     \ \_/ /   |  \| |
              Evalyn CLI — Streamlined Evaluation Framework
        |  __|     \ \/ /   / ____ \   | |      \   /    | . ` |
        | |____     \  /   / /    \ \  | |____   | |     | |\  |
        |______|     \/   /_/      \_\ |______|  |_|     |_| \_|

    """
    print("=" * 80)
    print(art)
    print("=" * 80)
    parser.print_help()


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="evalyn",
        description="Evalyn CLI - Instrument, trace, and evaluate LLM agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  evalyn list-calls --limit 20              List recent traced calls
  evalyn show-call --id <call_id>           Show details for a specific call
  evalyn build-dataset --project myproj     Build dataset from traces
  evalyn suggest-metrics --target app.py:func --mode basic
                                            Suggest metrics (fast heuristic)
  evalyn run-eval --dataset data/myproj/dataset.jsonl --metrics data/myproj/metrics/basic.json
                                            Run evaluation on dataset

For more info on a command: evalyn <command> --help
""",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress hint messages"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Help command
    help_parser = subparsers.add_parser(
        "help", help="Show available commands and examples", add_help=False
    )
    help_parser.set_defaults(func=lambda args: _print_ascii_help(parser))

    # Register commands from extracted modules
    traces.register_commands(subparsers)
    runs.register_commands(subparsers)
    dataset.register_commands(subparsers)
    simulate.register_commands(subparsers)
    export.register_commands(subparsers)
    analysis.register_commands(subparsers)
    annotation.register_commands(subparsers)
    calibration.register_commands(subparsers)
    evaluation.register_commands(subparsers)
    infrastructure.register_commands(subparsers)

    # Parse and execute
    args = parser.parse_args(argv)

    # Handle --version flag (only when it's a boolean True, not a string value from subcommands)
    if args.version is True:
        from .. import __version__

        print(f"evalyn {__version__}")
        return

    # Require a command if --version wasn't used
    if not args.command:
        parser.print_help()
        return

    try:
        args.func(args)
    except BrokenPipeError:
        # Allow piping to tools that close stdout early (e.g., `head`, `Select-Object -First`).
        try:
            sys.stdout.close()
        finally:
            return


if __name__ == "__main__":
    main()
