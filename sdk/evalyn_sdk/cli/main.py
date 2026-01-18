"""Evalyn CLI main entry point.

The Evalyn CLI provides commands for the complete LLM evaluation workflow:

TRACE INSPECTION:
  list-calls       List captured function calls from traced agents
  show-call        Show detailed information about a specific call
  show-trace       Show hierarchical span tree (LLM calls, tool calls)
  show-projects    Show summary of projects with traces

DATASET BUILDING:
  build-dataset    Build evaluation dataset from stored traces

METRIC SELECTION:
  suggest-metrics  Suggest metrics (basic/bundle/llm-registry/llm-brainstorm modes)
  select-metrics   LLM-guided metric selection from registry
  list-metrics     List all available metric templates

EVALUATION:
  run-eval         Run evaluation on dataset using metrics
  list-runs        List stored evaluation runs
  show-run         Show details for a specific run

ANALYSIS:
  status           Show comprehensive dataset status
  validate         Validate dataset format
  analyze          Analyze results and generate insights
  compare          Compare two evaluation runs
  trend            Show evaluation trends over time

HUMAN FEEDBACK:
  annotate         Interactive annotation interface
  annotation-stats Show annotation coverage statistics
  import-annotations Import annotations from file

CALIBRATION:
  calibrate        Calibrate LLM judges using annotations
  list-calibrations List calibration records
  cluster-misalignments Cluster judge vs human disagreements

SIMULATION:
  simulate         Generate synthetic test data

EXPORT:
  export           Export results (json/csv/markdown/html)
  export-for-annotation Export for external annotation tools

PIPELINE:
  init             Initialize configuration file
  one-click        Run complete pipeline in one command
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

# Import command modules
from .commands import (
    analysis,
    annotation,
    calibration,
    clustering,
    dataset,
    evaluation,
    export,
    infrastructure,
    runs,
    simulate,
    traces,
)


def _print_ascii_help(parser: argparse.ArgumentParser) -> None:
    """Print ASCII art banner and grouped command help."""
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

    # Print grouped commands instead of raw argparse help
    grouped_help = """
QUICK START
  workflow         Show evaluation workflow and next steps
  one-click        Run complete pipeline in one command
  init             Initialize configuration file

TRACING
  list-calls       List captured function calls
  show-call        Show details of a specific call (supports short IDs)
  show-trace       Show hierarchical span tree
  show-projects    Show project summary

DATASET
  build-dataset    Build dataset from stored traces
  validate         Validate dataset format
  status           Show dataset status

METRICS
  suggest-metrics  Suggest metrics for evaluation
  select-metrics   LLM-guided metric selection
  list-metrics     List available metric templates

EVALUATION
  run-eval         Run evaluation on dataset
  list-runs        List stored evaluation runs
  show-run         Show details for an eval run
  analyze          Analyze results and generate insights
  compare          Compare two evaluation runs
  trend            Show evaluation trends over time

ANNOTATION & CALIBRATION
  annotate         Interactive annotation interface
  annotation-stats Show annotation statistics
  calibrate        Calibrate LLM judges
  list-calibrations List calibration records

EXPORT & SIMULATION
  export           Export results (json/csv/markdown/html)
  export-for-annotation  Export for external annotation
  simulate         Generate synthetic test data

OPTIONS
  -h, --help       Show this help
  -q, --quiet      Suppress hint messages
  --version        Show version

EXAMPLES
  evalyn workflow                              # See what to do next
  evalyn list-calls --limit 5                  # View recent traces
  evalyn build-dataset --project myapp         # Build dataset
  evalyn run-eval --dataset data/myapp/        # Run evaluation
  evalyn show-run --last                       # View latest results

For command details: evalyn <command> --help
"""
    print(grouped_help)


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
  evalyn run-eval --dataset data/myproj/dataset.jsonl --metrics data/myproj/metrics/metrics.json
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
    clustering.register_commands(subparsers)
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
