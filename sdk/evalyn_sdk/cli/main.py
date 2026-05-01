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
  insights         Comprehensive diagnostic and prescriptive analysis

HUMAN FEEDBACK:
  annotate         Interactive annotation interface
  annotation-stats Show annotation coverage statistics
  import-annotations Import annotations from file

CALIBRATION:
  calibrate        Calibrate LLM judges using annotations
  list-calibrations List calibration records
  cluster-failures Cluster failed items by failure reason
  cluster-misalignments Cluster judge vs human disagreements

SIMULATION:
  simulate         Generate synthetic test data

EXPORT:
  export           Export results (json/csv/markdown/html)
  export-for-annotation Export for external annotation tools

PIPELINE:
  quickstart       Detect framework, generate snippet, set up evaluation
  init             Initialize configuration file
  one-click        Run complete pipeline in one command
  report           Generate and open interactive HTML insights report
                   (formerly 'dashboard'; the new IDE ships in evalyn-dashboard)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as importlib_metadata
import sys
from typing import List, Optional

# Command name -> module name (relative to .commands package).
# Only the module for the invoked command is imported at runtime.
_COMMAND_MODULE_MAP: dict[str, str] = {
    # traces
    "list-calls": "traces",
    "show-call": "traces",
    "show-trace": "traces",
    "show-span": "traces",
    "show-projects": "traces",
    "delete-traces": "traces",
    # runs
    "list-runs": "runs",
    "show-run": "runs",
    # dataset
    "build-dataset": "dataset",
    # simulate
    "simulate": "simulate",
    # export
    "export": "export",
    "export-for-annotation": "export",
    # analysis
    "status": "analysis",
    "validate": "analysis",
    "analyze": "analysis",
    "compare": "analysis",
    "trend": "analysis",
    # annotation
    "annotate": "annotation",
    "import-annotations": "annotation",
    "annotation-stats": "annotation",
    # calibration
    "calibrate": "calibration",
    "list-calibrations": "calibration",
    # clustering
    "cluster-failures": "clustering",
    "cluster-misalignments": "clustering",
    # evaluation
    "run-eval": "evaluation",
    "suggest-metrics": "evaluation",
    "select-metrics": "evaluation",
    "list-metrics": "evaluation",
    # insights
    "insights": "insights",
    # infrastructure
    "workflow": "infrastructure",
    "init": "infrastructure",
    "one-click": "infrastructure",
    # report (renamed from "dashboard"; static HTML insights report)
    "report": "report",
    # dashboard alias: prints deprecation warning, forwards to "report".
    # Skipped at runtime when the evalyn-dashboard plugin is installed
    # (plugin entry-point discovery takes precedence).
    "dashboard": "dashboard_alias",
    # quickstart
    "quickstart": "quickstart",
}


def _discover_plugin_commands() -> dict[str, str]:
    """Discover external command modules via the ``evalyn.commands`` entry point.

    Each entry point's ``name`` is the CLI subcommand name (e.g. ``dashboard``)
    and its ``value`` is a fully-qualified module path that exposes a
    ``register_commands(subparsers)`` callable.

    Returns a mapping ``{command_name: module_path}``. Failures during
    discovery are silently skipped so a broken plugin cannot bring down the
    core CLI.
    """

    plugins: dict[str, str] = {}
    try:
        eps = importlib_metadata.entry_points(group="evalyn.commands")
    except Exception:
        return plugins
    for ep in eps:
        try:
            plugins[ep.name] = ep.value
        except Exception:
            continue
    return plugins


def _resolve_command_module(command: str) -> Optional[str]:
    """Return the importable module path that owns ``command``.

    Plugin entry points take precedence over the built-in command map. This
    lets the ``evalyn-dashboard`` package replace the built-in ``dashboard``
    command without touching this file.
    """

    plugins = _discover_plugin_commands()
    if command in plugins:
        return plugins[command]
    static = _COMMAND_MODULE_MAP.get(command)
    if static is None:
        return None
    # Static modules live under .commands.<name>; return absolute path.
    return f"{__package__}.commands.{static}"


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
    # NOTE: This command list is manually maintained. Update when adding/removing commands.
    grouped_help = """
QUICK START
  quickstart       Detect framework, generate snippet, set up evaluation
  workflow         Show evaluation workflow and next steps
  one-click        Run complete pipeline in one command
  init             Initialize configuration file

TRACING
  list-calls       List captured function calls
  show-call        Show details of a specific call (supports short IDs)
  show-trace       Show hierarchical span tree
  show-span        Show details of a specific span
  show-projects    Show project summary
  delete-traces    Delete traces from storage

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
  insights         Comprehensive diagnostic and prescriptive analysis
  report           Generate and open interactive HTML insights report

ANNOTATION & CALIBRATION
  annotate         Interactive annotation interface
  annotation-stats Show annotation statistics
  import-annotations Import annotations from file
  calibrate        Calibrate LLM judges
  list-calibrations List calibration records
  cluster-failures Cluster failed items by failure reason
  cluster-misalignments Cluster judge vs human disagreements

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


def _find_command(argv: List[str]) -> Optional[str]:
    """Return the first non-flag token (the subcommand name), or None."""
    for arg in argv:
        if not arg.startswith("-"):
            return arg
    return None


def _build_base_parser() -> argparse.ArgumentParser:
    """Build the top-level parser without any subcommands registered."""
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
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point.

    Uses lazy imports: only the command module needed for the invoked
    subcommand is loaded, keeping startup fast for every path.
    """
    effective_argv = argv if argv is not None else sys.argv[1:]
    command = _find_command(effective_argv)

    # --- Fast paths that need zero command-module imports ---

    if command is None:
        # No subcommand: just flags like --help, --version, or nothing
        if "--version" in effective_argv:
            from .. import __version__

            print(f"evalyn {__version__}")
            return
        _print_ascii_help(_build_base_parser())
        return

    if command == "help":
        _print_ascii_help(_build_base_parser())
        return

    # --- Lazy-load only the module that owns this command ---
    # Plugin entry points (group "evalyn.commands") win over static commands.

    module_path = _resolve_command_module(command)
    if module_path is None:
        print(f"evalyn: unknown command '{command}'", file=sys.stderr)
        print("Run 'evalyn help' for a list of commands.", file=sys.stderr)
        sys.exit(2)

    mod = importlib.import_module(module_path)

    parser = _build_base_parser()
    subparsers = parser.add_subparsers(dest="command")
    mod.register_commands(subparsers)

    args = parser.parse_args(argv)

    # Handle --version flag (only when it's a boolean True, not a string value from subcommands)
    if args.version is True:
        from .. import __version__

        print(f"evalyn {__version__}")
        return

    if not args.command:
        _print_ascii_help(parser)
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
