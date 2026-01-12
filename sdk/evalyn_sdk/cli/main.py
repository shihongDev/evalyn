"""Evalyn CLI main entry point."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

# Import command modules
from .commands import traces

# Temporarily import remaining commands from original cli_impl.py until extracted
# These will be migrated to separate modules incrementally
from ..cli_impl import (
    # Evaluation commands
    cmd_run_eval,
    cmd_suggest_metrics,
    cmd_list_runs,
    cmd_show_run,
    cmd_status,
    cmd_select_metrics,
    cmd_trend,
    # Dataset commands
    cmd_build_dataset,
    cmd_export_for_annotation,
    cmd_export,
    cmd_validate,
    cmd_list_metrics,
    # Annotation commands
    cmd_annotate,
    cmd_import_annotations,
    cmd_annotation_stats,
    # Calibration commands
    cmd_calibrate,
    cmd_list_calibrations,
    # Simulation commands
    cmd_simulate,
    # Analysis commands
    cmd_analyze,
    cmd_compare,
    # Infrastructure commands
    cmd_init,
    cmd_one_click,
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
    subparsers = parser.add_subparsers(dest="command")

    # Help command
    help_parser = subparsers.add_parser(
        "help", help="Show available commands and examples", add_help=False
    )
    help_parser.set_defaults(func=lambda args: _print_ascii_help(parser))

    # Register trace commands from new module
    traces.register_commands(subparsers)

    # ----- Evaluation commands (to be extracted to commands/evaluation.py) -----
    run_parser = subparsers.add_parser(
        "run-eval", help="Run evaluation on dataset using specified metrics"
    )
    run_parser.add_argument(
        "--dataset",
        help="Path to JSON/JSONL dataset file or directory containing dataset.jsonl",
    )
    run_parser.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    run_parser.add_argument(
        "--metrics",
        help="Path to metrics JSON file(s), comma-separated for multiple (auto-detected from meta.json if omitted)",
    )
    run_parser.add_argument(
        "--metrics-all",
        action="store_true",
        help="Use all metrics files from the metrics/ folder",
    )
    run_parser.add_argument(
        "--use-calibrated",
        action="store_true",
        help="Use calibrated prompts for subjective metrics (if available)",
    )
    run_parser.add_argument(
        "--dataset-name", help="Name for the eval run (defaults to dataset filename)"
    )
    run_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    run_parser.set_defaults(func=cmd_run_eval)

    suggest_parser = subparsers.add_parser(
        "suggest-metrics", help="Suggest metrics for a project or target function"
    )
    suggest_parser.add_argument(
        "--project",
        help="Project name (use 'evalyn show-projects' to see available projects)",
    )
    suggest_parser.add_argument(
        "--version", help="Filter by version (optional, used with --project)"
    )
    suggest_parser.add_argument(
        "--target",
        help="Callable to analyze in the form module:function (alternative to --project)",
    )
    suggest_parser.add_argument(
        "--num-traces",
        type=int,
        default=5,
        help="How many recent traces to include as examples",
    )
    suggest_parser.add_argument(
        "--num-metrics", type=int, help="Maximum number of metrics to return"
    )
    suggest_parser.add_argument(
        "--mode",
        choices=["auto", "llm-registry", "llm-brainstorm", "bundle", "basic"],
        default="auto",
        help="auto (use function's metric_mode if set), llm-registry (LLM picks from registry), llm-brainstorm (free-form), bundle (preset), or basic heuristic.",
    )
    suggest_parser.add_argument(
        "--llm-mode",
        choices=["local", "api"],
        default="api",
        help="When using LLM modes: choose local (ollama) or api (OpenAI/Gemini-compatible).",
    )
    suggest_parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="Model name (e.g., gemini-2.5-flash-lite, gpt-4, llama3.1 for Ollama)",
    )
    suggest_parser.add_argument(
        "--api-base", help="Custom API base URL for --llm-mode api (optional)"
    )
    suggest_parser.add_argument(
        "--api-key", help="API key override for --llm-mode api (optional)"
    )
    suggest_parser.add_argument(
        "--llm-caller",
        help="Optional callable path that accepts a prompt string and returns a list of metric dicts",
    )
    suggest_parser.add_argument(
        "--bundle",
        help="Bundle name when --mode bundle (e.g., summarization, orchestrator, research-agent)",
    )
    suggest_parser.add_argument(
        "--dataset",
        help="Dataset directory (or dataset.jsonl/meta.json) to save metrics into",
    )
    suggest_parser.add_argument(
        "--metrics-name", help="Metrics set name when saving to a dataset"
    )
    suggest_parser.add_argument(
        "--scope",
        choices=["all", "overall", "llm_call", "tool_call", "trace"],
        default="all",
        help="Filter metrics by scope: overall (final output), llm_call (per LLM), tool_call (per tool), trace (aggregates), or all.",
    )
    suggest_parser.set_defaults(func=cmd_suggest_metrics)

    runs_parser = subparsers.add_parser("list-runs", help="List stored eval runs")
    runs_parser.add_argument("--limit", type=int, default=10)
    runs_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    runs_parser.set_defaults(func=cmd_list_runs)

    show_run = subparsers.add_parser("show-run", help="Show details for an eval run")
    show_run.add_argument("--id", required=True, help="Eval run id to display")
    show_run.set_defaults(func=cmd_show_run)

    status_parser = subparsers.add_parser(
        "status",
        help="Show status of a dataset (items, metrics, runs, annotations, calibrations)",
    )
    status_parser.add_argument("--dataset", help="Path to dataset directory")
    status_parser.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    status_parser.set_defaults(func=cmd_status)

    select_parser = subparsers.add_parser(
        "select-metrics", help="LLM-guided selection from metric registry"
    )
    select_parser.add_argument(
        "--target",
        required=True,
        help="Callable to analyze in the form module:function",
    )
    select_parser.add_argument(
        "--llm-caller",
        required=True,
        help="Callable that accepts a prompt and returns metric ids or dicts",
    )
    select_parser.add_argument(
        "--limit", type=int, default=5, help="Recent traces to include as examples"
    )
    select_parser.set_defaults(func=cmd_select_metrics)

    # ----- Dataset commands (to be extracted to commands/datasets.py) -----
    build_ds = subparsers.add_parser(
        "build-dataset", help="Build dataset from stored traces"
    )
    build_ds.add_argument(
        "--output",
        help="Path to write dataset JSONL (default: data/<project>-<version>-<timestamp>.jsonl)",
    )
    build_ds.add_argument(
        "--project",
        help="Filter by metadata.project_id or project_name (recommended grouping)",
    )
    build_ds.add_argument("--version", help="Filter by metadata.version")
    build_ds.add_argument(
        "--simulation", action="store_true", help="Include only simulation traces"
    )
    build_ds.add_argument(
        "--production", action="store_true", help="Include only production traces"
    )
    build_ds.add_argument("--since", help="ISO timestamp lower bound for started_at")
    build_ds.add_argument("--until", help="ISO timestamp upper bound for started_at")
    build_ds.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max number of items to include (after filtering)",
    )
    build_ds.add_argument(
        "--include-errors",
        action="store_true",
        help="Include errored calls (default: skip)",
    )
    build_ds.set_defaults(func=cmd_build_dataset)

    export_ann = subparsers.add_parser(
        "export-for-annotation",
        help="Export dataset with eval results for human annotation",
    )
    export_ann.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory or dataset.jsonl file",
    )
    export_ann.add_argument(
        "--output", required=True, help="Output path for annotation JSONL file"
    )
    export_ann.add_argument(
        "--run-id", help="Specific eval run ID to use (defaults to latest)"
    )
    export_ann.set_defaults(func=cmd_export_for_annotation)

    export_parser = subparsers.add_parser(
        "export", help="Export evaluation results in various formats"
    )
    export_parser.add_argument("--run", help="Eval run ID to export")
    export_parser.add_argument(
        "--dataset", help="Dataset path (uses latest run from eval_runs/)"
    )
    export_parser.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    export_parser.add_argument(
        "--format",
        choices=["json", "csv", "markdown", "html"],
        default="json",
        help="Output format (default: json)",
    )
    export_parser.add_argument(
        "--output", "-o", help="Output file path (prints to stdout if not specified)"
    )
    export_parser.set_defaults(func=cmd_export)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate dataset format and detect potential issues"
    )
    validate_parser.add_argument("--dataset", help="Path to dataset directory or file")
    validate_parser.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    validate_parser.set_defaults(func=cmd_validate)

    list_metrics = subparsers.add_parser(
        "list-metrics", help="List available metric templates (objective + subjective)"
    )
    list_metrics.set_defaults(func=cmd_list_metrics)

    # ----- Annotation commands (to be extracted to commands/annotation.py) -----
    annotate_parser = subparsers.add_parser(
        "annotate", help="Interactive CLI for annotating dataset items"
    )
    annotate_parser.add_argument(
        "--dataset", help="Path to dataset directory or dataset.jsonl file"
    )
    annotate_parser.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    annotate_parser.add_argument(
        "--run-id", help="Eval run ID to show LLM judge results (defaults to latest)"
    )
    annotate_parser.add_argument(
        "--output",
        help="Output path for annotations (defaults to <dataset>/annotations.jsonl)",
    )
    annotate_parser.add_argument(
        "--annotator", default="human", help="Annotator name/id (default: human)"
    )
    annotate_parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart annotation from scratch (ignore existing)",
    )
    annotate_parser.add_argument(
        "--per-metric",
        action="store_true",
        help="Annotate each metric separately (agree/disagree with LLM)",
    )
    annotate_parser.add_argument(
        "--spans",
        action="store_true",
        help="Annotate individual spans (LLM calls, tool calls, etc.)",
    )
    annotate_parser.add_argument(
        "--span-type",
        choices=["all", "llm_call", "tool_call", "reasoning", "retrieval"],
        default="all",
        help="Filter span types to annotate",
    )
    annotate_parser.add_argument(
        "--only-disagreements",
        action="store_true",
        help="Only show items where LLM and prior human labels disagree",
    )
    annotate_parser.set_defaults(func=cmd_annotate)

    import_ann = subparsers.add_parser(
        "import-annotations", help="Import annotations from a JSONL file"
    )
    import_ann.add_argument("--path", required=True, help="Path to annotations JSONL")
    import_ann.set_defaults(func=cmd_import_annotations)

    ann_stats = subparsers.add_parser(
        "annotation-stats", help="Show annotation coverage statistics"
    )
    ann_stats.add_argument(
        "--dataset",
        required=True,
        help="Path to annotations.jsonl or dataset directory",
    )
    ann_stats.set_defaults(func=cmd_annotation_stats)

    # ----- Calibration commands (to be extracted to commands/calibration.py) -----
    calibrate_parser = subparsers.add_parser(
        "calibrate", help="Calibrate a subjective metric using human annotations"
    )
    calibrate_parser.add_argument(
        "--metric-id",
        required=True,
        help="Metric ID to calibrate (usually the judge metric id)",
    )
    calibrate_parser.add_argument(
        "--annotations",
        required=True,
        help="Path to annotations JSONL (target_id must match call_id)",
    )
    calibrate_parser.add_argument(
        "--run-id", help="Eval run id to calibrate; defaults to latest run"
    )
    calibrate_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Current threshold for pass/fail"
    )
    calibrate_parser.add_argument(
        "--dataset",
        help="Path to dataset folder (provides input/output context for optimization)",
    )
    calibrate_parser.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    calibrate_parser.add_argument(
        "--no-optimize", action="store_true", help="Skip LLM-based prompt optimization"
    )
    calibrate_parser.add_argument(
        "--optimizer",
        choices=["llm", "gepa"],
        default="llm",
        help="Optimization method: 'llm' (default) or 'gepa' (evolutionary)",
    )
    calibrate_parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="LLM model for prompt optimization (llm mode)",
    )
    calibrate_parser.add_argument(
        "--gepa-task-lm",
        default="gemini/gemini-2.5-flash",
        help="Task model for GEPA (model being optimized)",
    )
    calibrate_parser.add_argument(
        "--gepa-reflection-lm",
        default="gemini/gemini-2.5-flash",
        help="Reflection model for GEPA (strong model for reflection)",
    )
    calibrate_parser.add_argument(
        "--gepa-max-calls",
        type=int,
        default=150,
        help="Max metric calls budget for GEPA optimization",
    )
    calibrate_parser.add_argument(
        "--show-examples", action="store_true", help="Show example disagreement cases"
    )
    calibrate_parser.add_argument(
        "--output", help="Path to save calibration record JSON"
    )
    calibrate_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    calibrate_parser.set_defaults(func=cmd_calibrate)

    list_cal_parser = subparsers.add_parser(
        "list-calibrations", help="List calibration records for a dataset"
    )
    list_cal_parser.add_argument("--dataset", help="Path to dataset directory")
    list_cal_parser.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    list_cal_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    list_cal_parser.set_defaults(func=cmd_list_calibrations)

    # ----- Simulation commands (to be extracted to commands/simulation.py) -----
    simulate_parser = subparsers.add_parser(
        "simulate", help="Generate synthetic test data using LLM-based user simulation"
    )
    simulate_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to seed dataset directory or dataset.jsonl",
    )
    simulate_parser.add_argument(
        "--target",
        help="Target function to run queries against (module:func or path/to/file.py:func)",
    )
    simulate_parser.add_argument(
        "--output",
        help="Output directory for simulated data (default: <dataset>/simulations/)",
    )
    simulate_parser.add_argument(
        "--modes",
        default="similar,outlier",
        help="Simulation modes: similar,outlier (comma-separated)",
    )
    simulate_parser.add_argument(
        "--num-similar",
        type=int,
        default=3,
        help="Number of similar variations per seed item (default: 3)",
    )
    simulate_parser.add_argument(
        "--num-outlier",
        type=int,
        default=1,
        help="Number of outlier/edge cases per seed item (default: 1)",
    )
    simulate_parser.add_argument(
        "--max-seeds",
        type=int,
        default=50,
        help="Maximum seed items to use (default: 50)",
    )
    simulate_parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="LLM model for query generation",
    )
    simulate_parser.add_argument(
        "--temp-similar",
        type=float,
        default=0.3,
        help="Temperature for similar queries (default: 0.3)",
    )
    simulate_parser.add_argument(
        "--temp-outlier",
        type=float,
        default=0.8,
        help="Temperature for outlier queries (default: 0.8)",
    )
    simulate_parser.set_defaults(func=cmd_simulate)

    # ----- Analysis commands (to be extracted to commands/analysis.py) -----
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze evaluation results and generate insights"
    )
    analyze_parser.add_argument("--run", help="Eval run ID to analyze")
    analyze_parser.add_argument(
        "--dataset", help="Dataset path (uses latest run from eval_runs/)"
    )
    analyze_parser.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare two evaluation runs side-by-side"
    )
    compare_parser.add_argument(
        "--run1", required=True, help="First eval run ID or path to run JSON file"
    )
    compare_parser.add_argument(
        "--run2", required=True, help="Second eval run ID or path to run JSON file"
    )
    compare_parser.set_defaults(func=cmd_compare)

    trend_parser = subparsers.add_parser(
        "trend", help="Show evaluation trends over time for a project"
    )
    trend_parser.add_argument(
        "--project",
        required=True,
        help="Project/dataset name to analyze trends for",
    )
    trend_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of runs to analyze (default: 20)",
    )
    trend_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    trend_parser.set_defaults(func=cmd_trend)

    # ----- Infrastructure commands (to be extracted to commands/infrastructure.py) -----
    init_parser = subparsers.add_parser("init", help="Initialize configuration file")
    init_parser.add_argument(
        "--output",
        default="evalyn.yaml",
        help="Output path for config file (default: evalyn.yaml)",
    )
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing config file"
    )
    init_parser.set_defaults(func=cmd_init)

    oneclick_parser = subparsers.add_parser(
        "one-click",
        help="Run complete evaluation pipeline (dataset -> metrics -> eval -> annotate -> calibrate)",
    )
    oneclick_parser.add_argument(
        "--project", required=True, help="Project name to filter traces"
    )
    oneclick_parser.add_argument(
        "--target",
        help="Target function (file.py:func or module:func). Optional - if not provided, uses existing trace outputs",
    )
    oneclick_parser.add_argument(
        "--version", help="Version filter (default: all versions)"
    )
    oneclick_parser.add_argument(
        "--production-only", action="store_true", help="Use only production traces"
    )
    oneclick_parser.add_argument(
        "--simulation-only", action="store_true", help="Use only simulation traces"
    )
    oneclick_parser.add_argument(
        "--output-dir", help="Custom output directory (default: auto-generated)"
    )
    oneclick_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous incomplete run (auto-detects from output-dir)",
    )
    oneclick_parser.add_argument(
        "--dataset-limit",
        type=int,
        default=100,
        help="Max dataset items (default: 100)",
    )
    oneclick_parser.add_argument(
        "--since", help="Filter traces since date (ISO format)"
    )
    oneclick_parser.add_argument(
        "--until", help="Filter traces until date (ISO format)"
    )
    oneclick_parser.add_argument(
        "--metric-mode",
        choices=["basic", "llm-registry", "llm-brainstorm", "bundle", "all"],
        default="basic",
        help="Metric selection mode: basic (fast), llm-registry (LLM picks from templates), "
        "llm-brainstorm (LLM generates custom), bundle (preset), all (comprehensive - runs all modes)",
    )
    oneclick_parser.add_argument(
        "--llm-mode",
        choices=["api", "local"],
        default="api",
        help="LLM mode for metric selection (default: api)",
    )
    oneclick_parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="LLM model name (default: gemini-2.5-flash-lite)",
    )
    oneclick_parser.add_argument("--bundle", help="Bundle name (if metric-mode=bundle)")
    oneclick_parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Skip annotation step (default: false)",
    )
    oneclick_parser.add_argument(
        "--annotation-limit",
        type=int,
        default=20,
        help="Max items to annotate (default: 20)",
    )
    oneclick_parser.add_argument(
        "--per-metric", action="store_true", help="Use per-metric annotation mode"
    )
    oneclick_parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration step (default: false)",
    )
    oneclick_parser.add_argument(
        "--optimizer",
        choices=["llm", "gepa"],
        default="llm",
        help="Prompt optimization method (default: llm)",
    )
    oneclick_parser.add_argument(
        "--calibrate-all-metrics",
        action="store_true",
        help="Calibrate all subjective metrics (default: only poorly-aligned)",
    )
    oneclick_parser.add_argument(
        "--enable-simulation",
        action="store_true",
        help="Enable simulation step (default: false)",
    )
    oneclick_parser.add_argument(
        "--simulation-modes",
        default="similar",
        help="Simulation modes: similar,outlier (default: similar)",
    )
    oneclick_parser.add_argument(
        "--num-similar",
        type=int,
        default=3,
        help="Similar queries per seed (default: 3)",
    )
    oneclick_parser.add_argument(
        "--num-outlier",
        type=int,
        default=2,
        help="Outlier queries per seed (default: 2)",
    )
    oneclick_parser.add_argument(
        "--max-sim-seeds",
        type=int,
        default=10,
        help="Max seeds for simulation (default: 10)",
    )
    oneclick_parser.add_argument(
        "--auto-yes",
        action="store_true",
        help="Skip all confirmation prompts (default: false)",
    )
    oneclick_parser.add_argument(
        "--verbose", action="store_true", help="Show detailed logs (default: false)"
    )
    oneclick_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing (default: false)",
    )
    oneclick_parser.set_defaults(func=cmd_one_click)

    # Parse and execute
    args = parser.parse_args(argv)

    # Handle --version flag
    if args.version:
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
