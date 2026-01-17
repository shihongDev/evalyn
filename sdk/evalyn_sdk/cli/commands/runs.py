"""Run history commands: list-runs, show-run.

This module provides CLI commands for viewing evaluation run history.
Each time you run 'evalyn run-eval', a run record is stored in the database
with metrics used, results, and summary statistics.

Commands:
- list-runs: List stored eval runs with basic info (dataset, metrics count, results count)
- show-run: Show details for a specific eval run (metric summaries, individual results)

Run information includes:
- Run ID: Unique identifier for the run
- Dataset name: Which dataset was evaluated
- Metrics: Which metrics were used
- Summary: Aggregated pass rates and scores per metric
- Results: Individual metric results per item

Typical workflow:
1. Run evaluation: 'evalyn run-eval --dataset <path>'
2. List runs: 'evalyn list-runs'
3. View details: 'evalyn show-run --id <run_id>'
4. Analyze: 'evalyn analyze --run <run_id>'
"""

from __future__ import annotations

import argparse
import json

from ...decorators import get_default_tracer
from ..utils.errors import fatal_error
from ..utils.hints import print_hint


def cmd_list_runs(args: argparse.Namespace) -> None:
    """List stored eval runs."""
    tracer = get_default_tracer()
    if not tracer.storage:
        fatal_error("No storage configured")
    runs = tracer.storage.list_eval_runs(limit=args.limit)
    output_format = getattr(args, "format", "table")

    if not runs:
        if output_format == "json":
            print("[]")
        else:
            print("No eval runs found.")
        return

    if output_format == "json":
        result = []
        for run in runs:
            result.append(
                {
                    "id": run.id,
                    "dataset_name": run.dataset_name,
                    "created_at": run.created_at.isoformat()
                    if run.created_at
                    else None,
                    "metrics_count": len(run.metrics),
                    "results_count": len(run.metric_results),
                    "summary": run.summary,
                }
            )
        print(json.dumps(result, indent=2))
        return

    headers = ["id", "dataset", "created_at", "metrics", "results"]
    print(" | ".join(headers))
    print("-" * 120)
    first_run_id = None
    for run in runs:
        if first_run_id is None:
            first_run_id = run.id[:8]  # Use short ID for hint
        # Use short ID (first 8 chars) for easier copy-paste
        short_id = run.id[:8]
        row = [
            short_id,
            run.dataset_name,
            str(run.created_at),
            str(len(run.metrics)),
            str(len(run.metric_results)),
        ]
        print(" | ".join(row))

    if first_run_id:
        print_hint(
            f"To see details, run: evalyn show-run --id {first_run_id}",
            quiet=getattr(args, "quiet", False),
        )


def cmd_show_run(args: argparse.Namespace) -> None:
    """Show details for a specific eval run."""
    tracer = get_default_tracer()
    output_format = getattr(args, "format", "table")

    if not tracer.storage:
        fatal_error("No storage configured")

    # Handle --last flag to show most recent run
    if getattr(args, "last", False):
        runs = tracer.storage.list_eval_runs(limit=1)
        if not runs:
            fatal_error("No eval runs found")
        run = runs[0]
    elif args.id:
        # Resolve short ID to full ID (supports prefixes like '6cf21eb3')
        input_id = args.id
        if hasattr(tracer.storage, "resolve_eval_run_id"):
            resolved = tracer.storage.resolve_eval_run_id(input_id)
            if resolved:
                run_id = resolved
            else:
                fatal_error(
                    f"No eval run found matching '{input_id}'",
                    "Use more characters for a unique match",
                )
        else:
            run_id = input_id
        run = tracer.storage.get_eval_run(run_id)
    else:
        fatal_error("--id or --last required")

    if not run:
        fatal_error(f"No eval run found with id={args.id}")

    # JSON output mode
    if output_format == "json":
        result = {
            "id": run.id,
            "dataset_name": run.dataset_name,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "metrics": [
                m.as_dict() if hasattr(m, "as_dict") else m for m in run.metrics
            ],
            "summary": run.summary,
            "metric_results": [
                {
                    "metric_id": res.metric_id,
                    "item_id": res.item_id,
                    "call_id": res.call_id,
                    "score": res.score,
                    "passed": res.passed,
                    "details": res.details,
                }
                for res in run.metric_results
            ],
        }
        print(json.dumps(result, indent=2, default=str))
        return

    # Table output mode
    print(f"\n=== Eval Run {run.id} ===")
    print(f"Dataset: {run.dataset_name}")
    print("Metrics summary:")
    for mid, stats in run.summary.get("metrics", {}).items():
        print(
            f" - {mid:<18} count={stats['count']:<3} "
            f"avg_score={stats['avg_score']!s:<10} pass_rate={stats['pass_rate']!s:<10}"
        )
    if run.summary.get("failed_items"):
        print(f"Failed items: {run.summary['failed_items']}")
    print("\nMetric results:")
    for res in run.metric_results:
        print(
            f" [{res.metric_id}] item={res.item_id} call={res.call_id} "
            f"score={res.score} passed={res.passed} details={res.details}"
        )


def register_commands(subparsers) -> None:
    """Register run history commands."""
    # list-runs
    p = subparsers.add_parser("list-runs", help="List stored eval runs")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    p.set_defaults(func=cmd_list_runs)

    # show-run
    p = subparsers.add_parser("show-run", help="Show details for an eval run")
    p.add_argument("--id", help="Eval run id to display")
    p.add_argument("--last", action="store_true", help="Show the most recent eval run")
    p.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    p.set_defaults(func=cmd_show_run)


__all__ = ["cmd_list_runs", "cmd_show_run", "register_commands"]
