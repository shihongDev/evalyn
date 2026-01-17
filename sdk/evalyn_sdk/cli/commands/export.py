"""Export commands: export, export-for-annotation.

This module provides CLI commands for exporting evaluation results and datasets
in various formats for reporting, sharing, or further processing.

Commands:
- export: Export evaluation results in JSON, CSV, Markdown, or HTML formats
- export-for-annotation: Export dataset items with eval results for human annotation

Export formats:
- json: Full structured data, good for programmatic access
- csv: Tabular format with one row per metric result, good for spreadsheets
- markdown: Human-readable report with summary table
- html: Standalone HTML report with styling, good for sharing

Typical workflow:
1. Run evaluation: 'evalyn run-eval --dataset <path>'
2. Export results: 'evalyn export --dataset <path> --format html -o report.html'
3. For annotation: 'evalyn export-for-annotation --dataset <path> --output annotations.jsonl'
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime
from pathlib import Path

from ..utils.config import load_config, resolve_dataset_path
from ..utils.errors import fatal_error
from ..utils.hints import print_hint


def cmd_export_for_annotation(args: argparse.Namespace) -> None:
    """Export dataset items with eval results for human annotation."""
    from ...models import AnnotationItem
    from ...decorators import get_default_tracer

    tracer = get_default_tracer()
    if not tracer.storage:
        fatal_error("No storage configured")

    # Load dataset
    config = load_config()
    dataset_path = resolve_dataset_path(args.dataset, False, config)
    if not dataset_path:
        fatal_error("--dataset required")

    dataset_file = dataset_path / "dataset.jsonl"
    if not dataset_file.exists():
        fatal_error(f"Dataset not found at {dataset_file}")

    from ...datasets import load_dataset

    items = list(load_dataset(dataset_file))

    # Get eval run
    run = None
    if args.run_id:
        run = tracer.storage.get_eval_run(args.run_id)
    else:
        runs = tracer.storage.list_eval_runs(limit=1)
        if runs:
            run = runs[0]

    if not run:
        print("Warning: No eval run found, exporting without LLM results")

    # Build annotation items
    output_items = []
    for item in items:
        eval_results = {}

        # Add eval results if we have a run
        if run:
            for mr in run.metric_results:
                if mr.item_id == item.id:
                    eval_results[mr.metric_id] = {
                        "score": mr.score,
                        "passed": mr.passed,
                        "reason": mr.details.get("reason") if mr.details else None,
                    }

        ann_item = AnnotationItem(
            id=item.id,
            input=item.input if item.input else item.inputs,
            output=item.output,
            eval_results=eval_results,
            human_label=None,
            metadata={"expected": item.expected} if item.expected else {},
        )

        output_items.append(ann_item)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in output_items:
            f.write(
                json.dumps(
                    {
                        "id": item.id,
                        "input": item.input,
                        "output": item.output,
                        "eval_results": item.eval_results,
                        "human_label": None,
                        "metadata": item.metadata,
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    print(f"Exported {len(output_items)} items to {output_path}")

    print_hint(
        f"After annotating, import with: evalyn import-annotations --path {output_path}",
        quiet=getattr(args, "quiet", False),
    )


def cmd_export(args: argparse.Namespace) -> None:
    """Export evaluation results in various formats."""
    from ...storage import SQLiteStorage

    config = load_config()
    dataset_path = resolve_dataset_path(args.dataset, args.latest, config)

    # Load eval run
    run = None
    run_data = None

    if args.run:
        storage = SQLiteStorage()
        run = storage.get_eval_run(args.run)
        if run:
            run_data = {
                "id": run.id,
                "dataset_name": run.dataset_name,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "results": run.results,
                "summary": run.summary,
                "metadata": run.metadata,
            }
    elif dataset_path:
        runs_dir = dataset_path / "eval_runs"
        if runs_dir.exists():
            # Look for results.json in timestamped subdirectories
            run_files = sorted(runs_dir.glob("*/results.json"), reverse=True)
            if not run_files:
                # Fallback to direct json files
                run_files = sorted(runs_dir.glob("*.json"), reverse=True)
            if run_files:
                with open(run_files[0], encoding="utf-8") as f:
                    run_data = json.load(f)

    if not run_data:
        fatal_error("No eval run found", "Specify --run <id> or --dataset <path>")

    output_path = Path(args.output) if args.output else None
    format_type = args.format

    if format_type == "json":
        output = json.dumps(run_data, indent=2, default=str)
        if output_path:
            output_path.write_text(output)
            print(f"Exported to: {output_path}")
        else:
            print(output)

    elif format_type == "csv":
        rows = []
        for result in run_data.get("results", []):
            item_id = result.get("item_id", "")
            for mr in result.get("metrics", []):
                rows.append(
                    {
                        "item_id": item_id,
                        "metric_id": mr.get("metric_id", ""),
                        "score": mr.get("score", ""),
                        "passed": mr.get("passed", ""),
                        "reason": mr.get("reason", ""),
                    }
                )

        if output_path:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            print(f"Exported {len(rows)} rows to: {output_path}")
        else:
            output = io.StringIO()
            if rows:
                writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(output.getvalue())

    elif format_type == "markdown":
        lines = []
        lines.append("# Evaluation Report\n")
        lines.append(f"**Run ID:** {run_data.get('id', 'unknown')}\n")
        lines.append(f"**Dataset:** {run_data.get('dataset_name', 'unknown')}\n")
        lines.append(f"**Started:** {run_data.get('started_at', 'unknown')}\n")
        lines.append("\n## Summary\n")

        summary = run_data.get("summary", {})
        # Handle both old format (summary.items) and new format (summary.metrics.items)
        metrics_summary = (
            summary.get("metrics", summary) if isinstance(summary, dict) else {}
        )
        if metrics_summary and isinstance(metrics_summary, dict):
            lines.append("| Metric | Avg Score | Pass Rate |")
            lines.append("|--------|-----------|-----------|")
            for metric_id, stats in metrics_summary.items():
                if not isinstance(stats, dict):
                    continue
                avg = stats.get("avg_score", stats.get("avg", "N/A"))
                pass_rate = stats.get("pass_rate", "N/A")
                if isinstance(avg, float):
                    avg = f"{avg:.2f}"
                if isinstance(pass_rate, float):
                    pass_rate = f"{pass_rate:.0%}"
                lines.append(f"| {metric_id} | {avg} | {pass_rate} |")

        lines.append(f"\n## Results ({len(run_data.get('results', []))} items)\n")

        output = "\n".join(lines)
        if output_path:
            output_path.write_text(output)
            print(f"Exported to: {output_path}")
        else:
            print(output)

    elif format_type == "html":
        # Generate standalone HTML report
        summary = run_data.get("summary", {})
        # Handle both old format and new format
        metrics_summary = (
            summary.get("metrics", summary) if isinstance(summary, dict) else {}
        )
        results = run_data.get("results", [])

        # Build metric table
        metric_rows = ""
        for metric_id, stats in metrics_summary.items():
            if not isinstance(stats, dict):
                continue
            avg = stats.get("avg_score", stats.get("avg", "N/A"))
            pass_rate = stats.get("pass_rate", "N/A")
            if isinstance(avg, float):
                avg = f"{avg:.2f}"
            if isinstance(pass_rate, float):
                pct = pass_rate * 100
                color = (
                    "#4CAF50" if pct >= 80 else "#FF9800" if pct >= 60 else "#f44336"
                )
                pass_rate = f'<span style="color:{color}">{pct:.0f}%</span>'
            metric_rows += (
                f"<tr><td>{metric_id}</td><td>{avg}</td><td>{pass_rate}</td></tr>\n"
            )

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Evalyn Report - {run_data.get("id", "unknown")[:12]}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .meta {{ background: #fff; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .meta p {{ margin: 5px 0; color: #666; }}
        .meta strong {{ color: #333; }}
        table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th {{ background: #4CAF50; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f9f9f9; }}
        .pass {{ color: #4CAF50; }}
        .fail {{ color: #f44336; }}
        .footer {{ margin-top: 40px; color: #999; font-size: 12px; text-align: center; }}
    </style>
</head>
<body>
    <h1>Evalyn Evaluation Report</h1>
    <div class="meta">
        <p><strong>Run ID:</strong> {run_data.get("id", "unknown")}</p>
        <p><strong>Dataset:</strong> {run_data.get("dataset_name", "unknown")}</p>
        <p><strong>Started:</strong> {run_data.get("started_at", "unknown")}</p>
        <p><strong>Items:</strong> {len(results)}</p>
    </div>

    <h2>Metric Summary</h2>
    <table>
        <tr><th>Metric</th><th>Avg Score</th><th>Pass Rate</th></tr>
        {metric_rows}
    </table>

    <div class="footer">
        Generated by Evalyn on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</body>
</html>"""

        if output_path:
            output_path.write_text(html)
            print(f"Exported to: {output_path}")
        else:
            print(html)

    else:
        fatal_error(f"Unknown format '{format_type}'")


def register_commands(subparsers) -> None:
    """Register export commands."""
    # export-for-annotation
    p = subparsers.add_parser(
        "export-for-annotation",
        help="Export dataset with eval results for human annotation",
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory or dataset.jsonl file",
    )
    p.add_argument(
        "--output", required=True, help="Output path for annotation JSONL file"
    )
    p.add_argument("--run-id", help="Specific eval run ID to use (defaults to latest)")
    p.set_defaults(func=cmd_export_for_annotation)

    # export
    p = subparsers.add_parser(
        "export", help="Export evaluation results in various formats"
    )
    p.add_argument("--run", help="Eval run ID to export")
    p.add_argument("--dataset", help="Dataset path (uses latest run from eval_runs/)")
    p.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    p.add_argument(
        "--format",
        choices=["json", "csv", "markdown", "html"],
        default="json",
        help="Output format (default: json)",
    )
    p.add_argument(
        "--output", "-o", help="Output file path (prints to stdout if not specified)"
    )
    p.set_defaults(func=cmd_export)


__all__ = ["cmd_export_for_annotation", "cmd_export", "register_commands"]
