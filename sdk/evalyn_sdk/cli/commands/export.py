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

# Dashboard catalog group (used by evalyn_dashboard.introspect.build_catalog).
GROUP = "Export"

import argparse
import csv
import io
import json
from datetime import datetime
from pathlib import Path

from ..utils.command_common import load_eval_run_for_command
from ..utils.config import load_config, resolve_dataset_path
from ..utils.errors import fatal_error
from ..utils.hints import HintCollector
from ..utils.rich import icon


def cmd_export_for_annotation(args: argparse.Namespace) -> None:
    """Export dataset items with eval results for human annotation."""
    from ...decorators import get_default_tracer
    from ...models import AnnotationItem

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

    # Pre-index metric results by item_id
    from collections import defaultdict
    results_by_item: dict[str, list] = defaultdict(list)
    if run:
        for mr in run.metric_results:
            results_by_item[mr.item_id].append(mr)

    # Build annotation items
    output_items = []
    for item in items:
        eval_results = {}

        # Add eval results if we have a run
        if run:
            for mr in results_by_item.get(item.id, []):
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

    print(f"{icon('pass')} Exported {len(output_items)} items to {output_path}")

    hints = HintCollector(quiet=getattr(args, "quiet", False))
    hints.add(
        f"evalyn import-annotations --path {output_path}",
        "Import annotations after review",
    )
    hints.render()


def _load_export_run_data(args: argparse.Namespace) -> dict:
    """Load exportable eval run data from storage or dataset folder."""
    config = load_config()
    dataset_path = resolve_dataset_path(args.dataset, args.latest, config)

    loaded = load_eval_run_for_command(
        run_id=getattr(args, "run", None),
        dataset_path=dataset_path,
        error_message="No eval run found",
        error_hint="Specify --run <id> or --dataset <path>",
    )
    return loaded.run.as_dict()


def _write_or_print_export(content: str, output_path: Path | None, *, message: str) -> None:
    """Write export content to file or print to stdout."""
    if output_path:
        output_path.write_text(content, encoding="utf-8")
        print(f"{icon('pass')} {message.format(path=output_path)}")
    else:
        print(content)


def _flatten_export_rows(run_data: dict) -> list[dict]:
    """Flatten run results into CSV rows."""
    rows = []
    for mr in run_data.get("metric_results", []):
        rows.append({
            "item_id": mr.get("item_id", ""),
            "metric_id": mr.get("metric_id", ""),
            "score": mr.get("score", ""),
            "passed": mr.get("passed", ""),
            "reason": mr.get("details", {}).get("reason", "") if isinstance(mr.get("details"), dict) else "",
        })
    return rows


def _export_json(run_data: dict, output_path: Path | None) -> None:
    """Export JSON report."""
    output = json.dumps(run_data, indent=2, default=str)
    _write_or_print_export(output, output_path, message="Exported to: {path}")


def _export_csv(run_data: dict, output_path: Path | None) -> None:
    """Export CSV report."""
    rows = _flatten_export_rows(run_data)

    if output_path:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        print(f"{icon('pass')} Exported {len(rows)} rows to: {output_path}")
        return

    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output.getvalue())


def _metrics_summary_from_run_data(run_data: dict) -> dict:
    """Normalize summary shape across old/new run formats."""
    summary = run_data.get("summary", {})
    return summary.get("metrics", summary) if isinstance(summary, dict) else {}


def _build_markdown_export(run_data: dict) -> str:
    """Build markdown report."""
    lines = []
    lines.append("# Evaluation Report\n")
    lines.append(f"**Run ID:** {run_data.get('id', 'unknown')}\n")
    lines.append(f"**Dataset:** {run_data.get('dataset_name', 'unknown')}\n")
    lines.append(f"**Started:** {run_data.get('created_at', 'unknown')}\n")
    lines.append("\n## Summary\n")

    metrics_summary = _metrics_summary_from_run_data(run_data)
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

    lines.append(f"\n## Results ({len(run_data.get('metric_results', []))} items)\n")
    return "\n".join(lines)


def _build_html_export(run_data: dict) -> str:
    """Build standalone HTML report."""
    metrics_summary = _metrics_summary_from_run_data(run_data)
    results = run_data.get("metric_results", [])

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
            color = "#4CAF50" if pct >= 80 else "#FF9800" if pct >= 60 else "#f44336"
            pass_rate = f'<span style="color:{color}">{pct:.0f}%</span>'
        metric_rows += f"<tr><td>{metric_id}</td><td>{avg}</td><td>{pass_rate}</td></tr>\n"

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Evalyn Report - {run_data.get("id", "unknown")[:12]}</title>
    <style>
        body {{ font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
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
        <p><strong>Started:</strong> {run_data.get("created_at", "unknown")}</p>
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


def _export_markdown(run_data: dict, output_path: Path | None) -> None:
    """Export markdown report."""
    _write_or_print_export(
        _build_markdown_export(run_data),
        output_path,
        message="Exported to: {path}",
    )


def _export_html(run_data: dict, output_path: Path | None) -> None:
    """Export HTML report."""
    _write_or_print_export(
        _build_html_export(run_data),
        output_path,
        message="Exported to: {path}",
    )


def cmd_export(args: argparse.Namespace) -> None:
    """Export evaluation results in various formats."""
    run_data = _load_export_run_data(args)
    output_path = Path(args.output) if args.output else None
    format_type = args.format

    if format_type == "json":
        _export_json(run_data, output_path)

    elif format_type == "csv":
        _export_csv(run_data, output_path)

    elif format_type == "markdown":
        _export_markdown(run_data, output_path)

    elif format_type == "html":
        _export_html(run_data, output_path)

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
