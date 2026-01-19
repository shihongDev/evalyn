"""Clustering commands for eval results analysis.

This module provides CLI commands for clustering evaluation failures and disagreements
by semantic similarity of judge reasons.

Commands:
- cluster-failures: Cluster failed items from eval runs (no annotations needed)
- cluster-misalignments: Cluster judge vs human disagreements (requires annotations)

Output formats:
- html: Interactive Plotly scatter plot with hover details
- table: ASCII table for terminal output
- json: Raw clustering data

Typical workflows:

Failure clustering (after eval):
  evalyn run-eval --dataset data/...
  evalyn cluster-failures                    # All metrics with failures
  evalyn cluster-failures --metric-id <id>   # Single metric

Misalignment clustering (after annotation):
  evalyn annotate --dataset <path> --per-metric
  evalyn cluster-misalignments --metric-id <id> --annotations <path>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from ...analysis.clustering import (
    ReasonClusterer,
    generate_cluster_html,
    generate_cluster_text,
    generate_failure_cluster_html,
    generate_failure_cluster_text,
)
from ...annotation import import_annotations
from ...annotation.calibration import CalibrationEngine
from ...datasets import load_dataset
from ...decorators import get_default_tracer
from ...models import DatasetItem, EvalRun, MetricResult
from ..utils.config import load_config, resolve_dataset_path
from ..utils.errors import fatal_error
from ..utils.hints import print_hint


def _get_eval_run(args: argparse.Namespace) -> EvalRun:
    """Load eval run."""
    tracer = get_default_tracer()
    if not tracer.storage:
        fatal_error("No storage configured")

    run = tracer.storage.get_eval_run(args.run_id) if args.run_id else None
    if run is None:
        runs = tracer.storage.list_eval_runs(limit=1)
        run = runs[0] if runs else None
    if run is None:
        fatal_error("No eval runs available")

    return run


def _get_eval_run_and_metrics(
    args: argparse.Namespace,
) -> tuple[EvalRun, list[MetricResult]]:
    """Load eval run and filter metric results by metric_id."""
    run = _get_eval_run(args)

    metric_results = [r for r in run.metric_results if r.metric_id == args.metric_id]
    if not metric_results:
        fatal_error(
            f"No metric results found for metric_id={args.metric_id} in run {run.id}"
        )

    return run, metric_results


def _load_dataset_context(
    args: argparse.Namespace,
) -> tuple[Optional[list[DatasetItem]], Optional[Path]]:
    """Load dataset items and determine dataset directory."""
    config = load_config()
    dataset_items: Optional[list[DatasetItem]] = None
    dataset_dir: Optional[Path] = None

    dataset_arg = getattr(args, "dataset", None)
    use_latest = getattr(args, "latest", False)
    resolved_dataset = resolve_dataset_path(dataset_arg, use_latest, config)

    if resolved_dataset:
        dataset_dir = Path(resolved_dataset)
        dataset_file = (
            dataset_dir / "dataset.jsonl" if dataset_dir.is_dir() else dataset_dir
        )
        if dataset_file.exists():
            dataset_items = load_dataset(dataset_file)
            if dataset_dir.is_file():
                dataset_dir = dataset_dir.parent

    return dataset_items, dataset_dir


def _write_output(
    output_format: str,
    output_path: Optional[str],
    default_path: Path,
    result_dict: dict,
    html_content: str,
    text_content: str,
) -> None:
    """Write clustering output in the specified format."""
    if output_format == "json":
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2)
            print(f"Clustering result saved to: {output_path}")
        else:
            print(json.dumps(result_dict, indent=2))

    elif output_format == "html":
        target_path = Path(output_path) if output_path else default_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML report saved to: {target_path}")

    else:  # table format
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            print(f"Text report saved to: {output_path}")
        else:
            print(text_content)


def cmd_cluster_misalignments(args: argparse.Namespace) -> None:
    """Cluster LLM judge vs human disagreements by semantic similarity."""
    _, metric_results = _get_eval_run_and_metrics(args)

    # Load annotations
    anns = import_annotations(args.annotations)
    if not anns:
        fatal_error(f"No annotations found in {args.annotations}")

    # Load dataset items for context
    dataset_items, dataset_dir = _load_dataset_context(args)

    # Analyze disagreements
    engine = CalibrationEngine(judge_name=args.metric_id, optimize_prompts=False)
    disagreements = engine.analyze_disagreements(metric_results, anns, dataset_items)

    if disagreements.total_disagreements == 0:
        print(f"\nNo disagreements found for metric '{args.metric_id}'")
        print("LLM judge and human annotations are fully aligned!")
        return

    # Run clustering
    cache_dir = dataset_dir / "calibrations" / args.metric_id if dataset_dir else None
    clusterer = ReasonClusterer(model=args.model, cache_dir=cache_dir)
    compute_embeddings = args.format == "html"
    result = clusterer.cluster_reasons(disagreements, compute_embeddings=compute_embeddings)

    # Output
    default_html_path = (
        dataset_dir / "calibrations" / args.metric_id / "clusters.html"
        if dataset_dir
        else Path(f"clusters_{args.metric_id}.html")
    )
    _write_output(
        args.format,
        args.output,
        default_html_path,
        result.as_dict(),
        generate_cluster_html(result, args.metric_id),
        generate_cluster_text(result, args.metric_id),
    )

    # Show summary
    if args.format != "table":
        fp_count = len(result.false_positive_clusters)
        fn_count = len(result.false_negative_clusters)
        fp_cases = sum(c.count for c in result.false_positive_clusters)
        fn_cases = sum(c.count for c in result.false_negative_clusters)
        print(f"\nClustering complete:")
        print(f"  False Positives: {fp_count} clusters ({fp_cases} cases)")
        print(f"  False Negatives: {fn_count} clusters ({fn_cases} cases)")

    if dataset_dir:
        print_hint(
            f"To calibrate this metric, run: evalyn calibrate --metric-id {args.metric_id} --annotations {args.annotations} --dataset {dataset_dir}",
            quiet=getattr(args, "quiet", False),
        )


def cmd_cluster_failures(args: argparse.Namespace) -> None:
    """Cluster failed items from eval runs by semantic similarity of judge reasons."""
    run = _get_eval_run(args)

    # Load dataset items for context
    dataset_items, dataset_dir = _load_dataset_context(args)

    # Determine which metrics to cluster
    metric_id = getattr(args, "metric_id", None)

    if metric_id:
        # Single metric mode
        metric_ids = [metric_id]
    else:
        # All metrics mode: find metrics with failures
        metrics_with_failures = set()
        for r in run.metric_results:
            if r.passed is False:
                metrics_with_failures.add(r.metric_id)
        metric_ids = sorted(metrics_with_failures)

        if not metric_ids:
            print("\nNo failures found in any metric!")
            print(f"All items passed across {len(set(r.metric_id for r in run.metric_results))} metrics.")
            return

        print(f"\nClustering failures for {len(metric_ids)} metrics with failures...")

    clusterer = ReasonClusterer(model=args.model, cache_dir=None)
    compute_embeddings = args.format == "html"
    quiet = getattr(args, "quiet", False)

    for mid in metric_ids:
        metric_results = [r for r in run.metric_results if r.metric_id == mid]
        failed_results = [r for r in metric_results if r.passed is False]

        if not failed_results:
            if metric_id:  # Only print if single metric was explicitly requested
                print(f"\nNo failures found for metric '{mid}'")
                print(f"All {len(metric_results)} items passed!")
            continue

        # Set cache dir per metric
        if dataset_dir:
            clusterer.cache_dir = dataset_dir / "analysis" / mid

        result = clusterer.cluster_failures(
            metric_results, dataset_items, compute_embeddings=compute_embeddings
        )

        # Determine output path
        if args.output and len(metric_ids) == 1:
            output_path = args.output
        else:
            output_path = None  # Use default

        default_html_path = (
            dataset_dir / "analysis" / mid / "failures.html"
            if dataset_dir
            else Path(f"failures_{mid}.html")
        )

        _write_output(
            args.format,
            output_path,
            default_html_path,
            result.as_dict(),
            generate_failure_cluster_html(result, mid),
            generate_failure_cluster_text(result, mid),
        )

        # Show summary
        if args.format != "table":
            total_failures = result.total_cases
            total_items = len(metric_results)
            pct = 100 * total_failures / total_items if total_items else 0
            print(f"  {mid}: {total_failures}/{total_items} failed ({pct:.1f}%), {len(result.clusters)} patterns")

    print_hint(
        "To see full eval results: evalyn show-run --last",
        quiet=quiet,
    )


def register_commands(subparsers) -> None:
    """Register clustering commands."""
    # cluster-failures: Cluster failed items from eval runs
    failures_parser = subparsers.add_parser(
        "cluster-failures",
        help="Cluster failed items from eval runs by failure reason",
    )
    failures_parser.add_argument(
        "--metric-id",
        help="Metric ID to analyze (default: all metrics with failures)",
    )
    failures_parser.add_argument(
        "--run-id",
        help="Eval run id to analyze; defaults to latest run",
    )
    failures_parser.add_argument(
        "--dataset",
        help="Path to dataset folder (provides input/output context)",
    )
    failures_parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recently modified dataset",
    )
    failures_parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="LLM model for clustering (default: gemini-2.5-flash-lite)",
    )
    failures_parser.add_argument(
        "--format",
        choices=["html", "table", "json"],
        default="html",
        help="Output format: html (scatter plot), table (ASCII), json (default: html)",
    )
    failures_parser.add_argument(
        "--output",
        help="Path to save output file (default: auto-generated)",
    )
    failures_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress hints and extra output",
    )
    failures_parser.set_defaults(func=cmd_cluster_failures)

    # cluster-misalignments: Cluster judge vs human disagreements
    misalign_parser = subparsers.add_parser(
        "cluster-misalignments",
        help="Cluster LLM judge vs human disagreements by semantic similarity",
    )
    misalign_parser.add_argument(
        "--metric-id",
        required=True,
        help="Metric ID to analyze (usually the judge metric id)",
    )
    misalign_parser.add_argument(
        "--annotations",
        required=True,
        help="Path to annotations JSONL (target_id must match call_id)",
    )
    misalign_parser.add_argument(
        "--run-id",
        help="Eval run id to analyze; defaults to latest run",
    )
    misalign_parser.add_argument(
        "--dataset",
        help="Path to dataset folder (provides input/output context)",
    )
    misalign_parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recently modified dataset",
    )
    misalign_parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="LLM model for clustering (default: gemini-2.5-flash-lite)",
    )
    misalign_parser.add_argument(
        "--format",
        choices=["html", "table", "json"],
        default="html",
        help="Output format: html (scatter plot), table (ASCII), json (default: html)",
    )
    misalign_parser.add_argument(
        "--output",
        help="Path to save output file (default: auto-generated)",
    )
    misalign_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress hints and extra output",
    )
    misalign_parser.set_defaults(func=cmd_cluster_misalignments)


__all__ = ["cmd_cluster_failures", "cmd_cluster_misalignments", "register_commands"]
