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
  evalyn cluster-failures --metric-id <id>

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
from ...models import DatasetItem
from ..utils.config import load_config, resolve_dataset_path
from ..utils.errors import fatal_error
from ..utils.hints import print_hint


def cmd_cluster_misalignments(args: argparse.Namespace) -> None:
    """Cluster LLM judge vs human disagreements by semantic similarity.

    Clustering process:
    1. Load metric results from an eval run (LLM judge verdicts)
    2. Load human annotations (ground truth labels)
    3. Identify disagreement cases (false positives/negatives)
    4. Cluster by semantic similarity using LLM
    5. Optionally compute embeddings for visualization
    6. Output as HTML scatter plot, ASCII table, or JSON
    """
    tracer = get_default_tracer()
    if not tracer.storage:
        fatal_error("No storage configured")

    run = tracer.storage.get_eval_run(args.run_id) if args.run_id else None
    if run is None:
        runs = tracer.storage.list_eval_runs(limit=1)
        run = runs[0] if runs else None
    if run is None:
        fatal_error("No eval runs available")

    metric_results = [r for r in run.metric_results if r.metric_id == args.metric_id]
    if not metric_results:
        fatal_error(
            f"No metric results found for metric_id={args.metric_id} in run {run.id}"
        )

    # Load annotations
    anns = import_annotations(args.annotations)
    if not anns:
        fatal_error(f"No annotations found in {args.annotations}")

    # Load dataset items for context
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

    # Create calibration engine to analyze disagreements
    engine = CalibrationEngine(
        judge_name=args.metric_id,
        optimize_prompts=False,  # We only need disagreement analysis
    )

    # Get disagreements
    disagreements = engine.analyze_disagreements(metric_results, anns, dataset_items)

    if disagreements.total_disagreements == 0:
        print(f"\nNo disagreements found for metric '{args.metric_id}'")
        print("LLM judge and human annotations are fully aligned!")
        return

    # Set up cache directory
    cache_dir = None
    if dataset_dir:
        cache_dir = dataset_dir / "calibrations" / args.metric_id

    # Create clusterer
    clusterer = ReasonClusterer(
        model=args.model,
        cache_dir=cache_dir,
    )

    # Run clustering
    compute_embeddings = args.format == "html"
    result = clusterer.cluster_reasons(disagreements, compute_embeddings=compute_embeddings)

    # Output based on format
    output_format = args.format
    output_path = args.output

    if output_format == "json":
        output_data = result.as_dict()
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)
            print(f"Clustering result saved to: {output_path}")
        else:
            print(json.dumps(output_data, indent=2))

    elif output_format == "html":
        html_content = generate_cluster_html(result, args.metric_id)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"HTML report saved to: {output_path}")
        else:
            # Default output path
            default_path = Path(f"clusters_{args.metric_id}.html")
            if dataset_dir:
                default_path = dataset_dir / "calibrations" / args.metric_id / "clusters.html"
                default_path.parent.mkdir(parents=True, exist_ok=True)
            with open(default_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"HTML report saved to: {default_path}")

    else:  # table format
        text_output = generate_cluster_text(result, args.metric_id)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text_output)
            print(f"Text report saved to: {output_path}")
        else:
            print(text_output)

    # Show summary
    fp_count = len(result.false_positive_clusters)
    fn_count = len(result.false_negative_clusters)
    fp_cases = sum(c.count for c in result.false_positive_clusters)
    fn_cases = sum(c.count for c in result.false_negative_clusters)

    if output_format != "table":
        print(f"\nClustering complete:")
        print(f"  False Positives: {fp_count} clusters ({fp_cases} cases)")
        print(f"  False Negatives: {fn_count} clusters ({fn_cases} cases)")

    # Show hint
    if dataset_dir:
        print_hint(
            f"To calibrate this metric, run: evalyn calibrate --metric-id {args.metric_id} --annotations {args.annotations} --dataset {dataset_dir}",
            quiet=getattr(args, "quiet", False),
        )


def cmd_cluster_failures(args: argparse.Namespace) -> None:
    """Cluster failed items from eval runs by semantic similarity of judge reasons.

    This command clusters failures from LLM evaluation runs without requiring
    human annotations. It groups failed items by the judge's reasoning to
    identify common failure patterns.

    Clustering process:
    1. Load metric results from an eval run
    2. Filter to failed items (passed=False)
    3. Cluster by semantic similarity of judge reasons
    4. Output as HTML scatter plot, ASCII table, or JSON
    """
    tracer = get_default_tracer()
    if not tracer.storage:
        fatal_error("No storage configured")

    # Get eval run
    run = tracer.storage.get_eval_run(args.run_id) if args.run_id else None
    if run is None:
        runs = tracer.storage.list_eval_runs(limit=1)
        run = runs[0] if runs else None
    if run is None:
        fatal_error("No eval runs available")

    # Filter metric results
    metric_results = [r for r in run.metric_results if r.metric_id == args.metric_id]
    if not metric_results:
        fatal_error(
            f"No metric results found for metric_id={args.metric_id} in run {run.id}"
        )

    # Count failures
    failed_results = [r for r in metric_results if r.passed is False]
    if not failed_results:
        print(f"\nNo failures found for metric '{args.metric_id}'")
        print(f"All {len(metric_results)} items passed!")
        return

    # Load dataset items for context
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

    # Set up cache directory
    cache_dir = None
    if dataset_dir:
        cache_dir = dataset_dir / "analysis" / args.metric_id

    # Create clusterer
    clusterer = ReasonClusterer(
        model=args.model,
        cache_dir=cache_dir,
    )

    # Run failure clustering
    compute_embeddings = args.format == "html"
    result = clusterer.cluster_failures(
        metric_results, dataset_items, compute_embeddings=compute_embeddings
    )

    # Output based on format
    output_format = args.format
    output_path = args.output

    if output_format == "json":
        output_data = result.as_dict()
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)
            print(f"Clustering result saved to: {output_path}")
        else:
            print(json.dumps(output_data, indent=2))

    elif output_format == "html":
        html_content = generate_failure_cluster_html(result, args.metric_id)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"HTML report saved to: {output_path}")
        else:
            # Default output path
            default_path = Path(f"failures_{args.metric_id}.html")
            if dataset_dir:
                default_path = dataset_dir / "analysis" / args.metric_id / "failures.html"
                default_path.parent.mkdir(parents=True, exist_ok=True)
            with open(default_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"HTML report saved to: {default_path}")

    else:  # table format
        text_output = generate_failure_cluster_text(result, args.metric_id)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text_output)
            print(f"Text report saved to: {output_path}")
        else:
            print(text_output)

    # Show summary
    cluster_count = len(result.clusters)
    total_failures = result.total_cases
    total_items = len(metric_results)

    if output_format != "table":
        print(f"\nClustering complete:")
        print(f"  {total_failures}/{total_items} items failed ({100*total_failures/total_items:.1f}%)")
        print(f"  {cluster_count} failure patterns identified")

    # Show hint
    print_hint(
        f"To see full eval results: evalyn show-run --last",
        quiet=getattr(args, "quiet", False),
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
        required=True,
        help="Metric ID to analyze",
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
