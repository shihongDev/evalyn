"""Insights command: comprehensive diagnostic, prescriptive, and proactive analysis.

Surfaces actionable intelligence about evaluation results by combining:
- Metric correlations (find redundant or trading-off metrics)
- Regression detection (flag pass rate drops vs previous run)
- Input feature analysis (correlate input characteristics with outcomes)
- Score distribution analysis (detect unusual distribution shapes)
- Prioritized recommendations (what to do next)

Usage:
    evalyn insights --latest
    evalyn insights --run <id>
    evalyn insights --dataset <path>
    evalyn insights --format json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils.config import load_config, resolve_dataset_path
from ..utils.dataset_resolver import get_dataset
from ..utils.errors import fatal_error
from ..utils.hints import print_hint


def _load_dataset_items(dataset_path: Path) -> list[dict]:
    """Load raw dataset items from dataset.jsonl."""
    dataset_file = dataset_path / "dataset.jsonl"
    if not dataset_file.exists():
        return []
    items = []
    with open(dataset_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_previous_run(dataset_path: Path, current_run_path: Path):
    """Load the second-most-recent run for regression detection."""
    from ...models import EvalRun

    runs_dir = dataset_path / "eval_runs"
    if not runs_dir.exists():
        return None

    run_files = sorted(runs_dir.glob("*/results.json"), reverse=True)
    if not run_files:
        run_files = sorted(runs_dir.glob("*.json"), reverse=True)

    # Find the run after the current one
    for rf in run_files:
        if rf.resolve() != current_run_path.resolve():
            with open(rf, encoding="utf-8") as f:
                return EvalRun.from_dict(json.load(f))
    return None


def _print_insights_table(report, run, previous_run_id=None):
    """Print insights report in table format."""
    print(f"\n{'=' * 70}")
    print("  EVALYN INSIGHTS")
    print(f"{'=' * 70}")
    print(f"\n  Run: {run.id[:12]}... ({run.dataset_name})")
    if previous_run_id:
        print(f"  Compared to: {previous_run_id[:12]}...")

    # Diagnostics
    has_diagnostics = (
        report.correlations
        or report.distribution_insights
        or report.feature_insights
    )
    if has_diagnostics:
        print(f"\n{'=' * 70}")
        print("  DIAGNOSTICS")
        print(f"{'=' * 70}")

    if report.correlations:
        print("\n  Metric Correlations:")
        for c in report.correlations:
            label = "redundant" if c.relationship == "redundant" else "tradeoff"
            print(f"    {c.metric_a} <-> {c.metric_b}  r={c.pearson} ({label})")

    if report.distribution_insights:
        print("\n  Score Distributions:")
        for d in report.distribution_insights:
            print(f"    {d.metric_id}: {d.shape} - {d.finding}")

    if report.feature_insights:
        print("\n  Input Feature Analysis:")
        for f in report.feature_insights:
            print(f"    {f.finding}")

    # Regression alerts
    if report.regressions:
        print(f"\n{'=' * 70}")
        print("  REGRESSION ALERTS")
        print(f"{'=' * 70}\n")
        for r in report.regressions:
            severity_tag = f"[{r.severity.upper()}]"
            delta_pct = abs(r.delta) * 100
            print(
                f"  {severity_tag:10} {r.metric_id} dropped {delta_pct:.0f}% "
                f"({r.previous_pass_rate * 100:.0f}% -> {r.current_pass_rate * 100:.0f}%)"
            )

    # Recommendations
    if report.recommendations:
        print(f"\n{'=' * 70}")
        print("  RECOMMENDATIONS (by priority)")
        print(f"{'=' * 70}\n")
        for rec in report.recommendations:
            print(f"  {rec.priority}. [{rec.category}] {rec.message}")
            print(f"     -> {rec.action}")
    elif not has_diagnostics and not report.regressions:
        print(f"\n{'=' * 70}")
        print("  No significant insights found - evaluation looks healthy!")
        print(f"{'=' * 70}")

    print()


def cmd_insights(args: argparse.Namespace) -> None:
    """Run comprehensive insights analysis on evaluation results."""
    from ...models import EvalRun
    from ...analysis.core import analyze_run
    from ...analysis.insights import (
        compute_metric_correlations,
        detect_regressions,
        analyze_input_features,
        analyze_score_distributions,
        generate_recommendations,
        InsightsReport,
    )

    output_format = getattr(args, "format", "table")
    config = load_config()
    dataset_path = resolve_dataset_path(
        getattr(args, "dataset", None),
        getattr(args, "latest", False),
        config,
    )

    # Load the run
    run = None
    run_file_path = None
    run_id = getattr(args, "run", None)

    if run_id:
        from ...storage import SQLiteStorage
        storage = SQLiteStorage()
        run = storage.get_eval_run(run_id)
        if not run:
            fatal_error(f"No eval run found with ID '{run_id}'")
    elif dataset_path:
        runs_dir = dataset_path / "eval_runs"
        if runs_dir.exists():
            run_files = sorted(runs_dir.glob("*/results.json"), reverse=True)
            if not run_files:
                run_files = sorted(runs_dir.glob("*.json"), reverse=True)
            if run_files:
                run_file_path = run_files[0]
                with open(run_file_path, encoding="utf-8") as f:
                    run = EvalRun.from_dict(json.load(f))
                if output_format != "json":
                    print(f"Analyzing latest run: {run_file_path.parent.name}")

    if not run:
        fatal_error("No eval runs found", "Run 'evalyn run-eval' first")

    # Build RunAnalysis
    run_data = run.as_dict()
    analysis = analyze_run(run_data)

    # Correlations
    correlations = compute_metric_correlations(analysis)

    # Regressions (need previous run)
    regressions = []
    previous_run = None
    previous_run_id = None
    if dataset_path and run_file_path:
        previous_run_obj = _load_previous_run(dataset_path, run_file_path)
        if previous_run_obj:
            previous_run_id = previous_run_obj.id
            prev_data = previous_run_obj.as_dict()
            prev_analysis = analyze_run(prev_data)
            regressions = detect_regressions(analysis, prev_analysis)

    # Input feature analysis
    feature_insights = []
    if dataset_path:
        dataset_items = _load_dataset_items(dataset_path)
        if dataset_items:
            feature_insights = analyze_input_features(dataset_items, analysis)

    # Score distributions
    distribution_insights = analyze_score_distributions(analysis)

    # Recommendations
    recommendations = generate_recommendations(
        analysis,
        correlations,
        regressions,
        feature_insights,
        distribution_insights,
        dataset_path=str(dataset_path) if dataset_path else None,
    )

    report = InsightsReport(
        correlations=correlations,
        regressions=regressions,
        feature_insights=feature_insights,
        distribution_insights=distribution_insights,
        recommendations=recommendations,
    )

    if output_format == "json":
        result = {
            "run_id": run.id,
            "dataset_name": run.dataset_name,
            "previous_run_id": previous_run_id,
            "correlations": [
                {
                    "metric_a": c.metric_a,
                    "metric_b": c.metric_b,
                    "pearson": c.pearson,
                    "relationship": c.relationship,
                }
                for c in report.correlations
            ],
            "regressions": [
                {
                    "metric_id": r.metric_id,
                    "previous_pass_rate": r.previous_pass_rate,
                    "current_pass_rate": r.current_pass_rate,
                    "delta": r.delta,
                    "severity": r.severity,
                }
                for r in report.regressions
            ],
            "feature_insights": [
                {
                    "feature_name": f.feature_name,
                    "finding": f.finding,
                    "affected_items": f.affected_items,
                    "pass_rate_low": f.pass_rate_low,
                    "pass_rate_high": f.pass_rate_high,
                }
                for f in report.feature_insights
            ],
            "distribution_insights": [
                {
                    "metric_id": d.metric_id,
                    "shape": d.shape,
                    "finding": d.finding,
                }
                for d in report.distribution_insights
            ],
            "recommendations": [
                {
                    "priority": rec.priority,
                    "category": rec.category,
                    "message": rec.message,
                    "action": rec.action,
                }
                for rec in report.recommendations
            ],
        }
        print(json.dumps(result, indent=2, default=str))
        return

    _print_insights_table(report, run, previous_run_id)

    print_hint(
        "To drill into failures, run: evalyn cluster-failures"
        + (f" --dataset {dataset_path}" if dataset_path else " --latest"),
        quiet=getattr(args, "quiet", False),
        format=output_format,
    )


def register_commands(subparsers) -> None:
    """Register insights command."""
    p = subparsers.add_parser(
        "insights",
        help="Comprehensive diagnostic, prescriptive, and proactive analysis",
    )
    p.add_argument("--run", help="Eval run ID to analyze")
    p.add_argument("--dataset", help="Dataset path (uses latest run)")
    p.add_argument(
        "--latest", action="store_true",
        help="Use the most recently modified dataset",
    )
    p.add_argument(
        "--format", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )
    p.set_defaults(func=cmd_insights)


__all__ = ["cmd_insights", "register_commands"]
