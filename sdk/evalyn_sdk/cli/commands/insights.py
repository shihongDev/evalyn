"""Insights command: comprehensive diagnostic, prescriptive, and proactive analysis.

Surfaces actionable intelligence about evaluation results by combining:
- Metric correlations (find redundant or trading-off metrics)
- Regression detection (flag pass rate drops vs previous run)
- Input feature analysis (correlate input characteristics with outcomes)
- Score distribution analysis (detect unusual distribution shapes)
- Prioritized recommendations (what to do next)
- LLM expert panel analysis (--deep flag)
- Interactive HTML dashboard (--format html)

Usage:
    evalyn insights --latest
    evalyn insights --run <id>
    evalyn insights --dataset <path>
    evalyn insights --format json
    evalyn insights --format html
    evalyn insights --latest --deep --provider gemini
    evalyn insights --latest --deep --experts quality_analyst,strategist
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..utils.command_common import load_eval_run_for_command
from ..utils.config import load_config, resolve_dataset_path
from ..utils.dataset_resolver import get_dataset
from ..utils.errors import fatal_error
from ..utils.formatters import output_json
from ..utils.hints import print_hint
from ...analysis.core import find_eval_runs


def load_dataset_items(dataset_path: Path) -> list[dict]:
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


def load_previous_run(dataset_path: Path, current_run_path: Path) -> "EvalRun | None":
    """Load the second-most-recent run for regression detection."""
    from ...models import EvalRun

    run_files = find_eval_runs(dataset_path)

    # Find the first run that is not the current one
    for rf in run_files:
        if rf.resolve() != current_run_path.resolve():
            with open(rf, encoding="utf-8") as f:
                return EvalRun.from_dict(json.load(f))
    return None


def _print_insights_table(report, run, previous_run_id=None, panel_discussion=None):
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
        for feat in report.feature_insights:
            print(f"    {feat.finding}")

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

    # Expert panel discussion
    if panel_discussion:
        print(f"\n{'=' * 70}")
        print("  EXPERT PANEL ANALYSIS")
        print(f"{'=' * 70}")

        for expert in panel_discussion.experts:
            role_display = expert.role.replace("_", " ").title()
            print(f"\n  [{role_display}] (confidence: {expert.confidence:.0%})")
            print(f"    {expert.summary}")
            for f in expert.findings:
                print(f"      - {f}")
            if expert.concerns:
                print(f"    Concerns: {'; '.join(expert.concerns)}")

        if panel_discussion.synthesis:
            print(f"\n  SYNTHESIS:")
            print(f"    {panel_discussion.synthesis}")

        if panel_discussion.action_plan:
            print(f"\n  ACTION PLAN:")
            for i, step in enumerate(panel_discussion.action_plan, 1):
                print(f"    {i}. {step}")

        if panel_discussion.dissenting_views:
            print(f"\n  DISSENTING VIEWS:")
            for d in panel_discussion.dissenting_views:
                print(f"    - {d}")

        total_tokens = panel_discussion.total_input_tokens + panel_discussion.total_output_tokens
        print(f"\n  Tokens: {panel_discussion.total_input_tokens} in / "
              f"{panel_discussion.total_output_tokens} out ({total_tokens} total)")

    print()


def build_insights_report(run, dataset_path, run_file_path):
    """Build a RunAnalysis and InsightsReport from an eval run.

    Shared by cmd_insights and cmd_dashboard to avoid duplicating the
    analysis pipeline (correlations, regressions, features, distributions,
    recommendations).

    Returns (analysis, report, dataset_items).
    """
    from ...analysis.core import analyze_run
    from ...analysis.insights import (
        compute_metric_correlations,
        detect_regressions,
        analyze_input_features,
        analyze_score_distributions,
        generate_recommendations,
        InsightsReport,
    )

    run_data = run.as_dict()
    analysis = analyze_run(run_data)
    correlations = compute_metric_correlations(analysis)

    regressions = []
    if dataset_path and run_file_path:
        previous_run_obj = load_previous_run(dataset_path, run_file_path)
        if previous_run_obj:
            prev_analysis = analyze_run(previous_run_obj.as_dict())
            regressions = detect_regressions(analysis, prev_analysis)

    dataset_items = []
    feature_insights = []
    if dataset_path:
        dataset_items = load_dataset_items(dataset_path)
        if dataset_items:
            feature_insights = analyze_input_features(dataset_items, analysis)

    distribution_insights = analyze_score_distributions(analysis)

    report = InsightsReport(
        correlations=correlations,
        regressions=regressions,
        feature_insights=feature_insights,
        distribution_insights=distribution_insights,
    )

    report.recommendations = generate_recommendations(
        analysis,
        report,
        dataset_path=str(dataset_path) if dataset_path else None,
    )

    return analysis, report, dataset_items


def cmd_insights(args: argparse.Namespace) -> None:
    """Run comprehensive insights analysis on evaluation results."""
    output_format = getattr(args, "format", "table")
    use_deep = getattr(args, "deep", False)
    config = load_config()
    dataset_path = resolve_dataset_path(
        getattr(args, "dataset", None),
        getattr(args, "latest", False),
        config,
    )

    loaded = load_eval_run_for_command(
        run_id=getattr(args, "run", None),
        dataset_path=dataset_path,
    )
    run = loaded.run
    run_file_path = loaded.run_file_path
    if run_file_path and output_format not in ("json", "html"):
        print(f"Analyzing latest run: {run_file_path.parent.name}")

    analysis, report, dataset_items = build_insights_report(
        run, dataset_path, run_file_path
    )

    # Get previous run ID for display (lightweight file read)
    previous_run_id = None
    if dataset_path and run_file_path:
        previous_run_obj = load_previous_run(dataset_path, run_file_path)
        if previous_run_obj:
            previous_run_id = previous_run_obj.id

    # Expert panel (--deep)
    panel_discussion = None
    if use_deep:
        from ...analysis.panel import create_api_client, run_expert_panel

        provider = getattr(args, "provider", None) or "gemini"
        model = getattr(args, "model", None)
        experts_arg = getattr(args, "experts", None)
        expert_roles = experts_arg.split(",") if experts_arg else None

        if output_format not in ("json", "html"):
            print(f"\nRunning expert panel ({provider})...")

        client = create_api_client(provider=provider, model=model)
        panel_discussion = run_expert_panel(
            run_analysis=analysis,
            insights_report=report,
            client=client,
            dataset_items=dataset_items or None,
            experts=expert_roles,
            verbose=output_format not in ("json", "html"),
        )

    # Output
    if output_format == "json":
        result = {
            "run_id": run.id,
            "dataset_name": run.dataset_name,
            "previous_run_id": previous_run_id,
            "correlations": [asdict(c) for c in report.correlations],
            "regressions": [asdict(r) for r in report.regressions],
            "feature_insights": [asdict(f) for f in report.feature_insights],
            "distribution_insights": [asdict(d) for d in report.distribution_insights],
            "recommendations": [asdict(r) for r in report.recommendations],
        }
        if panel_discussion:
            result["expert_panel"] = panel_discussion.as_dict()
        output_json(result)
        return

    if output_format == "html":
        from ...analysis.insights_dashboard import generate_insights_html

        html_content = generate_insights_html(
            run_analysis=analysis,
            insights_report=report,
            panel_discussion=panel_discussion,
            dataset_items=dataset_items or None,
        )

        # Save to dataset dir or current dir
        if dataset_path:
            output_path = Path(dataset_path) / "insights_report.html"
        else:
            output_path = Path("insights_report.html")

        output_path.write_text(html_content, encoding="utf-8")
        print(f"Insights dashboard saved to: {output_path}")
        return

    _print_insights_table(report, run, previous_run_id, panel_discussion)

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
        "--format", choices=["table", "json", "html"], default="table",
        help="Output format (default: table)",
    )
    # LLM expert panel flags
    p.add_argument(
        "--deep", action="store_true",
        help="Enable LLM expert panel analysis",
    )
    p.add_argument(
        "--provider", choices=["gemini", "openai", "ollama"],
        help="LLM provider for expert panel (default: gemini)",
    )
    p.add_argument(
        "--model",
        help="Model override for expert panel",
    )
    p.add_argument(
        "--experts",
        help="Comma-separated expert subset (quality_analyst,metric_critic,data_scientist,strategist)",
    )
    p.set_defaults(func=cmd_insights)


__all__ = [
    "build_insights_report",
    "cmd_insights",
    "load_dataset_items",
    "load_previous_run",
    "register_commands",
]
