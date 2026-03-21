"""Dashboard command: generate and open an HTML insights dashboard.

Opens a visual insights dashboard in the default browser, providing
a "local LangSmith" experience with one command.

Usage:
    evalyn dashboard
    evalyn dashboard --output report.html
    evalyn dashboard --dataset data/myapp/
    evalyn dashboard --latest
    evalyn dashboard --run <id>
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from ..utils.config import load_config, resolve_dataset_path
from ..utils.errors import fatal_error
from ..utils.hints import print_hint
from ...analysis.core import find_eval_runs
from .insights import _load_dataset_items, _load_previous_run


def _open_in_browser(file_path: Path) -> bool:
    """Try to open file in the default browser. Returns True on success."""
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(file_path)], check=True)
        elif system == "Linux":
            subprocess.run(["xdg-open", str(file_path)], check=True)
        elif system == "Windows":
            subprocess.run(["start", str(file_path)], check=True, shell=True)
        else:
            return False
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Generate and open an HTML insights dashboard."""
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
    from ...analysis.insights_dashboard import generate_insights_html

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
        run_files = find_eval_runs(dataset_path)
        if run_files:
            run_file_path = run_files[0]
            with open(run_file_path, encoding="utf-8") as f:
                run = EvalRun.from_dict(json.load(f))

    if not run:
        print("No evaluation results found. Run `evalyn run-eval` first.")
        sys.exit(1)

    print(f"Analyzing run: {run.id[:12]}...")

    # Build RunAnalysis
    run_data = run.as_dict()
    analysis = analyze_run(run_data)

    # Correlations
    correlations = compute_metric_correlations(analysis)

    # Regressions (need previous run)
    regressions = []
    if dataset_path and run_file_path:
        previous_run_obj = _load_previous_run(dataset_path, run_file_path)
        if previous_run_obj:
            prev_data = previous_run_obj.as_dict()
            prev_analysis = analyze_run(prev_data)
            regressions = detect_regressions(analysis, prev_analysis)

    # Input feature analysis
    dataset_items = []
    feature_insights = []
    if dataset_path:
        dataset_items = _load_dataset_items(dataset_path)
        if dataset_items:
            feature_insights = analyze_input_features(dataset_items, analysis)

    # Score distributions
    distribution_insights = analyze_score_distributions(analysis)

    # Build report
    report = InsightsReport(
        correlations=correlations,
        regressions=regressions,
        feature_insights=feature_insights,
        distribution_insights=distribution_insights,
    )

    # Recommendations
    report.recommendations = generate_recommendations(
        analysis,
        report,
        dataset_path=str(dataset_path) if dataset_path else None,
    )

    # Generate HTML dashboard
    html_content = generate_insights_html(
        run_analysis=analysis,
        insights_report=report,
        panel_discussion=None,
        dataset_items=dataset_items or None,
    )

    # Determine output path
    output_path = Path(getattr(args, "output", None) or ".evalyn/dashboard.html")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")

    print(f"Dashboard saved to: {output_path}")

    # Try to open in browser
    if _open_in_browser(output_path):
        print("Opened dashboard in default browser.")
    else:
        file_url = output_path.as_uri()
        print(f"Could not open browser automatically.")
        print(f"Open this file in your browser: {file_url}")

    print_hint(
        "Run `evalyn insights --deep` for LLM expert panel analysis",
        quiet=getattr(args, "quiet", False),
    )


def register_commands(subparsers) -> None:
    """Register dashboard command."""
    p = subparsers.add_parser(
        "dashboard",
        help="Generate and open an HTML insights dashboard in the browser",
    )
    p.add_argument("--run", help="Eval run ID to analyze")
    p.add_argument("--dataset", help="Dataset path (uses latest run)")
    p.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recently modified dataset",
    )
    p.add_argument(
        "--output",
        help="Output file path (default: .evalyn/dashboard.html)",
    )
    p.set_defaults(func=cmd_dashboard)


__all__ = ["cmd_dashboard", "register_commands"]
