"""
Analysis module for comprehensive eval results analysis and visualization.

This module provides tools for analyzing evaluation runs, generating reports,
and tracking trends over time.

Usage:
    from evalyn_sdk.analysis import (
        analyze_run,
        generate_html_report,
        generate_text_report,
        analyze_trends,
    )

    # Analyze a run
    run_data = load_eval_run("path/to/results.json")
    analysis = analyze_run(run_data)

    # Generate reports
    html = generate_html_report(analysis)
    text = generate_text_report(analysis)

    # Analyze trends
    trend = analyze_trends(runs)
"""

from .core import (
    MetricStats,
    ItemStats,
    RunAnalysis,
    load_eval_run,
    find_eval_runs,
    analyze_run,
)

from .reports import (
    ascii_bar,
    ascii_score_distribution,
    format_pass_rate_bar,
    generate_text_report,
    generate_comparison_report,
)

from .trends import (
    TrendAnalysis,
    analyze_trends,
    generate_trend_text_report,
)

from .html_report import (
    generate_html_report,
    generate_report,
)

__all__ = [
    # Core classes
    "MetricStats",
    "ItemStats",
    "RunAnalysis",
    # Core functions
    "load_eval_run",
    "find_eval_runs",
    "analyze_run",
    # Text reports
    "ascii_bar",
    "ascii_score_distribution",
    "format_pass_rate_bar",
    "generate_text_report",
    "generate_comparison_report",
    # Trends
    "TrendAnalysis",
    "analyze_trends",
    "generate_trend_text_report",
    # HTML reports
    "generate_html_report",
    "generate_report",
]
