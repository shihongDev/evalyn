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

from .insights import (
    CorrelationResult,
    RegressionAlert,
    FeatureInsight,
    DistributionInsight,
    Recommendation,
    InsightsReport,
    compute_metric_correlations,
    detect_regressions,
    analyze_input_features,
    analyze_score_distributions,
    generate_recommendations,
)

from .clustering import (
    ReasonCluster,
    ClusteringResult,
    ReasonClusterer,
    generate_cluster_html,
    generate_cluster_text,
    # Failure clustering
    FailureCase,
    FailureCluster,
    FailureClusteringResult,
    generate_failure_cluster_html,
    generate_failure_cluster_text,
)

from .panel import (
    ExpertOpinion,
    PanelDiscussion,
    EXPERT_ROLES,
    prepare_panel_context,
    build_expert_prompt,
    build_moderator_prompt,
    parse_expert_response,
    create_api_client,
    run_expert_panel,
)

from .insights_dashboard import (
    generate_insights_html,
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
    # Clustering
    "ReasonCluster",
    "ClusteringResult",
    "ReasonClusterer",
    "generate_cluster_html",
    "generate_cluster_text",
    # Failure clustering
    "FailureCase",
    "FailureCluster",
    "FailureClusteringResult",
    "generate_failure_cluster_html",
    "generate_failure_cluster_text",
    # Insights
    "CorrelationResult",
    "RegressionAlert",
    "FeatureInsight",
    "DistributionInsight",
    "Recommendation",
    "InsightsReport",
    "compute_metric_correlations",
    "detect_regressions",
    "analyze_input_features",
    "analyze_score_distributions",
    "generate_recommendations",
    # Expert panel
    "ExpertOpinion",
    "PanelDiscussion",
    "EXPERT_ROLES",
    "prepare_panel_context",
    "build_expert_prompt",
    "build_moderator_prompt",
    "parse_expert_response",
    "create_api_client",
    "run_expert_panel",
    # Insights dashboard
    "generate_insights_html",
]
