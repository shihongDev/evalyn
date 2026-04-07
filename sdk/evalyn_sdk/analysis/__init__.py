"""
Analysis module for comprehensive eval results analysis and visualization.

Submodules are loaded lazily to avoid importing calibration, clustering,
and panel infrastructure when only core analysis functions are needed.
"""

from __future__ import annotations

import importlib as _importlib

# ---- Lazy import map (PEP 562) ----
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # core
    "MetricStats": (".core", "MetricStats"),
    "ItemStats": (".core", "ItemStats"),
    "RunAnalysis": (".core", "RunAnalysis"),
    "load_eval_run": (".core", "load_eval_run"),
    "find_eval_runs": (".core", "find_eval_runs"),
    "analyze_run": (".core", "analyze_run"),
    # reports
    "ascii_bar": (".reports", "ascii_bar"),
    "ascii_score_distribution": (".reports", "ascii_score_distribution"),
    "format_pass_rate_bar": (".reports", "format_pass_rate_bar"),
    "generate_text_report": (".reports", "generate_text_report"),
    "generate_comparison_report": (".reports", "generate_comparison_report"),
    # trends
    "TrendAnalysis": (".trends", "TrendAnalysis"),
    "analyze_trends": (".trends", "analyze_trends"),
    "generate_trend_text_report": (".trends", "generate_trend_text_report"),
    # html_report
    "generate_html_report": (".html_report", "generate_html_report"),
    "generate_report": (".html_report", "generate_report"),
    # insights
    "CorrelationResult": (".insights", "CorrelationResult"),
    "RegressionAlert": (".insights", "RegressionAlert"),
    "FeatureInsight": (".insights", "FeatureInsight"),
    "DistributionInsight": (".insights", "DistributionInsight"),
    "Recommendation": (".insights", "Recommendation"),
    "InsightsReport": (".insights", "InsightsReport"),
    "compute_metric_correlations": (".insights", "compute_metric_correlations"),
    "detect_regressions": (".insights", "detect_regressions"),
    "analyze_input_features": (".insights", "analyze_input_features"),
    "analyze_score_distributions": (".insights", "analyze_score_distributions"),
    "generate_recommendations": (".insights", "generate_recommendations"),
    # clustering
    "ReasonCluster": (".clustering", "ReasonCluster"),
    "ClusteringResult": (".clustering", "ClusteringResult"),
    "ReasonClusterer": (".clustering", "ReasonClusterer"),
    "generate_cluster_html": (".clustering", "generate_cluster_html"),
    "generate_cluster_text": (".clustering", "generate_cluster_text"),
    "FailureCase": (".clustering", "FailureCase"),
    "FailureCluster": (".clustering", "FailureCluster"),
    "FailureClusteringResult": (".clustering", "FailureClusteringResult"),
    "generate_failure_cluster_html": (".clustering", "generate_failure_cluster_html"),
    "generate_failure_cluster_text": (".clustering", "generate_failure_cluster_text"),
    # panel
    "ExpertOpinion": (".panel", "ExpertOpinion"),
    "PanelDiscussion": (".panel", "PanelDiscussion"),
    "EXPERT_ROLES": (".panel", "EXPERT_ROLES"),
    "prepare_panel_context": (".panel", "prepare_panel_context"),
    "build_expert_prompt": (".panel", "build_expert_prompt"),
    "build_moderator_prompt": (".panel", "build_moderator_prompt"),
    "parse_expert_response": (".panel", "parse_expert_response"),
    "create_api_client": (".panel", "create_api_client"),
    "run_expert_panel": (".panel", "run_expert_panel"),
    # insights_dashboard
    "generate_insights_html": (".insights_dashboard", "generate_insights_html"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())
