"""Tests for the insights dashboard HTML generation."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from conftest import make_run_analysis
from evalyn_sdk.analysis.core import RunAnalysis, MetricStats, ItemStats
from evalyn_sdk.analysis.insights import (
    InsightsReport,
    CorrelationResult,
    RegressionAlert,
    DistributionInsight,
    FeatureInsight,
    Recommendation,
)
from evalyn_sdk.analysis.panel import ExpertOpinion, PanelDiscussion
from evalyn_sdk.analysis.insights_dashboard import generate_insights_html


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_analysis():
    items = {}
    for i in range(10):
        items[f"item_{i}"] = {
            "helpfulness": {
                "passed": i < 7,
                "score": 0.9 if i < 7 else 0.3,
            },
            "accuracy": {
                "passed": i < 8,
                "score": 0.85 if i < 8 else 0.2,
            },
            "relevance": {
                "passed": i < 9,
                "score": 0.95 if i < 9 else 0.1,
            },
        }
    return make_run_analysis(items, dataset_name="dashboard-test")


def _sample_report():
    return InsightsReport(
        correlations=[
            CorrelationResult(
                metric_a="helpfulness", metric_b="accuracy",
                pearson=0.82, relationship="redundant",
            ),
        ],
        regressions=[
            RegressionAlert(
                metric_id="helpfulness",
                previous_pass_rate=0.9,
                current_pass_rate=0.7,
                delta=-0.2,
                severity="critical",
            ),
        ],
        distribution_insights=[
            DistributionInsight(
                metric_id="helpfulness",
                shape="bimodal",
                finding="Split outcomes",
            ),
        ],
        feature_insights=[
            FeatureInsight(
                feature_name="input_length",
                finding="Longer inputs fail more",
                affected_items=10,
                pass_rate_low=0.9,
                pass_rate_high=0.5,
            ),
        ],
        recommendations=[
            Recommendation(
                priority=1,
                category="regression",
                message="Investigate helpfulness drop",
                action="evalyn analyze --latest",
            ),
            Recommendation(
                priority=2,
                category="metric_config",
                message="Consider removing redundant metric",
                action="evalyn list-metrics --latest",
            ),
        ],
    )


def _sample_panel():
    return PanelDiscussion(
        experts=[
            ExpertOpinion(
                role="quality_analyst",
                summary="Quality is declining on longer inputs",
                findings=["Helpfulness drops sharply", "Accuracy stable"],
                concerns=["Sample size small"],
                suggestions=["Add more test cases"],
                confidence=0.8,
            ),
            ExpertOpinion(
                role="strategist",
                summary="Focus on helpfulness metric first",
                findings=["Quick wins available"],
                concerns=[],
                suggestions=["Retrain on long inputs"],
                confidence=0.75,
            ),
        ],
        synthesis="The team should prioritize helpfulness improvements",
        action_plan=["Fix helpfulness metric", "Add regression tests"],
        dissenting_views=["Strategist questions metric validity"],
        total_input_tokens=1200,
        total_output_tokens=600,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateInsightsHtml:
    def test_returns_valid_html(self):
        analysis = _sample_analysis()
        report = InsightsReport()
        html = generate_insights_html(analysis, report)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html
        assert "chart.js" in html.lower()

    def test_contains_dataset_name(self):
        analysis = _sample_analysis()
        report = InsightsReport()
        html = generate_insights_html(analysis, report)
        assert "dashboard-test" in html

    def test_contains_kpi_bar(self):
        analysis = _sample_analysis()
        report = InsightsReport()
        html = generate_insights_html(analysis, report)
        assert "Total Items" in html
        assert "Overall Pass Rate" in html
        assert "Metrics" in html
        assert "Health Status" in html

    def test_contains_pass_rate_chart(self):
        analysis = _sample_analysis()
        report = InsightsReport()
        html = generate_insights_html(analysis, report)
        assert "passRateChart" in html
        assert "radarChart" in html

    def test_contains_distributions(self):
        analysis = _sample_analysis()
        report = InsightsReport()
        html = generate_insights_html(analysis, report)
        assert "Score Distributions" in html
        assert "histChart" in html

    def test_contains_heatmap(self):
        analysis = _sample_analysis()
        report = InsightsReport()
        html = generate_insights_html(analysis, report)
        assert "Item-Metric Matrix" in html
        assert "heatmap" in html

    def test_contains_correlation_matrix(self):
        analysis = _sample_analysis()
        report = _sample_report()
        html = generate_insights_html(analysis, report)
        assert "Correlation Matrix" in html

    def test_contains_regression_chart(self):
        analysis = _sample_analysis()
        report = _sample_report()
        html = generate_insights_html(analysis, report)
        assert "regressionChart" in html
        assert "Regression" in html

    def test_contains_recommendations(self):
        analysis = _sample_analysis()
        report = _sample_report()
        html = generate_insights_html(analysis, report)
        assert "Recommendations" in html
        assert "Investigate helpfulness" in html
        assert "regression" in html

    def test_contains_scatter_with_dataset_items(self):
        analysis = _sample_analysis()
        report = InsightsReport()
        dataset_items = [
            {"id": f"item_{i}", "input": "x" * (i * 10 + 10)} for i in range(10)
        ]
        html = generate_insights_html(analysis, report, dataset_items=dataset_items)
        assert "scatterChart" in html
        assert "Input Analysis" in html

    def test_no_scatter_without_dataset(self):
        analysis = _sample_analysis()
        report = InsightsReport()
        html = generate_insights_html(analysis, report, dataset_items=None)
        assert "Input Analysis" not in html

    def test_no_regression_section_without_regressions(self):
        analysis = _sample_analysis()
        report = InsightsReport()
        html = generate_insights_html(analysis, report)
        assert "Regression Analysis" not in html

    def test_no_recommendations_section_when_empty(self):
        analysis = _sample_analysis()
        report = InsightsReport()
        html = generate_insights_html(analysis, report)
        # No recommendations section title when list is empty
        assert "section-title\">Recommendations" not in html

    def test_expert_panel_rendering(self):
        analysis = _sample_analysis()
        report = _sample_report()
        panel = _sample_panel()
        html = generate_insights_html(analysis, report, panel_discussion=panel)
        assert "Expert Panel Analysis" in html
        assert "Quality Analyst" in html
        assert "Strategist" in html
        assert "Quality is declining" in html
        assert "prioritize helpfulness" in html
        assert "Action Plan" in html

    def test_no_expert_panel_when_none(self):
        analysis = _sample_analysis()
        report = InsightsReport()
        html = generate_insights_html(analysis, report, panel_discussion=None)
        # No expert panel section title when no panel
        assert "section-title\">Expert Panel" not in html

    def test_warm_theme_colors(self):
        analysis = _sample_analysis()
        report = InsightsReport()
        html = generate_insights_html(analysis, report)
        assert "#FFFBF7" in html  # bg-primary
        assert "#D4A27F" in html  # accent
        assert "#6B8E8E" in html  # status-pass

    def test_html_escaping(self):
        """Metric names with special chars should be escaped in HTML sections."""
        items = {
            "item_0": {
                '<b>bold</b>': {"passed": True, "score": 1.0},
            },
        }
        analysis = make_run_analysis(items)
        report = InsightsReport()
        html = generate_insights_html(analysis, report)
        # In the heatmap HTML section, the metric name should be escaped
        assert '&lt;b&gt;bold&lt;/b&gt;' in html

    def test_single_metric(self):
        """Dashboard should work with a single metric."""
        items = {f"item_{i}": {"m": {"passed": True, "score": 0.9}} for i in range(5)}
        analysis = make_run_analysis(items)
        report = InsightsReport()
        html = generate_insights_html(analysis, report)
        assert "<!DOCTYPE html>" in html
        # No correlation matrix with single metric
        assert "Correlation Matrix" not in html

    def test_empty_analysis(self):
        """Dashboard should handle empty analysis without crashing."""
        analysis = RunAnalysis(
            run_id="empty",
            dataset_name="empty",
            created_at="2026-01-01",
            total_items=0,
            total_metrics=0,
            metric_stats={},
            item_stats={},
            failed_items=[],
        )
        report = InsightsReport()
        html = generate_insights_html(analysis, report)
        assert "<!DOCTYPE html>" in html


class TestDashboardTokenInfo:
    def test_token_info_displayed(self):
        analysis = _sample_analysis()
        report = InsightsReport()
        panel = _sample_panel()
        html = generate_insights_html(analysis, report, panel_discussion=panel)
        assert "1200" in html  # input tokens
        assert "600" in html  # output tokens
