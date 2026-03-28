"""Tests for improvement priority ranking."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.improvement_priority import (
    MetricROI,
    PriorityReport,
    compute_headroom,
    compute_priority_score,
    project_overall_lift,
    rank_improvements,
    render_priority_chart,
    suggest_improvements,
)


# ---------------------------------------------------------------------------
# compute_headroom
# ---------------------------------------------------------------------------


class TestComputeHeadroom:
    def test_headroom_zero_pass_rate(self):
        assert compute_headroom(0.0) == 1.0

    def test_headroom_half_pass_rate(self):
        assert compute_headroom(0.5) == 0.5

    def test_headroom_full_pass_rate(self):
        assert compute_headroom(1.0) == 0.0


# ---------------------------------------------------------------------------
# compute_priority_score
# ---------------------------------------------------------------------------


class TestComputePriorityScore:
    def test_basic_score(self):
        score = compute_priority_score(0.5, 1.0, 10)
        # 0.5 * 1.0 * log2(11) ~ 0.5 * 3.459 = 1.73
        assert score > 1.0

    def test_zero_headroom(self):
        assert compute_priority_score(0.0, 1.0, 100) == 0.0

    def test_zero_weight(self):
        assert compute_priority_score(0.5, 0.0, 100) == 0.0

    def test_higher_weight_higher_score(self):
        low = compute_priority_score(0.5, 1.0, 10)
        high = compute_priority_score(0.5, 3.0, 10)
        assert high > low

    def test_more_items_higher_score(self):
        few = compute_priority_score(0.5, 1.0, 2)
        many = compute_priority_score(0.5, 1.0, 100)
        assert many > few


# ---------------------------------------------------------------------------
# project_overall_lift
# ---------------------------------------------------------------------------


class TestProjectOverallLift:
    def test_single_metric(self):
        lift = project_overall_lift("a", {"a": 0.5})
        # With default improvement=0.5: headroom=0.5, gain=0.25
        assert abs(lift - 0.25) < 1e-6

    def test_no_improvement_for_perfect(self):
        lift = project_overall_lift("a", {"a": 1.0})
        assert lift == 0.0

    def test_missing_metric_returns_zero(self):
        lift = project_overall_lift("missing", {"a": 0.5})
        assert lift == 0.0

    def test_weighted_lift(self):
        rates = {"a": 0.5, "b": 0.9}
        weights = {"a": 2.0, "b": 1.0}
        lift = project_overall_lift("a", rates, weights)
        assert lift > 0.0

    def test_two_metrics_equal_weight(self):
        rates = {"a": 0.0, "b": 1.0}
        lift = project_overall_lift("a", rates)
        # headroom=1.0, improvement=0.5, so a goes 0->0.5
        # Overall: before = (0+1)/2 = 0.5, after = (0.5+1)/2 = 0.75
        assert abs(lift - 0.25) < 1e-6


# ---------------------------------------------------------------------------
# rank_improvements
# ---------------------------------------------------------------------------


class TestRankImprovements:
    def test_sorted_by_priority_score(self):
        rates = {"low": 0.9, "mid": 0.5, "high": 0.1}
        report = rank_improvements(rates)
        scores = [r.priority_score for r in report.rankings]
        assert scores == sorted(scores, reverse=True)

    def test_top_priority_is_highest_scored(self):
        rates = {"a": 0.9, "b": 0.1}
        report = rank_improvements(rates)
        assert report.top_priority == "b"

    def test_total_metrics_count(self):
        rates = {"a": 0.5, "b": 0.7, "c": 0.3}
        report = rank_improvements(rates)
        assert report.total_metrics == 3

    def test_max_projected_lift(self):
        rates = {"a": 0.0, "b": 1.0}
        report = rank_improvements(rates)
        assert report.max_projected_lift > 0.0

    def test_with_weights(self):
        rates = {"a": 0.5, "b": 0.5}
        weights = {"a": 10.0, "b": 1.0}
        report = rank_improvements(rates, weights=weights)
        assert report.rankings[0].metric_id == "a"

    def test_with_item_counts(self):
        rates = {"a": 0.5, "b": 0.5}
        item_counts = {"a": 1, "b": 1000}
        report = rank_improvements(rates, item_counts=item_counts)
        assert report.rankings[0].metric_id == "b"


# ---------------------------------------------------------------------------
# render_priority_chart
# ---------------------------------------------------------------------------


class TestRenderPriorityChart:
    def test_output_contains_metric_names(self):
        rates = {"alpha": 0.3, "beta": 0.7}
        report = rank_improvements(rates)
        chart = render_priority_chart(report)
        assert "alpha" in chart
        assert "beta" in chart

    def test_output_contains_header(self):
        report = rank_improvements({"a": 0.5})
        chart = render_priority_chart(report)
        assert "PRIORITY CHART" in chart

    def test_empty_report(self):
        report = PriorityReport()
        chart = render_priority_chart(report)
        assert "No metrics" in chart


# ---------------------------------------------------------------------------
# suggest_improvements
# ---------------------------------------------------------------------------


class TestSuggestImprovements:
    def test_max_items_respected(self):
        rates = {f"m{i}": 0.1 * i for i in range(10)}
        report = rank_improvements(rates)
        suggestions = suggest_improvements(report, max_suggestions=2)
        assert len(suggestions) == 2

    def test_default_max_three(self):
        rates = {f"m{i}": 0.1 * i for i in range(10)}
        report = rank_improvements(rates)
        suggestions = suggest_improvements(report)
        assert len(suggestions) == 3

    def test_fewer_metrics_than_max(self):
        report = rank_improvements({"a": 0.5})
        suggestions = suggest_improvements(report, max_suggestions=5)
        assert len(suggestions) == 1

    def test_suggestion_contains_metric_id(self):
        report = rank_improvements({"my_metric": 0.3})
        suggestions = suggest_improvements(report)
        assert "my_metric" in suggestions[0]


# ---------------------------------------------------------------------------
# PriorityReport format_text
# ---------------------------------------------------------------------------


class TestPriorityReportFormatText:
    def test_format_text_contains_header(self):
        report = rank_improvements({"a": 0.5, "b": 0.8})
        text = report.format_text()
        assert "IMPROVEMENT PRIORITY RANKING" in text

    def test_format_text_contains_metric_ids(self):
        report = rank_improvements({"alpha": 0.2, "beta": 0.9})
        text = report.format_text()
        assert "alpha" in text
        assert "beta" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_metric_roi_as_dict(self):
        roi = MetricROI(metric_id="x", current_pass_rate=0.7, weight=2.0, headroom=0.3)
        d = roi.as_dict()
        assert d["metric_id"] == "x"
        assert d["weight"] == 2.0

    def test_priority_report_roundtrip(self):
        report = rank_improvements({"a": 0.4, "b": 0.8})
        d = report.as_dict()
        restored = PriorityReport.from_dict(d)
        assert restored.top_priority == report.top_priority
        assert restored.total_metrics == report.total_metrics
        assert len(restored.rankings) == len(report.rankings)
        for orig, rest in zip(report.rankings, restored.rankings):
            assert orig.metric_id == rest.metric_id
            assert abs(orig.priority_score - rest.priority_score) < 1e-6

    def test_empty_report_roundtrip(self):
        report = PriorityReport()
        d = report.as_dict()
        restored = PriorityReport.from_dict(d)
        assert restored.rankings == []
        assert restored.top_priority == ""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_metric(self):
        report = rank_improvements({"only": 0.6})
        assert report.total_metrics == 1
        assert report.top_priority == "only"

    def test_all_perfect_metrics(self):
        rates = {"a": 1.0, "b": 1.0, "c": 1.0}
        report = rank_improvements(rates)
        for r in report.rankings:
            assert r.headroom == 0.0
            assert r.priority_score == 0.0
        assert report.max_projected_lift == 0.0

    def test_all_zero_pass_rate(self):
        rates = {"a": 0.0, "b": 0.0}
        report = rank_improvements(rates)
        for r in report.rankings:
            assert r.headroom == 1.0
