"""Tests for metric contribution analysis module."""

import pytest
from evalyn_sdk.analysis.metric_contribution import (
    ContributionReport,
    MetricContribution,
    compute_all_contributions,
    compute_marginal_contribution,
    identify_bottleneck_metrics,
    rank_by_contribution,
    render_contribution_chart,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(item_id, metrics_passed, overall_passed):
    """Build an item_result dict.

    metrics_passed: dict of metric_id -> bool
    """
    return {
        "item_id": item_id,
        "metrics": {mid: {"passed": p} for mid, p in metrics_passed.items()},
        "overall_passed": overall_passed,
    }


# ---------------------------------------------------------------------------
# compute_marginal_contribution - positive
# ---------------------------------------------------------------------------


class TestMarginalContributionPositive:
    def test_metric_helps_pass(self):
        # Two metrics: A passes, B fails. Overall passes.
        # Removing A: only B (fail) -> overall fail. A is positive.
        items = [
            _make_item("i1", {"A": True, "B": False}, True),
        ]
        c = compute_marginal_contribution("A", items)
        assert c.contribution_score > 0
        assert c.direction == "positive"
        assert c.items_influenced >= 1

    def test_metric_not_deciding(self):
        # All metrics pass. Removing any one still passes.
        items = [
            _make_item("i1", {"A": True, "B": True, "C": True}, True),
        ]
        c = compute_marginal_contribution("A", items)
        assert c.contribution_score == 0.0
        assert c.direction == "neutral"

    def test_empty_items(self):
        c = compute_marginal_contribution("A", [])
        assert c.contribution_score == 0.0
        assert c.direction == "neutral"
        assert c.items_influenced == 0


# ---------------------------------------------------------------------------
# compute_marginal_contribution - negative
# ---------------------------------------------------------------------------


class TestMarginalContributionNegative:
    def test_metric_causes_failure(self):
        # Two metrics: A fails, B passes. Overall fails.
        # Removing A: only B (pass) -> overall pass. A is negative.
        items = [
            _make_item("i1", {"A": False, "B": True}, False),
        ]
        c = compute_marginal_contribution("A", items)
        assert c.contribution_score < 0
        assert c.direction == "negative"
        assert c.items_influenced >= 1

    def test_multiple_items_mixed(self):
        items = [
            _make_item("i1", {"A": False, "B": True}, False),
            _make_item("i2", {"A": True, "B": True}, True),
            _make_item("i3", {"A": False, "B": True}, False),
        ]
        c = compute_marginal_contribution("A", items)
        # Items 1 and 3: removing A flips fail->pass (negative influence)
        # Item 2: removing A, only B passes -> still pass (no influence)
        assert c.contribution_score < 0
        assert c.direction == "negative"


# ---------------------------------------------------------------------------
# compute_all_contributions
# ---------------------------------------------------------------------------


class TestComputeAllContributions:
    def test_basic(self):
        items = [
            _make_item("i1", {"A": True, "B": False}, False),
            _make_item("i2", {"A": True, "B": True}, True),
        ]
        report = compute_all_contributions(items, ["A", "B"])
        assert report.total_items == 2
        assert len(report.contributions) == 2

    def test_most_positive_set(self):
        items = [
            _make_item("i1", {"A": True, "B": False}, True),
        ]
        report = compute_all_contributions(items, ["A", "B"])
        # A helps pass, B does not
        assert isinstance(report.most_positive, str)

    def test_most_negative_set(self):
        items = [
            _make_item("i1", {"A": False, "B": True}, False),
        ]
        report = compute_all_contributions(items, ["A", "B"])
        assert report.most_negative == "A"

    def test_sorted_metric_ids(self):
        items = [_make_item("i1", {"Z": True, "A": True}, True)]
        report = compute_all_contributions(items, ["Z", "A"])
        assert report.contributions[0].metric_id == "A"
        assert report.contributions[1].metric_id == "Z"

    def test_empty_items(self):
        report = compute_all_contributions([], ["A", "B"])
        assert report.total_items == 0
        assert len(report.contributions) == 2


# ---------------------------------------------------------------------------
# rank_by_contribution
# ---------------------------------------------------------------------------


class TestRankByContribution:
    def test_sorted_descending_abs(self):
        report = ContributionReport(
            contributions=[
                MetricContribution("A", 0.1, "positive"),
                MetricContribution("B", -0.5, "negative"),
                MetricContribution("C", 0.3, "positive"),
            ]
        )
        ranked = rank_by_contribution(report)
        assert ranked[0].metric_id == "B"  # abs 0.5
        assert ranked[1].metric_id == "C"  # abs 0.3
        assert ranked[2].metric_id == "A"  # abs 0.1

    def test_empty(self):
        report = ContributionReport()
        assert rank_by_contribution(report) == []

    def test_single_metric(self):
        report = ContributionReport(
            contributions=[MetricContribution("X", 0.8, "positive")]
        )
        ranked = rank_by_contribution(report)
        assert len(ranked) == 1
        assert ranked[0].metric_id == "X"


# ---------------------------------------------------------------------------
# identify_bottleneck_metrics
# ---------------------------------------------------------------------------


class TestIdentifyBottleneckMetrics:
    def test_finds_bottlenecks(self):
        report = ContributionReport(
            contributions=[
                MetricContribution("A", -0.5, "negative"),
                MetricContribution("B", 0.3, "positive"),
                MetricContribution("C", -0.35, "negative"),
            ]
        )
        bottlenecks = identify_bottleneck_metrics(report, threshold=0.3)
        assert "A" in bottlenecks
        assert "C" in bottlenecks
        assert "B" not in bottlenecks

    def test_no_bottlenecks(self):
        report = ContributionReport(
            contributions=[
                MetricContribution("A", 0.5, "positive"),
                MetricContribution("B", 0.1, "positive"),
            ]
        )
        assert identify_bottleneck_metrics(report) == []

    def test_threshold_boundary(self):
        report = ContributionReport(
            contributions=[MetricContribution("A", -0.3, "negative")]
        )
        # Exactly at threshold: -0.3 <= -0.3 is True
        assert identify_bottleneck_metrics(report, threshold=0.3) == ["A"]

    def test_custom_threshold(self):
        report = ContributionReport(
            contributions=[MetricContribution("A", -0.15, "negative")]
        )
        assert identify_bottleneck_metrics(report, threshold=0.1) == ["A"]
        assert identify_bottleneck_metrics(report, threshold=0.2) == []


# ---------------------------------------------------------------------------
# render_contribution_chart
# ---------------------------------------------------------------------------


class TestRenderContributionChart:
    def test_output_is_string(self):
        report = ContributionReport(
            contributions=[MetricContribution("A", 0.5, "positive")]
        )
        chart = render_contribution_chart(report)
        assert isinstance(chart, str)
        assert "A" in chart

    def test_contains_header(self):
        report = ContributionReport(
            contributions=[MetricContribution("A", 0.5, "positive")]
        )
        chart = render_contribution_chart(report)
        assert "Metric Contributions" in chart

    def test_positive_and_negative(self):
        report = ContributionReport(
            contributions=[
                MetricContribution("pos", 0.4, "positive"),
                MetricContribution("neg", -0.3, "negative"),
            ]
        )
        chart = render_contribution_chart(report)
        assert "+0.400" in chart
        assert "-0.300" in chart

    def test_empty_report(self):
        report = ContributionReport()
        chart = render_contribution_chart(report)
        assert "No contributions" in chart

    def test_custom_width(self):
        report = ContributionReport(
            contributions=[MetricContribution("X", 0.5, "positive")]
        )
        chart = render_contribution_chart(report, width=40)
        assert isinstance(chart, str)


# ---------------------------------------------------------------------------
# ContributionReport.format_text
# ---------------------------------------------------------------------------


class TestContributionReportFormatText:
    def test_contains_header(self):
        report = ContributionReport(total_items=10)
        text = report.format_text()
        assert "Metric Contribution Analysis" in text

    def test_shows_total_items(self):
        report = ContributionReport(total_items=42)
        text = report.format_text()
        assert "42" in text

    def test_shows_most_positive_negative(self):
        report = ContributionReport(
            contributions=[
                MetricContribution("A", 0.5, "positive", 3),
                MetricContribution("B", -0.3, "negative", 2),
            ],
            most_positive="A",
            most_negative="B",
            total_items=5,
        )
        text = report.format_text()
        assert "Most positive contributor: A" in text
        assert "Most negative contributor: B" in text

    def test_shows_items_influenced(self):
        report = ContributionReport(
            contributions=[MetricContribution("A", 0.5, "positive", 7)],
            total_items=10,
        )
        text = report.format_text()
        assert "7 items influenced" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_metric_contribution_as_dict(self):
        c = MetricContribution("acc", 0.75, "positive", 5)
        d = c.as_dict()
        assert d["metric_id"] == "acc"
        assert d["contribution_score"] == 0.75
        assert d["direction"] == "positive"
        assert d["items_influenced"] == 5

    def test_metric_contribution_from_dict(self):
        d = {
            "metric_id": "lat",
            "contribution_score": -0.3,
            "direction": "negative",
            "items_influenced": 2,
        }
        c = MetricContribution.from_dict(d)
        assert c.metric_id == "lat"
        assert c.contribution_score == -0.3
        assert c.direction == "negative"

    def test_metric_contribution_roundtrip(self):
        c = MetricContribution("m", 0.42, "positive", 10)
        c2 = MetricContribution.from_dict(c.as_dict())
        assert c2.metric_id == c.metric_id
        assert c2.contribution_score == c.contribution_score
        assert c2.direction == c.direction
        assert c2.items_influenced == c.items_influenced

    def test_contribution_report_roundtrip(self):
        report = ContributionReport(
            contributions=[
                MetricContribution("A", 0.5, "positive", 3),
                MetricContribution("B", -0.2, "negative", 1),
            ],
            most_positive="A",
            most_negative="B",
            total_items=10,
        )
        d = report.as_dict()
        report2 = ContributionReport.from_dict(d)
        assert report2.total_items == 10
        assert report2.most_positive == "A"
        assert report2.most_negative == "B"
        assert len(report2.contributions) == 2

    def test_contribution_report_as_dict_keys(self):
        report = ContributionReport()
        d = report.as_dict()
        expected = {"contributions", "most_positive", "most_negative", "total_items"}
        assert set(d.keys()) == expected


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_metric_all_pass(self):
        items = [
            _make_item("i1", {"A": True}, True),
            _make_item("i2", {"A": True}, True),
        ]
        report = compute_all_contributions(items, ["A"])
        assert len(report.contributions) == 1

    def test_single_metric_all_fail(self):
        items = [
            _make_item("i1", {"A": False}, False),
            _make_item("i2", {"A": False}, False),
        ]
        report = compute_all_contributions(items, ["A"])
        c = report.contributions[0]
        # Removing the only metric: no remaining metrics -> default pass.
        # Overall was fail, without A it passes -> negative contribution.
        assert c.direction == "negative"

    def test_all_items_pass_all_metrics(self):
        items = [
            _make_item("i1", {"A": True, "B": True}, True),
            _make_item("i2", {"A": True, "B": True}, True),
        ]
        report = compute_all_contributions(items, ["A", "B"])
        for c in report.contributions:
            assert c.contribution_score == 0.0
            assert c.direction == "neutral"

    def test_many_metrics(self):
        metrics = {f"m{i}": True for i in range(10)}
        items = [_make_item("i1", metrics, True)]
        report = compute_all_contributions(items, list(metrics.keys()))
        assert len(report.contributions) == 10

    def test_no_metrics_in_item(self):
        items = [{"item_id": "i1", "metrics": {}, "overall_passed": True}]
        c = compute_marginal_contribution("A", items)
        assert c.contribution_score == 0.0
