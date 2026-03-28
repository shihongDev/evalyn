"""Tests for unit_reporting module.

All tests use synthetic item dicts - no storage or ML deps required.
"""

from __future__ import annotations

import pytest

from evalyn_sdk.analysis.unit_reporting import (
    UnitReport,
    UnitTypeStats,
    classify_item_unit_type,
    compute_unit_breakdown,
    filter_by_unit_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_item(
    item_id: str = "item-1",
    unit_type: str = "outcome",
    metrics: dict | None = None,
) -> dict:
    return {
        "item_id": item_id,
        "unit_type": unit_type,
        "metrics": metrics or {},
    }


def make_metric(score: float = 1.0, passed: bool = True) -> dict:
    return {"score": score, "passed": passed}


# ---------------------------------------------------------------------------
# classify_item_unit_type
# ---------------------------------------------------------------------------


class TestClassifyItemUnitType:
    def test_explicit_unit_type(self):
        assert classify_item_unit_type({"unit_type": "tool_use"}) == "tool_use"

    def test_default_to_outcome(self):
        assert classify_item_unit_type({}) == "outcome"

    def test_non_string_coerced(self):
        assert classify_item_unit_type({"unit_type": 42}) == "42"

    def test_empty_string_unit_type(self):
        assert classify_item_unit_type({"unit_type": ""}) == ""


# ---------------------------------------------------------------------------
# compute_unit_breakdown - single type
# ---------------------------------------------------------------------------


class TestComputeUnitBreakdownSingleType:
    def test_single_item_all_passed(self):
        items = [make_item(metrics={"accuracy": make_metric(1.0, True)})]
        report = compute_unit_breakdown(items)
        assert report.total_items == 1
        assert len(report.unit_stats) == 1
        assert report.unit_stats[0].pass_rate == 1.0

    def test_single_item_failed(self):
        items = [make_item(metrics={"accuracy": make_metric(0.0, False)})]
        report = compute_unit_breakdown(items)
        assert report.unit_stats[0].pass_rate == 0.0

    def test_two_items_half_pass(self):
        items = [
            make_item(item_id="a", metrics={"m": make_metric(1.0, True)}),
            make_item(item_id="b", metrics={"m": make_metric(0.0, False)}),
        ]
        report = compute_unit_breakdown(items)
        assert report.unit_stats[0].pass_rate == pytest.approx(0.5)

    def test_avg_score(self):
        items = [
            make_item(item_id="a", metrics={"m": make_metric(0.8, True)}),
            make_item(item_id="b", metrics={"m": make_metric(0.4, False)}),
        ]
        report = compute_unit_breakdown(items)
        assert report.unit_stats[0].avg_score == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# compute_unit_breakdown - multiple types
# ---------------------------------------------------------------------------


class TestComputeUnitBreakdownMultipleTypes:
    def test_two_types(self):
        items = [
            make_item(item_id="a", unit_type="outcome", metrics={"m": make_metric(1.0, True)}),
            make_item(item_id="b", unit_type="tool_use", metrics={"m": make_metric(0.0, False)}),
        ]
        report = compute_unit_breakdown(items)
        assert report.total_items == 2
        assert len(report.unit_stats) == 2
        types = {s.unit_type for s in report.unit_stats}
        assert types == {"outcome", "tool_use"}

    def test_distribution_counts(self):
        items = [
            make_item(item_id="a", unit_type="outcome"),
            make_item(item_id="b", unit_type="outcome"),
            make_item(item_id="c", unit_type="single_turn"),
        ]
        report = compute_unit_breakdown(items)
        assert report.unit_type_distribution == {"outcome": 2, "single_turn": 1}

    def test_per_type_stats_independent(self):
        items = [
            make_item(item_id="a", unit_type="outcome", metrics={"m": make_metric(1.0, True)}),
            make_item(item_id="b", unit_type="tool_use", metrics={"m": make_metric(0.0, False)}),
        ]
        report = compute_unit_breakdown(items)
        by_type = {s.unit_type: s for s in report.unit_stats}
        assert by_type["outcome"].pass_rate == 1.0
        assert by_type["tool_use"].pass_rate == 0.0


# ---------------------------------------------------------------------------
# Per-type metric breakdown
# ---------------------------------------------------------------------------


class TestMetricBreakdown:
    def test_metric_breakdown_per_type(self):
        items = [
            make_item(
                item_id="a",
                unit_type="outcome",
                metrics={
                    "accuracy": make_metric(1.0, True),
                    "relevance": make_metric(0.5, False),
                },
            ),
            make_item(
                item_id="b",
                unit_type="outcome",
                metrics={
                    "accuracy": make_metric(1.0, True),
                    "relevance": make_metric(0.9, True),
                },
            ),
        ]
        report = compute_unit_breakdown(items)
        stats = report.unit_stats[0]
        assert stats.metric_breakdown["accuracy"] == 1.0
        assert stats.metric_breakdown["relevance"] == pytest.approx(0.5)

    def test_metric_breakdown_empty_metrics(self):
        items = [make_item(item_id="a", metrics={})]
        report = compute_unit_breakdown(items)
        assert report.unit_stats[0].metric_breakdown == {}


# ---------------------------------------------------------------------------
# filter_by_unit_type
# ---------------------------------------------------------------------------


class TestFilterByUnitType:
    def test_filters_correctly(self):
        items = [
            make_item(item_id="a", unit_type="outcome"),
            make_item(item_id="b", unit_type="tool_use"),
            make_item(item_id="c", unit_type="outcome"),
        ]
        filtered = filter_by_unit_type(items, "outcome")
        assert len(filtered) == 2
        assert all(i["unit_type"] == "outcome" for i in filtered)

    def test_no_match(self):
        items = [make_item(item_id="a", unit_type="outcome")]
        filtered = filter_by_unit_type(items, "tool_use")
        assert filtered == []

    def test_default_unit_type_filtering(self):
        # Item without explicit unit_type defaults to "outcome"
        item = {"item_id": "a", "metrics": {}}
        filtered = filter_by_unit_type([item], "outcome")
        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# UnitReport format_text
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_format_text_nonempty(self):
        items = [
            make_item(item_id="a", unit_type="outcome", metrics={"m": make_metric(1.0, True)}),
        ]
        report = compute_unit_breakdown(items)
        text = report.format_text()
        assert "1 total items" in text
        assert "outcome" in text
        assert "pass rate" in text

    def test_format_text_empty(self):
        report = compute_unit_breakdown([])
        text = report.format_text()
        assert "0 total items" in text
        assert "No unit data" in text

    def test_format_text_contains_metric_breakdown(self):
        items = [
            make_item(
                item_id="a",
                metrics={"accuracy": make_metric(1.0, True), "relevance": make_metric(0.0, False)},
            ),
        ]
        report = compute_unit_breakdown(items)
        text = report.format_text()
        assert "accuracy" in text
        assert "relevance" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_unit_type_stats_as_dict(self):
        stats = UnitTypeStats(
            unit_type="outcome",
            item_count=5,
            pass_rate=0.8,
            avg_score=0.75,
            metric_breakdown={"m1": 0.9},
        )
        d = stats.as_dict()
        assert d["unit_type"] == "outcome"
        assert d["item_count"] == 5
        assert d["metric_breakdown"] == {"m1": 0.9}

    def test_unit_report_roundtrip(self):
        items = [
            make_item(item_id="a", unit_type="outcome", metrics={"m": make_metric(0.8, True)}),
            make_item(item_id="b", unit_type="tool_use", metrics={"m": make_metric(0.3, False)}),
        ]
        report = compute_unit_breakdown(items)
        d = report.as_dict()
        restored = UnitReport.from_dict(d)
        assert restored.total_items == report.total_items
        assert len(restored.unit_stats) == len(report.unit_stats)
        assert restored.unit_type_distribution == report.unit_type_distribution

    def test_from_dict_empty(self):
        restored = UnitReport.from_dict({})
        assert restored.total_items == 0
        assert restored.unit_stats == []

    def test_roundtrip_preserves_metric_breakdown(self):
        stats = UnitTypeStats(
            unit_type="single_turn",
            item_count=3,
            pass_rate=0.667,
            avg_score=0.5,
            metric_breakdown={"accuracy": 1.0, "relevance": 0.333},
        )
        report = UnitReport(
            unit_stats=[stats],
            total_items=3,
            unit_type_distribution={"single_turn": 3},
        )
        d = report.as_dict()
        restored = UnitReport.from_dict(d)
        assert restored.unit_stats[0].metric_breakdown == {"accuracy": 1.0, "relevance": 0.333}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_items(self):
        report = compute_unit_breakdown([])
        assert report.total_items == 0
        assert report.unit_stats == []
        assert report.unit_type_distribution == {}

    def test_items_with_no_metrics(self):
        items = [make_item(item_id="a", metrics={})]
        report = compute_unit_breakdown(items)
        assert report.total_items == 1
        # No metrics means item is not counted as passed (metrics is empty)
        assert report.unit_stats[0].pass_rate == 0.0
        assert report.unit_stats[0].avg_score == 0.0

    def test_distribution_sums_to_total(self):
        items = [
            make_item(item_id=f"i{i}", unit_type=t)
            for i, t in enumerate(["outcome", "outcome", "tool_use", "single_turn"])
        ]
        report = compute_unit_breakdown(items)
        assert sum(report.unit_type_distribution.values()) == report.total_items

    def test_stats_sorted_by_unit_type(self):
        items = [
            make_item(item_id="a", unit_type="tool_use"),
            make_item(item_id="b", unit_type="outcome"),
            make_item(item_id="c", unit_type="single_turn"),
        ]
        report = compute_unit_breakdown(items)
        types = [s.unit_type for s in report.unit_stats]
        assert types == sorted(types)
