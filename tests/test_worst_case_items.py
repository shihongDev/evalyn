"""Tests for worst-case item identification."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.worst_case_items import (
    WorstCaseItem,
    WorstCaseReport,
    analyze_item_failures,
    build_worst_case_report,
    compute_failure_correlation,
    find_universally_failing,
    rank_worst_items,
    render_worst_case_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results(*specs):
    """Build item_results list from (item_id, metric_id, passed, score) tuples."""
    return [
        {"item_id": s[0], "metric_id": s[1], "passed": s[2], "score": s[3]}
        for s in specs
    ]


# ---------------------------------------------------------------------------
# analyze_item_failures
# ---------------------------------------------------------------------------

class TestAnalyzeItemFailures:
    def test_basic_mixed(self):
        results = _make_results(
            ("a", "m1", True, 0.9),
            ("a", "m2", False, 0.2),
            ("b", "m1", True, 0.8),
            ("b", "m2", True, 0.7),
        )
        items = analyze_item_failures(results)
        by_id = {i.item_id: i for i in items}
        assert len(by_id) == 2
        assert by_id["a"].failed_metrics == ["m2"]
        assert by_id["a"].total_metrics == 2
        assert by_id["b"].failed_metrics == []

    def test_all_pass(self):
        results = _make_results(
            ("a", "m1", True, 1.0),
            ("a", "m2", True, 0.9),
        )
        items = analyze_item_failures(results)
        assert items[0].failure_rate == 0.0
        assert items[0].failed_metrics == []

    def test_all_fail(self):
        results = _make_results(
            ("a", "m1", False, 0.1),
            ("a", "m2", False, 0.0),
        )
        items = analyze_item_failures(results)
        assert items[0].failure_rate == 1.0
        assert len(items[0].failed_metrics) == 2

    def test_single_item_single_metric(self):
        results = _make_results(("x", "m1", False, 0.3))
        items = analyze_item_failures(results)
        assert len(items) == 1
        assert items[0].item_id == "x"
        assert items[0].failure_rate == 1.0

    def test_avg_score_computed(self):
        results = _make_results(
            ("a", "m1", True, 0.8),
            ("a", "m2", False, 0.2),
        )
        items = analyze_item_failures(results)
        assert abs(items[0].avg_score - 0.5) < 1e-9

    def test_empty_input(self):
        items = analyze_item_failures([])
        assert items == []


# ---------------------------------------------------------------------------
# rank_worst_items
# ---------------------------------------------------------------------------

class TestRankWorstItems:
    def test_top_n(self):
        items = [
            WorstCaseItem(item_id="a", failure_rate=0.1),
            WorstCaseItem(item_id="b", failure_rate=0.9),
            WorstCaseItem(item_id="c", failure_rate=0.5),
        ]
        ranked = rank_worst_items(items, top_n=2)
        assert len(ranked) == 2
        assert ranked[0].item_id == "b"
        assert ranked[1].item_id == "c"

    def test_top_n_larger_than_list(self):
        items = [WorstCaseItem(item_id="a", failure_rate=0.5)]
        ranked = rank_worst_items(items, top_n=10)
        assert len(ranked) == 1

    def test_empty(self):
        assert rank_worst_items([], top_n=5) == []


# ---------------------------------------------------------------------------
# find_universally_failing
# ---------------------------------------------------------------------------

class TestFindUniversallyFailing:
    def test_some_universal(self):
        items = [
            WorstCaseItem(item_id="a", failed_metrics=["m1", "m2"], total_metrics=2),
            WorstCaseItem(item_id="b", failed_metrics=["m1"], total_metrics=2),
            WorstCaseItem(item_id="c", failed_metrics=["m1", "m2", "m3"], total_metrics=3),
        ]
        result = find_universally_failing(items)
        assert set(result) == {"a", "c"}

    def test_none_universal(self):
        items = [
            WorstCaseItem(item_id="a", failed_metrics=["m1"], total_metrics=2),
        ]
        assert find_universally_failing(items) == []

    def test_empty(self):
        assert find_universally_failing([]) == []

    def test_zero_total_metrics_excluded(self):
        items = [
            WorstCaseItem(item_id="a", failed_metrics=[], total_metrics=0),
        ]
        assert find_universally_failing(items) == []


# ---------------------------------------------------------------------------
# compute_failure_correlation
# ---------------------------------------------------------------------------

class TestComputeFailureCorrelation:
    def test_basic_pairs(self):
        items = [
            WorstCaseItem(item_id="a", failed_metrics=["m1", "m2"]),
            WorstCaseItem(item_id="b", failed_metrics=["m1", "m2"]),
            WorstCaseItem(item_id="c", failed_metrics=["m1"]),
        ]
        corr = compute_failure_correlation(items)
        assert corr["m1,m2"] == 2

    def test_three_metrics(self):
        items = [
            WorstCaseItem(item_id="a", failed_metrics=["m1", "m2", "m3"]),
        ]
        corr = compute_failure_correlation(items)
        assert corr["m1,m2"] == 1
        assert corr["m1,m3"] == 1
        assert corr["m2,m3"] == 1

    def test_single_failure_no_pairs(self):
        items = [
            WorstCaseItem(item_id="a", failed_metrics=["m1"]),
        ]
        corr = compute_failure_correlation(items)
        assert corr == {}

    def test_empty(self):
        assert compute_failure_correlation([]) == {}


# ---------------------------------------------------------------------------
# build_worst_case_report
# ---------------------------------------------------------------------------

class TestBuildWorstCaseReport:
    def test_basic(self):
        items = [
            WorstCaseItem(item_id="a", failed_metrics=["m1", "m2"], total_metrics=3, failure_rate=0.67),
            WorstCaseItem(item_id="b", failed_metrics=["m1"], total_metrics=3, failure_rate=0.33),
        ]
        report = build_worst_case_report(items)
        assert report.total_items == 2
        assert report.worst_item_id == "a"
        assert report.max_failures == 2
        assert abs(report.avg_failure_rate - 0.5) < 0.01

    def test_empty(self):
        report = build_worst_case_report([])
        assert report.total_items == 0
        assert report.worst_item_id == ""
        assert report.max_failures == 0


# ---------------------------------------------------------------------------
# render_worst_case_table
# ---------------------------------------------------------------------------

class TestRenderWorstCaseTable:
    def test_output_contains_header(self):
        items = [
            WorstCaseItem(item_id="item_1", failed_metrics=["m1"], total_metrics=2,
                          failure_rate=0.5, avg_score=0.6),
        ]
        table = render_worst_case_table(items)
        assert "Item ID" in table
        assert "item_1" in table

    def test_max_items_respected(self):
        items = [
            WorstCaseItem(item_id=f"i{i}", failure_rate=i / 20.0, total_metrics=2,
                          failed_metrics=[], avg_score=0.5)
            for i in range(20)
        ]
        table = render_worst_case_table(items, max_items=3)
        lines = [l for l in table.split("\n") if l.strip() and not l.startswith("-")]
        # header + 3 data rows
        assert len(lines) == 4

    def test_empty_items(self):
        table = render_worst_case_table([])
        assert "No items" in table


# ---------------------------------------------------------------------------
# WorstCaseReport format_text
# ---------------------------------------------------------------------------

class TestWorstCaseReportFormatText:
    def test_includes_summary(self):
        report = WorstCaseReport(
            items=[WorstCaseItem(item_id="a", failed_metrics=["m1"], total_metrics=2,
                                 failure_rate=0.5, avg_score=0.4)],
            total_items=1,
            worst_item_id="a",
            max_failures=1,
            avg_failure_rate=0.5,
        )
        text = report.format_text()
        assert "Worst-Case Items Report" in text
        assert "a" in text
        assert "50.0%" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------

class TestRoundTrips:
    def test_worst_case_item_as_dict(self):
        item = WorstCaseItem(
            item_id="x", failed_metrics=["m1"], total_metrics=2,
            failure_rate=0.5, avg_score=0.4,
        )
        d = item.as_dict()
        assert d["item_id"] == "x"
        assert d["failed_metrics"] == ["m1"]
        assert d["failure_rate"] == 0.5

    def test_worst_case_report_roundtrip(self):
        original = WorstCaseReport(
            items=[WorstCaseItem(item_id="a", failed_metrics=["m1", "m2"],
                                 total_metrics=3, failure_rate=0.67, avg_score=0.3)],
            total_items=1,
            worst_item_id="a",
            max_failures=2,
            avg_failure_rate=0.67,
        )
        d = original.as_dict()
        restored = WorstCaseReport.from_dict(d)
        assert restored.total_items == original.total_items
        assert restored.worst_item_id == original.worst_item_id
        assert restored.max_failures == original.max_failures
        assert len(restored.items) == 1
        assert restored.items[0].item_id == "a"
        assert restored.items[0].failed_metrics == ["m1", "m2"]

    def test_empty_report_roundtrip(self):
        original = WorstCaseReport()
        d = original.as_dict()
        restored = WorstCaseReport.from_dict(d)
        assert restored.total_items == 0
        assert restored.items == []
