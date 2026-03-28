"""Tests for analysis report diff."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.report_diff import (
    MetricChange,
    ReportDiff,
    compute_metric_change,
    diff_reports,
    filter_changes,
    is_regression,
    render_diff_table,
    summarize_diff,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report(run_id, pass_rate, total_items, metric_stats):
    """Build a minimal report dict."""
    return {
        "run_id": run_id,
        "pass_rate": pass_rate,
        "total_items": total_items,
        "metric_stats": metric_stats,
    }


# ---------------------------------------------------------------------------
# compute_metric_change
# ---------------------------------------------------------------------------

class TestComputeMetricChange:
    def test_improved(self):
        c = compute_metric_change("m1", 0.5, 0.8)
        assert c.status == "improved"
        assert abs(c.delta - 0.3) < 1e-9

    def test_degraded(self):
        c = compute_metric_change("m1", 0.8, 0.5)
        assert c.status == "degraded"
        assert abs(c.delta - (-0.3)) < 1e-9

    def test_unchanged_within_threshold(self):
        c = compute_metric_change("m1", 0.5, 0.505, threshold=0.01)
        assert c.status == "unchanged"

    def test_added(self):
        c = compute_metric_change("m1", None, 0.9)
        assert c.status == "added"
        assert c.old_pass_rate is None
        assert c.new_pass_rate == 0.9

    def test_removed(self):
        c = compute_metric_change("m1", 0.9, None)
        assert c.status == "removed"
        assert c.old_pass_rate == 0.9
        assert c.new_pass_rate is None

    def test_both_none(self):
        c = compute_metric_change("m1", None, None)
        assert c.status == "unchanged"

    def test_exact_threshold_unchanged(self):
        # delta == threshold is not exceeded, so unchanged
        c = compute_metric_change("m1", 0.5, 0.509, threshold=0.01)
        assert c.status == "unchanged"

    def test_above_threshold_improved(self):
        c = compute_metric_change("m1", 0.5, 0.52, threshold=0.01)
        assert c.status == "improved"


# ---------------------------------------------------------------------------
# diff_reports
# ---------------------------------------------------------------------------

class TestDiffReports:
    def test_improved(self):
        a = _report("run-a", 0.6, 10, {"m1": {"pass_rate": 0.5}})
        b = _report("run-b", 0.8, 10, {"m1": {"pass_rate": 0.9}})
        diff = diff_reports(a, b)
        assert diff.run_a_id == "run-a"
        assert diff.run_b_id == "run-b"
        assert diff.improved_count == 1
        assert diff.degraded_count == 0
        assert abs(diff.pass_rate_delta - 0.2) < 1e-9

    def test_degraded(self):
        a = _report("run-a", 0.9, 10, {"m1": {"pass_rate": 0.9}})
        b = _report("run-b", 0.5, 10, {"m1": {"pass_rate": 0.5}})
        diff = diff_reports(a, b)
        assert diff.degraded_count == 1
        assert diff.improved_count == 0

    def test_added_metric(self):
        a = _report("a", 0.5, 10, {})
        b = _report("b", 0.5, 10, {"m_new": {"pass_rate": 0.7}})
        diff = diff_reports(a, b)
        assert any(c.status == "added" for c in diff.metric_changes)

    def test_removed_metric(self):
        a = _report("a", 0.5, 10, {"m_old": {"pass_rate": 0.7}})
        b = _report("b", 0.5, 10, {})
        diff = diff_reports(a, b)
        assert any(c.status == "removed" for c in diff.metric_changes)

    def test_identical_reports(self):
        stats = {"m1": {"pass_rate": 0.8}}
        a = _report("a", 0.8, 10, stats)
        b = _report("b", 0.8, 10, stats)
        diff = diff_reports(a, b)
        assert diff.improved_count == 0
        assert diff.degraded_count == 0
        assert all(c.status == "unchanged" for c in diff.metric_changes)

    def test_empty_reports(self):
        a = _report("a", 0.0, 0, {})
        b = _report("b", 0.0, 0, {})
        diff = diff_reports(a, b)
        assert diff.metric_changes == []
        assert diff.pass_rate_delta == 0.0

    def test_item_count_delta(self):
        a = _report("a", 0.5, 10, {})
        b = _report("b", 0.5, 15, {})
        diff = diff_reports(a, b)
        assert diff.item_count_delta == 5

    def test_multiple_metrics(self):
        a = _report("a", 0.5, 10, {
            "m1": {"pass_rate": 0.5},
            "m2": {"pass_rate": 0.9},
        })
        b = _report("b", 0.6, 10, {
            "m1": {"pass_rate": 0.8},
            "m2": {"pass_rate": 0.3},
        })
        diff = diff_reports(a, b)
        assert diff.improved_count == 1
        assert diff.degraded_count == 1


# ---------------------------------------------------------------------------
# filter_changes
# ---------------------------------------------------------------------------

class TestFilterChanges:
    def test_filter_improved(self):
        diff = ReportDiff(
            run_a_id="a", run_b_id="b",
            metric_changes=[
                MetricChange(metric_id="m1", status="improved"),
                MetricChange(metric_id="m2", status="degraded"),
                MetricChange(metric_id="m3", status="improved"),
            ],
        )
        result = filter_changes(diff, "improved")
        assert len(result) == 2
        assert all(c.status == "improved" for c in result)

    def test_filter_no_match(self):
        diff = ReportDiff(
            run_a_id="a", run_b_id="b",
            metric_changes=[
                MetricChange(metric_id="m1", status="unchanged"),
            ],
        )
        result = filter_changes(diff, "degraded")
        assert result == []

    def test_filter_empty_changes(self):
        diff = ReportDiff(run_a_id="a", run_b_id="b")
        assert filter_changes(diff, "improved") == []


# ---------------------------------------------------------------------------
# render_diff_table
# ---------------------------------------------------------------------------

class TestRenderDiffTable:
    def test_output_contains_header(self):
        diff = ReportDiff(
            run_a_id="a", run_b_id="b",
            metric_changes=[
                MetricChange(metric_id="m1", old_pass_rate=0.5, new_pass_rate=0.8,
                             delta=0.3, status="improved"),
            ],
        )
        table = render_diff_table(diff)
        assert "Metric" in table
        assert "m1" in table

    def test_positive_delta_has_plus(self):
        diff = ReportDiff(
            run_a_id="a", run_b_id="b",
            metric_changes=[
                MetricChange(metric_id="m1", old_pass_rate=0.5, new_pass_rate=0.8,
                             delta=0.3, status="improved"),
            ],
        )
        table = render_diff_table(diff)
        assert "+" in table

    def test_negative_delta_has_minus(self):
        diff = ReportDiff(
            run_a_id="a", run_b_id="b",
            metric_changes=[
                MetricChange(metric_id="m1", old_pass_rate=0.8, new_pass_rate=0.5,
                             delta=-0.3, status="degraded"),
            ],
        )
        table = render_diff_table(diff)
        assert "-" in table

    def test_empty_changes(self):
        diff = ReportDiff(run_a_id="a", run_b_id="b")
        table = render_diff_table(diff)
        assert "No metric changes" in table


# ---------------------------------------------------------------------------
# summarize_diff
# ---------------------------------------------------------------------------

class TestSummarizeDiff:
    def test_one_line(self):
        diff = ReportDiff(
            run_a_id="a", run_b_id="b",
            improved_count=3, degraded_count=1,
            metric_changes=[
                MetricChange(metric_id="m1", status="improved"),
                MetricChange(metric_id="m2", status="improved"),
                MetricChange(metric_id="m3", status="improved"),
                MetricChange(metric_id="m4", status="degraded"),
                MetricChange(metric_id="m5", status="unchanged"),
            ],
        )
        s = summarize_diff(diff)
        assert "3 improved" in s
        assert "1 degraded" in s
        assert "1 unchanged" in s

    def test_no_changes(self):
        diff = ReportDiff(run_a_id="a", run_b_id="b")
        assert summarize_diff(diff) == "no changes"

    def test_added_and_removed(self):
        diff = ReportDiff(
            run_a_id="a", run_b_id="b",
            metric_changes=[
                MetricChange(metric_id="m1", status="added"),
                MetricChange(metric_id="m2", status="removed"),
            ],
        )
        s = summarize_diff(diff)
        assert "1 added" in s
        assert "1 removed" in s


# ---------------------------------------------------------------------------
# is_regression
# ---------------------------------------------------------------------------

class TestIsRegression:
    def test_regression_true(self):
        diff = ReportDiff(run_a_id="a", run_b_id="b", pass_rate_delta=-0.10)
        assert is_regression(diff, threshold=0.05) is True

    def test_regression_false(self):
        diff = ReportDiff(run_a_id="a", run_b_id="b", pass_rate_delta=-0.02)
        assert is_regression(diff, threshold=0.05) is False

    def test_improvement_not_regression(self):
        diff = ReportDiff(run_a_id="a", run_b_id="b", pass_rate_delta=0.10)
        assert is_regression(diff) is False

    def test_exact_threshold_not_regression(self):
        diff = ReportDiff(run_a_id="a", run_b_id="b", pass_rate_delta=-0.05)
        assert is_regression(diff, threshold=0.05) is False


# ---------------------------------------------------------------------------
# ReportDiff format_text
# ---------------------------------------------------------------------------

class TestReportDiffFormatText:
    def test_includes_run_ids(self):
        diff = ReportDiff(
            run_a_id="run-1", run_b_id="run-2",
            pass_rate_delta=0.1, item_count_delta=5,
            improved_count=1, degraded_count=0,
            metric_changes=[
                MetricChange(metric_id="m1", old_pass_rate=0.5, new_pass_rate=0.6,
                             delta=0.1, status="improved"),
            ],
        )
        text = diff.format_text()
        assert "run-1" in text
        assert "run-2" in text
        assert "m1" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------

class TestRoundTrips:
    def test_metric_change_as_dict(self):
        c = MetricChange(metric_id="m1", old_pass_rate=0.5, new_pass_rate=0.8,
                         delta=0.3, status="improved")
        d = c.as_dict()
        assert d["metric_id"] == "m1"
        assert d["delta"] == 0.3

    def test_report_diff_roundtrip(self):
        original = ReportDiff(
            run_a_id="a", run_b_id="b",
            metric_changes=[
                MetricChange(metric_id="m1", old_pass_rate=0.5, new_pass_rate=0.8,
                             delta=0.3, status="improved"),
                MetricChange(metric_id="m2", old_pass_rate=None, new_pass_rate=0.9,
                             delta=0.0, status="added"),
            ],
            pass_rate_delta=0.15,
            item_count_delta=3,
            improved_count=1,
            degraded_count=0,
        )
        d = original.as_dict()
        restored = ReportDiff.from_dict(d)
        assert restored.run_a_id == "a"
        assert restored.run_b_id == "b"
        assert len(restored.metric_changes) == 2
        assert restored.metric_changes[0].status == "improved"
        assert restored.metric_changes[1].status == "added"
        assert restored.pass_rate_delta == 0.15
        assert restored.improved_count == 1

    def test_empty_diff_roundtrip(self):
        original = ReportDiff(run_a_id="", run_b_id="")
        d = original.as_dict()
        restored = ReportDiff.from_dict(d)
        assert restored.metric_changes == []
        assert restored.pass_rate_delta == 0.0
