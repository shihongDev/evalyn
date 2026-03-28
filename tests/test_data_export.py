"""Tests for analysis data export API."""

import pytest
from evalyn_sdk.analysis.core import RunAnalysis, MetricStats, ItemStats
from evalyn_sdk.analysis.data_export import (
    analyze_to_dict,
    metric_summary_to_dict,
    scores_matrix,
    export_flat_records,
)


def _make_analysis():
    """Build a minimal RunAnalysis for testing."""
    metric_stats = {
        "help": MetricStats(
            metric_id="help", metric_type="subjective",
            count=3, passed=2, failed=1, scores=[0.8, 0.9, 0.3],
            has_pass_fail=True,
        ),
        "safe": MetricStats(
            metric_id="safe", metric_type="objective",
            count=3, passed=3, failed=0, scores=[1.0, 1.0, 1.0],
            has_pass_fail=True,
        ),
    }
    item_stats = {
        "i1": ItemStats(
            item_id="i1", metrics_passed=2, metrics_failed=0,
            metric_results={
                "help": {"score": 0.8, "passed": True},
                "safe": {"score": 1.0, "passed": True},
            },
        ),
        "i2": ItemStats(
            item_id="i2", metrics_passed=2, metrics_failed=0,
            metric_results={
                "help": {"score": 0.9, "passed": True},
                "safe": {"score": 1.0, "passed": True},
            },
        ),
        "i3": ItemStats(
            item_id="i3", metrics_passed=1, metrics_failed=1,
            metric_results={
                "help": {"score": 0.3, "passed": False},
                "safe": {"score": 1.0, "passed": True},
            },
        ),
    }
    return RunAnalysis(
        run_id="r1",
        dataset_name="test",
        created_at="2026-01-01",
        total_items=3,
        total_metrics=2,
        metric_stats=metric_stats,
        item_stats=item_stats,
        failed_items=["i3"],
    )


# =============================================================================
# analyze_to_dict
# =============================================================================

class TestAnalyzeToDict:
    def test_columns_present(self):
        d = analyze_to_dict(_make_analysis())
        assert "item_id" in d
        assert "all_passed" in d
        assert "metrics_passed" in d
        assert "help_score" in d
        assert "help_passed" in d
        assert "safe_score" in d

    def test_equal_length_columns(self):
        d = analyze_to_dict(_make_analysis())
        lengths = {len(v) for v in d.values()}
        assert len(lengths) == 1
        assert 3 in lengths

    def test_item_order_sorted(self):
        d = analyze_to_dict(_make_analysis())
        assert d["item_id"] == ["i1", "i2", "i3"]

    def test_scores_correct(self):
        d = analyze_to_dict(_make_analysis())
        assert d["help_score"] == [0.8, 0.9, 0.3]
        assert d["safe_score"] == [1.0, 1.0, 1.0]

    def test_passed_correct(self):
        d = analyze_to_dict(_make_analysis())
        assert d["help_passed"] == [True, True, False]
        assert d["all_passed"] == [True, True, False]

    def test_empty_analysis(self):
        a = RunAnalysis(
            run_id="r", dataset_name="d", created_at="t",
            total_items=0, total_metrics=0,
            metric_stats={}, item_stats={}, failed_items=[],
        )
        d = analyze_to_dict(a)
        assert d["item_id"] == []

    def test_missing_metric_result_is_none(self):
        """If an item is missing a metric result, export as None."""
        a = _make_analysis()
        # Remove help from i1
        del a.item_stats["i1"].metric_results["help"]
        d = analyze_to_dict(a)
        # First item (i1) should have None for help_score
        assert d["help_score"][0] is None
        assert d["help_passed"][0] is None


# =============================================================================
# metric_summary_to_dict
# =============================================================================

class TestMetricSummaryToDict:
    def test_columns_present(self):
        d = metric_summary_to_dict(_make_analysis())
        for col in ["metric_id", "metric_type", "count", "passed", "failed",
                     "pass_rate", "avg_score", "min_score", "max_score", "std_dev"]:
            assert col in d

    def test_two_metrics(self):
        d = metric_summary_to_dict(_make_analysis())
        assert len(d["metric_id"]) == 2

    def test_sorted_by_id(self):
        d = metric_summary_to_dict(_make_analysis())
        assert d["metric_id"] == ["help", "safe"]

    def test_pass_rate_values(self):
        d = metric_summary_to_dict(_make_analysis())
        # help: 2/3, safe: 3/3
        help_idx = d["metric_id"].index("help")
        safe_idx = d["metric_id"].index("safe")
        assert d["pass_rate"][help_idx] == pytest.approx(2 / 3)
        assert d["pass_rate"][safe_idx] == pytest.approx(1.0)

    def test_avg_scores(self):
        d = metric_summary_to_dict(_make_analysis())
        help_idx = d["metric_id"].index("help")
        assert d["avg_score"][help_idx] == pytest.approx((0.8 + 0.9 + 0.3) / 3)


# =============================================================================
# scores_matrix
# =============================================================================

class TestScoresMatrix:
    def test_shape(self):
        m = scores_matrix(_make_analysis())
        assert len(m["item_id"]) == 3
        assert len(m["help"]) == 3
        assert len(m["safe"]) == 3

    def test_values(self):
        m = scores_matrix(_make_analysis())
        assert m["help"] == [0.8, 0.9, 0.3]
        assert m["safe"] == [1.0, 1.0, 1.0]

    def test_missing_is_none(self):
        a = _make_analysis()
        del a.item_stats["i2"].metric_results["help"]
        m = scores_matrix(a)
        assert m["help"][1] is None


# =============================================================================
# export_flat_records
# =============================================================================

class TestExportFlatRecords:
    def test_record_count(self):
        # 3 items x 2 metrics = 6 records
        records = export_flat_records(_make_analysis())
        assert len(records) == 6

    def test_record_fields(self):
        records = export_flat_records(_make_analysis())
        for r in records:
            assert "item_id" in r
            assert "metric_id" in r
            assert "score" in r
            assert "passed" in r

    def test_sorted_order(self):
        records = export_flat_records(_make_analysis())
        # Should be sorted by item_id, then metric_id
        assert records[0]["item_id"] == "i1"
        assert records[0]["metric_id"] == "help"
        assert records[1]["metric_id"] == "safe"

    def test_values(self):
        records = export_flat_records(_make_analysis())
        i1_help = [r for r in records if r["item_id"] == "i1" and r["metric_id"] == "help"][0]
        assert i1_help["score"] == 0.8
        assert i1_help["passed"] is True

    def test_empty_analysis(self):
        a = RunAnalysis(
            run_id="r", dataset_name="d", created_at="t",
            total_items=0, total_metrics=0,
            metric_stats={}, item_stats={}, failed_items=[],
        )
        assert export_flat_records(a) == []
