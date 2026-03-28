"""Tests for span type distribution statistics."""

import pytest
from datetime import datetime, timezone
from evalyn_sdk.models import Span
from evalyn_sdk.analysis.span_distribution import (
    TypeCount,
    DistributionReport,
    compute_distribution,
    compute_distribution_trend,
    detect_distribution_shift,
)


def _span(sid="s1", span_type="custom", parent_id=None, **attrs):
    return Span(
        id=sid, name="test", span_type=span_type,
        parent_id=parent_id, start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# =============================================================================
# TypeCount
# =============================================================================

class TestTypeCount:
    def test_as_dict(self):
        tc = TypeCount(span_type="llm_call", count=5, percentage=0.5)
        d = tc.as_dict()
        assert d["span_type"] == "llm_call"
        assert d["count"] == 5
        assert d["percentage"] == 0.5

    def test_percentage_rounding(self):
        tc = TypeCount(span_type="x", count=1, percentage=0.33333333)
        d = tc.as_dict()
        assert d["percentage"] == 0.3333


# =============================================================================
# DistributionReport
# =============================================================================

class TestDistributionReport:
    def test_as_dict(self):
        r = DistributionReport(
            counts=[TypeCount("llm_call", 3, 0.6), TypeCount("tool_call", 2, 0.4)],
            total_spans=5,
            unique_types=2,
        )
        d = r.as_dict()
        assert d["total_spans"] == 5
        assert d["unique_types"] == 2
        assert len(d["counts"]) == 2

    def test_from_dict(self):
        data = {
            "counts": [
                {"span_type": "llm_call", "count": 3, "percentage": 0.6},
                {"span_type": "tool_call", "count": 2, "percentage": 0.4},
            ],
            "total_spans": 5,
            "unique_types": 2,
        }
        r = DistributionReport.from_dict(data)
        assert r.total_spans == 5
        assert len(r.counts) == 2
        assert r.counts[0].span_type == "llm_call"

    def test_roundtrip(self):
        original = DistributionReport(
            counts=[TypeCount("a", 10, 0.5), TypeCount("b", 10, 0.5)],
            total_spans=20,
            unique_types=2,
        )
        restored = DistributionReport.from_dict(original.as_dict())
        assert restored.total_spans == original.total_spans
        assert restored.unique_types == original.unique_types
        assert len(restored.counts) == len(original.counts)
        assert restored.counts[0].span_type == original.counts[0].span_type

    def test_format_text(self):
        r = DistributionReport(
            counts=[TypeCount("llm_call", 3, 0.75), TypeCount("tool_call", 1, 0.25)],
            total_spans=4,
            unique_types=2,
        )
        text = r.format_text()
        assert "4 spans" in text
        assert "2 types" in text
        assert "llm_call" in text
        assert "75.0%" in text

    def test_format_text_empty(self):
        r = DistributionReport()
        text = r.format_text()
        assert "0 spans" in text


# =============================================================================
# compute_distribution
# =============================================================================

class TestComputeDistribution:
    def test_empty(self):
        r = compute_distribution([])
        assert r.total_spans == 0
        assert r.unique_types == 0
        assert r.counts == []

    def test_single_type(self):
        spans = [_span(sid=f"s{i}", span_type="llm_call") for i in range(3)]
        r = compute_distribution(spans)
        assert r.total_spans == 3
        assert r.unique_types == 1
        assert r.counts[0].span_type == "llm_call"
        assert r.counts[0].percentage == 1.0

    def test_multiple_types(self):
        spans = [
            _span(sid="s1", span_type="llm_call"),
            _span(sid="s2", span_type="llm_call"),
            _span(sid="s3", span_type="tool_call"),
            _span(sid="s4", span_type="retrieval"),
        ]
        r = compute_distribution(spans)
        assert r.total_spans == 4
        assert r.unique_types == 3

    def test_percentages_sum_to_one(self):
        spans = [
            _span(sid="s1", span_type="llm_call"),
            _span(sid="s2", span_type="tool_call"),
            _span(sid="s3", span_type="retrieval"),
            _span(sid="s4", span_type="llm_call"),
            _span(sid="s5", span_type="custom"),
        ]
        r = compute_distribution(spans)
        total_pct = sum(tc.percentage for tc in r.counts)
        assert abs(total_pct - 1.0) < 1e-9

    def test_sorted_by_count_descending(self):
        spans = [
            _span(sid="s1", span_type="a"),
            _span(sid="s2", span_type="b"),
            _span(sid="s3", span_type="b"),
            _span(sid="s4", span_type="c"),
            _span(sid="s5", span_type="c"),
            _span(sid="s6", span_type="c"),
        ]
        r = compute_distribution(spans)
        counts = [tc.count for tc in r.counts]
        assert counts == sorted(counts, reverse=True)
        assert r.counts[0].span_type == "c"


# =============================================================================
# compute_distribution_trend
# =============================================================================

class TestComputeDistributionTrend:
    def test_single_period(self):
        spans = [
            _span(sid="s1", span_type="llm_call"),
            _span(sid="s2", span_type="tool_call"),
        ]
        trend = compute_distribution_trend([("week1", spans)])
        assert "llm_call" in trend
        assert "tool_call" in trend
        assert trend["llm_call"][0] == ("week1", 0.5)

    def test_multiple_periods(self):
        w1 = [_span(sid="a1", span_type="llm_call"), _span(sid="a2", span_type="tool_call")]
        w2 = [_span(sid="b1", span_type="llm_call"), _span(sid="b2", span_type="llm_call")]
        trend = compute_distribution_trend([("w1", w1), ("w2", w2)])
        assert len(trend["llm_call"]) == 2
        assert trend["llm_call"][0][1] == 0.5  # w1: 50%
        assert trend["llm_call"][1][1] == 1.0  # w2: 100%

    def test_type_missing_in_period_gets_zero(self):
        w1 = [_span(sid="a1", span_type="llm_call"), _span(sid="a2", span_type="tool_call")]
        w2 = [_span(sid="b1", span_type="llm_call")]
        trend = compute_distribution_trend([("w1", w1), ("w2", w2)])
        # tool_call should have 0.0 in w2
        tool_entries = trend["tool_call"]
        assert tool_entries[1] == ("w2", 0.0)

    def test_empty_groups(self):
        trend = compute_distribution_trend([])
        assert trend == {}


# =============================================================================
# detect_distribution_shift
# =============================================================================

class TestDetectDistributionShift:
    def test_detects_shift(self):
        before = [_span(sid=f"b{i}", span_type="llm_call") for i in range(8)] + \
                 [_span(sid=f"b{i+8}", span_type="tool_call") for i in range(2)]
        after = [_span(sid=f"a{i}", span_type="llm_call") for i in range(4)] + \
                [_span(sid=f"a{i+4}", span_type="tool_call") for i in range(6)]
        shifts = detect_distribution_shift(before, after, threshold=0.1)
        assert len(shifts) >= 1
        types_shifted = {s["span_type"] for s in shifts}
        assert "llm_call" in types_shifted or "tool_call" in types_shifted

    def test_no_shift_below_threshold(self):
        # Same distribution
        spans = [_span(sid=f"s{i}", span_type="llm_call") for i in range(5)]
        shifts = detect_distribution_shift(spans, spans, threshold=0.1)
        assert shifts == []

    def test_shift_includes_delta(self):
        before = [_span(sid="b1", span_type="a")]
        after = [_span(sid="a1", span_type="b")]
        shifts = detect_distribution_shift(before, after, threshold=0.0)
        for s in shifts:
            assert "delta" in s
            assert "before_pct" in s
            assert "after_pct" in s

    def test_new_type_appears(self):
        before = [_span(sid="b1", span_type="llm_call")]
        after = [_span(sid="a1", span_type="llm_call"), _span(sid="a2", span_type="new_type")]
        shifts = detect_distribution_shift(before, after, threshold=0.1)
        types_shifted = {s["span_type"] for s in shifts}
        assert "new_type" in types_shifted

    def test_type_disappears(self):
        before = [_span(sid="b1", span_type="llm_call"), _span(sid="b2", span_type="old_type")]
        after = [_span(sid="a1", span_type="llm_call")]
        shifts = detect_distribution_shift(before, after, threshold=0.1)
        types_shifted = {s["span_type"] for s in shifts}
        assert "old_type" in types_shifted

    def test_empty_before(self):
        after = [_span(sid="a1", span_type="llm_call")]
        shifts = detect_distribution_shift([], after, threshold=0.0)
        # Empty before means all types are "new"
        assert len(shifts) >= 0  # no crash

    def test_empty_after(self):
        before = [_span(sid="b1", span_type="llm_call")]
        shifts = detect_distribution_shift(before, [], threshold=0.0)
        assert len(shifts) >= 0  # no crash

    def test_threshold_boundary(self):
        # 60/40 split vs 50/50: delta is 0.1 exactly
        before = [_span(sid=f"b{i}", span_type="a") for i in range(6)] + \
                 [_span(sid=f"b{i+6}", span_type="b") for i in range(4)]
        after = [_span(sid=f"a{i}", span_type="a") for i in range(5)] + \
                [_span(sid=f"a{i+5}", span_type="b") for i in range(5)]
        # threshold=0.1 means > 0.1, so exactly 0.1 should NOT be included
        shifts = detect_distribution_shift(before, after, threshold=0.1)
        assert shifts == []
