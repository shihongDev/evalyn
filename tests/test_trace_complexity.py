"""Tests for trace complexity scoring."""

import pytest
from datetime import datetime, timezone
from evalyn_sdk.models import Span
from evalyn_sdk.analysis.trace_complexity import (
    ComplexityScore,
    ComplexityThreshold,
    compute_complexity,
    check_complexity_alerts,
    _max_depth,
)


def _now():
    return datetime.now(timezone.utc)


def _span(sid, name="s", span_type="custom", parent_id=None):
    return Span(
        id=sid, name=name, span_type=span_type,
        parent_id=parent_id, start_time=_now(),
    )


# =============================================================================
# ComplexityScore
# =============================================================================

class TestComplexityScore:
    def test_defaults(self):
        s = ComplexityScore()
        assert s.score == 0.0
        assert s.total_spans == 0

    def test_as_dict(self):
        s = ComplexityScore(total_spans=5, max_depth=3, score=45.5)
        d = s.as_dict()
        assert d["total_spans"] == 5
        assert d["max_depth"] == 3
        assert d["score"] == 45.5

    def test_from_dict(self):
        d = {"total_spans": 10, "max_depth": 4, "score": 60.0}
        s = ComplexityScore.from_dict(d)
        assert s.total_spans == 10
        assert s.score == 60.0

    def test_format_text(self):
        s = ComplexityScore(total_spans=5, max_depth=3, max_breadth=2, score=45.0,
                           tool_call_count=1, llm_call_count=2)
        text = s.format_text()
        assert "45.0" in text
        assert "spans=5" in text
        assert "depth=3" in text


# =============================================================================
# compute_complexity
# =============================================================================

class TestComputeComplexity:
    def test_empty(self):
        s = compute_complexity([])
        assert s.total_spans == 0
        assert s.score == 0.0

    def test_single_span(self):
        spans = [_span("s1")]
        s = compute_complexity(spans)
        assert s.total_spans == 1
        assert s.max_depth == 1
        assert s.score > 0

    def test_flat_tree(self):
        # Root with 5 children
        spans = [_span("root")] + [_span(f"c{i}", parent_id="root") for i in range(5)]
        s = compute_complexity(spans)
        assert s.total_spans == 6
        assert s.max_depth == 2
        assert s.max_breadth == 5

    def test_deep_tree(self):
        # Chain: s0 -> s1 -> s2 -> s3 -> s4
        spans = [_span("s0")]
        for i in range(1, 5):
            spans.append(_span(f"s{i}", parent_id=f"s{i-1}"))
        s = compute_complexity(spans)
        assert s.max_depth == 5
        assert s.max_breadth == 1

    def test_tool_and_llm_counts(self):
        spans = [
            _span("s1", span_type="tool_call"),
            _span("s2", span_type="tool_call"),
            _span("s3", span_type="llm_call"),
            _span("s4", span_type="custom"),
        ]
        s = compute_complexity(spans)
        assert s.tool_call_count == 2
        assert s.llm_call_count == 1
        assert s.unique_span_types == 3

    def test_score_increases_with_complexity(self):
        simple = [_span("s1")]
        complex_spans = [_span("root")] + [
            _span(f"c{i}", span_type="tool_call", parent_id="root") for i in range(10)
        ]
        s_simple = compute_complexity(simple)
        s_complex = compute_complexity(complex_spans)
        assert s_complex.score > s_simple.score

    def test_custom_weights(self):
        spans = [_span("s1")]
        # All weight on depth
        w1 = compute_complexity(spans, weights={"depth": 100, "breadth": 0, "count": 0, "tools": 0, "types": 0})
        # All weight on count
        w2 = compute_complexity(spans, weights={"depth": 0, "breadth": 0, "count": 100, "tools": 0, "types": 0})
        # Both should produce valid scores
        assert 0 <= w1.score <= 100
        assert 0 <= w2.score <= 100

    def test_orphan_spans_treated_as_roots(self):
        # Span with parent_id pointing to non-existent span
        spans = [_span("s1", parent_id="nonexistent"), _span("s2")]
        s = compute_complexity(spans)
        # Both should be treated as roots
        assert s.max_depth == 1


# =============================================================================
# check_complexity_alerts
# =============================================================================

class TestCheckComplexityAlerts:
    def test_no_alerts(self):
        s = ComplexityScore(score=30, max_depth=3, total_spans=10)
        alerts = check_complexity_alerts(s)
        assert alerts == []

    def test_score_alert(self):
        s = ComplexityScore(score=80)
        alerts = check_complexity_alerts(s)
        assert len(alerts) == 1
        assert "score" in alerts[0].lower()

    def test_depth_alert(self):
        s = ComplexityScore(max_depth=15)
        alerts = check_complexity_alerts(s)
        assert any("depth" in a.lower() for a in alerts)

    def test_span_count_alert(self):
        s = ComplexityScore(total_spans=60)
        alerts = check_complexity_alerts(s)
        assert any("count" in a.lower() for a in alerts)

    def test_custom_threshold(self):
        s = ComplexityScore(score=40, max_depth=5, total_spans=20)
        t = ComplexityThreshold(max_score=30, max_depth=3, max_spans=10)
        alerts = check_complexity_alerts(s, threshold=t)
        assert len(alerts) == 3  # all exceeded

    def test_multiple_alerts(self):
        s = ComplexityScore(score=90, max_depth=20, total_spans=100)
        alerts = check_complexity_alerts(s)
        assert len(alerts) == 3


# =============================================================================
# _max_depth helper
# =============================================================================

class TestMaxDepth:
    def test_empty(self):
        assert _max_depth([], {}) == 0

    def test_single_root(self):
        assert _max_depth(["r"], {}) == 1

    def test_chain(self):
        children = {"r": ["a"], "a": ["b"], "b": ["c"]}
        assert _max_depth(["r"], children) == 4

    def test_wide(self):
        children = {"r": ["a", "b", "c"]}
        assert _max_depth(["r"], children) == 2
