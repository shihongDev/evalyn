"""Tests for hot path detection."""

import pytest
from datetime import datetime, timezone, timedelta
from evalyn_sdk.models import Span
from evalyn_sdk.analysis.hot_path import (
    PathPattern,
    HotPathReport,
    detect_hot_paths,
)


def _now():
    return datetime.now(timezone.utc)


def _span(sid, span_type="custom", offset_ms=0):
    t = _now() + timedelta(milliseconds=offset_ms)
    return Span(
        id=sid, name=f"span-{sid}", span_type=span_type,
        parent_id=None, start_time=t, end_time=t + timedelta(milliseconds=10),
    )


def _trace(*types):
    """Build a trace from span type names, ordered by creation."""
    return [_span(f"s{i}", t, offset_ms=i * 100) for i, t in enumerate(types)]


# =============================================================================
# PathPattern
# =============================================================================

class TestPathPattern:
    def test_as_dict(self):
        p = PathPattern(
            sequence=("llm_call", "tool_call"),
            count=5,
            total_duration_ms=100.0,
            avg_duration_ms=20.0,
        )
        d = p.as_dict()
        assert d["sequence"] == ["llm_call", "tool_call"]
        assert d["count"] == 5

    def test_format_text(self):
        p = PathPattern(
            sequence=("llm_call", "tool_call"),
            count=3,
            avg_duration_ms=15.5,
            total_duration_ms=46.5,
        )
        text = p.format_text()
        assert "llm_call -> tool_call" in text
        assert "x3" in text


# =============================================================================
# HotPathReport
# =============================================================================

class TestHotPathReport:
    def test_empty(self):
        r = HotPathReport()
        assert r.total_traces == 0
        assert r.top_by_frequency == []

    def test_as_dict(self):
        r = HotPathReport(
            patterns=[PathPattern(("a", "b"), count=3)],
            total_traces=5,
        )
        d = r.as_dict()
        assert d["total_traces"] == 5
        assert d["num_patterns"] == 1

    def test_from_dict(self):
        r = HotPathReport(
            patterns=[PathPattern(("a", "b"), count=3, total_duration_ms=30.0)],
            total_traces=2,
        )
        d = r.as_dict()
        restored = HotPathReport.from_dict(d)
        assert restored.total_traces == 2
        assert len(restored.patterns) == 1
        assert restored.patterns[0].sequence == ("a", "b")

    def test_top_by_frequency(self):
        r = HotPathReport(patterns=[
            PathPattern(("a",), count=1),
            PathPattern(("b",), count=5),
            PathPattern(("c",), count=3),
        ])
        top = r.top_by_frequency
        assert top[0].sequence == ("b",)
        assert top[1].sequence == ("c",)

    def test_top_by_cost(self):
        r = HotPathReport(patterns=[
            PathPattern(("a",), total_duration_ms=10),
            PathPattern(("b",), total_duration_ms=100),
        ])
        top = r.top_by_cost
        assert top[0].sequence == ("b",)

    def test_format_text(self):
        r = HotPathReport(
            patterns=[PathPattern(("llm_call", "tool_call"), count=5,
                                  avg_duration_ms=10, total_duration_ms=50)],
            total_traces=3,
        )
        text = r.format_text()
        assert "Hot Paths" in text
        assert "3 traces" in text


# =============================================================================
# detect_hot_paths
# =============================================================================

class TestDetectHotPaths:
    def test_empty(self):
        r = detect_hot_paths([])
        assert r.total_traces == 0
        assert len(r.patterns) == 0

    def test_single_trace_no_pattern(self):
        # With min_count=2, single trace can't produce patterns
        traces = [_trace("llm_call", "tool_call", "llm_call")]
        r = detect_hot_paths(traces, window_size=2, min_count=2)
        assert len(r.patterns) == 0

    def test_repeated_pattern(self):
        # Same pattern in multiple traces
        traces = [
            _trace("llm_call", "tool_call", "llm_call"),
            _trace("llm_call", "tool_call", "custom"),
        ]
        r = detect_hot_paths(traces, window_size=2, min_count=2)
        # "llm_call -> tool_call" appears in both traces
        seqs = [p.sequence for p in r.patterns]
        assert ("llm_call", "tool_call") in seqs

    def test_count_correct(self):
        traces = [
            _trace("a", "b", "c"),
            _trace("a", "b", "d"),
            _trace("a", "b", "e"),
        ]
        r = detect_hot_paths(traces, window_size=2, min_count=1)
        ab_pattern = [p for p in r.patterns if p.sequence == ("a", "b")]
        assert len(ab_pattern) == 1
        assert ab_pattern[0].count == 3

    def test_window_size_3(self):
        traces = [
            _trace("a", "b", "c", "d"),
            _trace("a", "b", "c", "e"),
        ]
        r = detect_hot_paths(traces, window_size=3, min_count=2)
        seqs = [p.sequence for p in r.patterns]
        assert ("a", "b", "c") in seqs

    def test_min_count_filter(self):
        traces = [
            _trace("a", "b"),
            _trace("c", "d"),
        ]
        r = detect_hot_paths(traces, window_size=2, min_count=2)
        assert len(r.patterns) == 0

    def test_max_patterns_limit(self):
        # Generate many unique patterns
        traces = [_trace(f"t{i}", f"t{i+1}") for i in range(20)]
        r = detect_hot_paths(traces, window_size=2, min_count=1, max_patterns=5)
        assert len(r.patterns) <= 5

    def test_window_clamped(self):
        # window_size < 2 should be clamped to 2
        traces = [_trace("a", "b", "c")] * 3
        r = detect_hot_paths(traces, window_size=1, min_count=2)
        # Should work with effective window=2
        assert len(r.patterns) > 0

    def test_examples_collected(self):
        traces = [_trace("a", "b")] * 5
        r = detect_hot_paths(traces, window_size=2, min_count=1)
        if r.patterns:
            assert len(r.patterns[0].example_span_ids) <= 3
