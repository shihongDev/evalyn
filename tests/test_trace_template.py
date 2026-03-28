"""Tests for trace template matching."""

import pytest
from datetime import datetime, timezone, timedelta
from evalyn_sdk.models import Span
from evalyn_sdk.analysis.trace_template import (
    TracePattern,
    MatchResult,
    ClassificationReport,
    match_trace,
    classify_traces,
    BUILTIN_PATTERNS,
    RAG_PATTERN,
    RETRY_LOOP,
    TOOL_USE,
    AGENT_LOOP,
    FAN_OUT,
    _subsequence_match,
    _contiguous_match,
)


def _now():
    return datetime.now(timezone.utc)


def _span(sid, span_type="custom", offset_ms=0):
    t = _now() + timedelta(milliseconds=offset_ms)
    return Span(
        id=sid, name=f"span-{sid}", span_type=span_type,
        parent_id=None, start_time=t,
    )


def _trace(*types):
    return [_span(f"s{i}", t, offset_ms=i * 100) for i, t in enumerate(types)]


# =============================================================================
# TracePattern
# =============================================================================

class TestTracePattern:
    def test_as_dict(self):
        p = TracePattern(name="test", description="desc", sequence=["a", "b"])
        d = p.as_dict()
        assert d["name"] == "test"
        assert d["sequence"] == ["a", "b"]
        assert d["allow_gaps"] is True

    def test_from_dict(self):
        d = {"name": "custom", "description": "x", "sequence": ["c", "d"], "allow_gaps": False}
        p = TracePattern.from_dict(d)
        assert p.name == "custom"
        assert p.sequence == ["c", "d"]
        assert p.allow_gaps is False


# =============================================================================
# Builtin patterns
# =============================================================================

class TestBuiltinPatterns:
    def test_five_patterns(self):
        assert len(BUILTIN_PATTERNS) == 5

    def test_rag(self):
        assert RAG_PATTERN.sequence == ["retrieval", "llm_call"]

    def test_retry_loop(self):
        assert RETRY_LOOP.sequence == ["llm_call", "llm_call"]

    def test_tool_use(self):
        assert TOOL_USE.sequence == ["llm_call", "tool_call"]

    def test_agent_loop(self):
        assert AGENT_LOOP.sequence == ["llm_call", "tool_call", "llm_call"]


# =============================================================================
# match_trace
# =============================================================================

class TestMatchTrace:
    def test_exact_match(self):
        trace = _trace("retrieval", "llm_call")
        result = match_trace(trace, RAG_PATTERN)
        assert result.matched is True
        assert result.pattern_name == "rag"
        assert len(result.matched_span_ids) == 2

    def test_match_with_gaps(self):
        # RAG pattern with extra spans in between
        trace = _trace("custom", "retrieval", "processing", "llm_call", "custom")
        result = match_trace(trace, RAG_PATTERN)
        assert result.matched is True
        assert result.coverage == pytest.approx(2 / 5)

    def test_no_match(self):
        trace = _trace("tool_call", "tool_call")
        result = match_trace(trace, RAG_PATTERN)
        assert result.matched is False

    def test_contiguous_required(self):
        pattern = TracePattern(
            name="strict", description="",
            sequence=["a", "b"], allow_gaps=False,
        )
        # With gap
        trace = _trace("a", "x", "b")
        assert match_trace(trace, pattern).matched is False

        # Without gap
        trace = _trace("a", "b")
        assert match_trace(trace, pattern).matched is True

    def test_agent_loop_match(self):
        trace = _trace("llm_call", "tool_call", "llm_call")
        result = match_trace(trace, AGENT_LOOP)
        assert result.matched is True

    def test_empty_trace(self):
        result = match_trace([], RAG_PATTERN)
        assert result.matched is False

    def test_coverage_full_match(self):
        trace = _trace("llm_call", "tool_call")
        result = match_trace(trace, TOOL_USE)
        assert result.coverage == pytest.approx(1.0)


# =============================================================================
# classify_traces
# =============================================================================

class TestClassifyTraces:
    def test_empty(self):
        r = classify_traces([])
        assert r.trace_count == 0
        assert r.classified == 0

    def test_single_classified(self):
        traces = [_trace("retrieval", "llm_call")]
        r = classify_traces(traces)
        assert r.classified == 1
        assert r.unclassified == 0
        assert "rag" in r.pattern_counts

    def test_unclassified(self):
        traces = [_trace("custom", "custom")]
        r = classify_traces(traces)
        assert r.unclassified == 1
        assert r.classified == 0

    def test_multiple_traces(self):
        traces = [
            _trace("retrieval", "llm_call"),
            _trace("llm_call", "tool_call"),
            _trace("custom", "custom"),
        ]
        r = classify_traces(traces)
        assert r.trace_count == 3
        assert r.classified == 2
        assert r.unclassified == 1

    def test_coverage_rate(self):
        traces = [_trace("retrieval", "llm_call")] * 3 + [_trace("custom")] * 1
        r = classify_traces(traces)
        assert r.coverage_rate == pytest.approx(0.75)

    def test_custom_patterns(self):
        custom = [{"name": "my_pattern", "sequence": ["x", "y"], "description": "test"}]
        traces = [_trace("x", "y")]
        r = classify_traces(traces, custom_patterns=custom)
        assert "my_pattern" in r.pattern_counts

    def test_custom_patterns_extend_builtin(self):
        custom = [{"name": "extra", "sequence": ["z", "z"], "description": ""}]
        traces = [_trace("retrieval", "llm_call")]
        r = classify_traces(traces, custom_patterns=custom)
        # Should still match RAG from builtins
        assert "rag" in r.pattern_counts

    def test_as_dict(self):
        r = classify_traces([_trace("llm_call", "tool_call")])
        d = r.as_dict()
        assert "trace_count" in d
        assert "coverage_rate" in d
        assert "pattern_counts" in d

    def test_from_dict(self):
        r = ClassificationReport(
            trace_count=5, classified=3, unclassified=2,
            pattern_counts={"rag": 2, "tool_use": 1},
        )
        d = r.as_dict()
        restored = ClassificationReport.from_dict(d)
        assert restored.trace_count == 5
        assert restored.classified == 3

    def test_format_text(self):
        r = classify_traces([_trace("retrieval", "llm_call")])
        text = r.format_text()
        assert "Classification" in text
        assert "rag" in text


# =============================================================================
# Internal helpers
# =============================================================================

class TestSubsequenceMatch:
    def test_exact(self):
        assert _subsequence_match(["a", "b"], ["a", "b"]) == [0, 1]

    def test_with_gaps(self):
        assert _subsequence_match(["a", "x", "b"], ["a", "b"]) == [0, 2]

    def test_no_match(self):
        assert _subsequence_match(["a", "c"], ["a", "b"]) is None

    def test_empty_pattern(self):
        assert _subsequence_match(["a"], []) == []


class TestContiguousMatch:
    def test_exact(self):
        assert _contiguous_match(["a", "b", "c"], ["a", "b"]) == [0, 1]

    def test_middle(self):
        assert _contiguous_match(["x", "a", "b", "y"], ["a", "b"]) == [1, 2]

    def test_no_match(self):
        assert _contiguous_match(["a", "x", "b"], ["a", "b"]) is None

    def test_pattern_longer_than_types(self):
        assert _contiguous_match(["a"], ["a", "b"]) is None
