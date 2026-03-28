"""Tests for unit type auto-detection module."""

import pytest
from datetime import datetime, timezone

from evalyn_sdk.models import Span
from evalyn_sdk.evaluation.unit_detection import (
    UNIT_TYPES,
    TracePattern,
    DetectionResult,
    analyze_trace_pattern,
    detect_unit_type,
    detect_unit_types_batch,
)


def _span(sid, span_type="llm_call", parent_id=None):
    return Span(
        id=sid,
        name=f"span-{sid}",
        span_type=span_type,
        parent_id=parent_id,
        start_time=datetime.now(timezone.utc),
    )


class TestUnitTypes:
    def test_contains_expected_types(self):
        assert "outcome" in UNIT_TYPES
        assert "per_call" in UNIT_TYPES
        assert "per_turn" in UNIT_TYPES
        assert "per_tool" in UNIT_TYPES
        assert "per_node" in UNIT_TYPES

    def test_length(self):
        assert len(UNIT_TYPES) == 5


class TestTracePattern:
    def test_as_dict(self):
        p = TracePattern(
            has_multi_turn=True,
            has_tool_calls=False,
            has_retrieval=True,
            has_graph_nodes=False,
            llm_call_count=3,
            tool_call_count=0,
            max_depth=2,
        )
        d = p.as_dict()
        assert d["has_multi_turn"] is True
        assert d["llm_call_count"] == 3
        assert d["max_depth"] == 2

    def test_as_dict_all_false(self):
        p = TracePattern(
            has_multi_turn=False,
            has_tool_calls=False,
            has_retrieval=False,
            has_graph_nodes=False,
            llm_call_count=0,
            tool_call_count=0,
            max_depth=0,
        )
        d = p.as_dict()
        assert d["has_multi_turn"] is False
        assert d["has_tool_calls"] is False


class TestDetectionResult:
    def test_as_dict(self):
        p = TracePattern(False, False, False, False, 1, 0, 0)
        r = DetectionResult(
            recommended_unit_type="per_call",
            confidence=0.9,
            pattern=p,
            reasoning="Single LLM call",
        )
        d = r.as_dict()
        assert d["recommended_unit_type"] == "per_call"
        assert d["confidence"] == 0.9
        assert "pattern" in d

    def test_from_dict_roundtrip(self):
        p = TracePattern(True, True, False, False, 5, 2, 3)
        r = DetectionResult(
            recommended_unit_type="per_tool",
            confidence=0.80,
            pattern=p,
            reasoning="Tool calls present",
        )
        restored = DetectionResult.from_dict(r.as_dict())
        assert restored.recommended_unit_type == "per_tool"
        assert restored.confidence == pytest.approx(0.80)
        assert restored.pattern.llm_call_count == 5
        assert restored.reasoning == "Tool calls present"

    def test_confidence_bounds(self):
        p = TracePattern(False, False, False, False, 0, 0, 0)
        r = DetectionResult("outcome", 0.70, p, "default")
        assert 0.0 <= r.confidence <= 1.0


class TestAnalyzeTracePattern:
    def test_empty_spans(self):
        p = analyze_trace_pattern([])
        assert p.llm_call_count == 0
        assert p.tool_call_count == 0
        assert p.has_multi_turn is False
        assert p.max_depth == 0

    def test_single_llm_call(self):
        spans = [_span("s1", "llm_call")]
        p = analyze_trace_pattern(spans)
        assert p.llm_call_count == 1
        assert p.has_multi_turn is False

    def test_multi_turn_detection(self):
        # 3 llm_calls at depth 0 (no parent)
        spans = [
            _span("s1", "llm_call"),
            _span("s2", "llm_call"),
            _span("s3", "llm_call"),
        ]
        p = analyze_trace_pattern(spans)
        assert p.has_multi_turn is True
        assert p.llm_call_count == 3

    def test_tool_call_detection(self):
        spans = [
            _span("s1", "llm_call"),
            _span("s2", "tool_call", parent_id="s1"),
        ]
        p = analyze_trace_pattern(spans)
        assert p.has_tool_calls is True
        assert p.tool_call_count == 1

    def test_retrieval_detection(self):
        spans = [_span("s1", "retrieval")]
        p = analyze_trace_pattern(spans)
        assert p.has_retrieval is True

    def test_graph_node_detection(self):
        spans = [_span("s1", "node")]
        p = analyze_trace_pattern(spans)
        assert p.has_graph_nodes is True

    def test_counts_correct(self):
        spans = [
            _span("s1", "llm_call"),
            _span("s2", "llm_call"),
            _span("s3", "tool_call"),
            _span("s4", "tool_call"),
            _span("s5", "tool_call"),
        ]
        p = analyze_trace_pattern(spans)
        assert p.llm_call_count == 2
        assert p.tool_call_count == 3

    def test_depth_computation(self):
        # root -> child -> grandchild
        spans = [
            _span("root", "session"),
            _span("child", "llm_call", parent_id="root"),
            _span("grandchild", "tool_call", parent_id="child"),
        ]
        p = analyze_trace_pattern(spans)
        assert p.max_depth == 2

    def test_multi_turn_same_depth(self):
        # Two llm_calls under the same parent (same depth)
        spans = [
            _span("root", "session"),
            _span("c1", "llm_call", parent_id="root"),
            _span("c2", "llm_call", parent_id="root"),
        ]
        p = analyze_trace_pattern(spans)
        assert p.has_multi_turn is True

    def test_not_multi_turn_different_depths(self):
        # One llm_call at depth 1, one at depth 2 - not multi-turn
        spans = [
            _span("root", "session"),
            _span("c1", "llm_call", parent_id="root"),
            _span("c2", "llm_call", parent_id="c1"),
        ]
        p = analyze_trace_pattern(spans)
        assert p.has_multi_turn is False


class TestDetectUnitType:
    def test_single_llm_call_per_call(self):
        spans = [_span("s1", "llm_call")]
        r = detect_unit_type(spans)
        assert r.recommended_unit_type == "per_call"
        assert r.confidence == pytest.approx(0.90)

    def test_multi_turn_three_calls(self):
        spans = [
            _span("s1", "llm_call"),
            _span("s2", "llm_call"),
            _span("s3", "llm_call"),
        ]
        r = detect_unit_type(spans)
        assert r.recommended_unit_type == "per_turn"
        assert r.confidence == pytest.approx(0.85)

    def test_tool_calls_per_tool(self):
        spans = [
            _span("s1", "llm_call"),
            _span("s2", "tool_call", parent_id="s1"),
        ]
        r = detect_unit_type(spans)
        assert r.recommended_unit_type == "per_tool"

    def test_graph_nodes_per_node(self):
        spans = [_span("s1", "node")]
        r = detect_unit_type(spans)
        assert r.recommended_unit_type == "per_node"
        assert r.confidence == pytest.approx(0.75)

    def test_default_outcome(self):
        # No llm_calls, no tools, no nodes - just a custom span
        spans = [_span("s1", "custom")]
        r = detect_unit_type(spans)
        assert r.recommended_unit_type == "outcome"
        assert r.confidence == pytest.approx(0.70)

    def test_empty_spans_outcome(self):
        r = detect_unit_type([])
        assert r.recommended_unit_type == "outcome"

    def test_multi_turn_priority_over_tool(self):
        # Multi-turn with tool calls - multi-turn wins (higher priority)
        spans = [
            _span("s1", "llm_call"),
            _span("s2", "llm_call"),
            _span("s3", "llm_call"),
            _span("s4", "tool_call"),
        ]
        r = detect_unit_type(spans)
        assert r.recommended_unit_type == "per_turn"

    def test_tool_priority_over_node(self):
        spans = [
            _span("s1", "tool_call"),
            _span("s2", "node"),
        ]
        r = detect_unit_type(spans)
        assert r.recommended_unit_type == "per_tool"

    def test_reasoning_not_empty(self):
        spans = [_span("s1", "llm_call")]
        r = detect_unit_type(spans)
        assert len(r.reasoning) > 0

    def test_two_llm_calls_not_multi_turn(self):
        # Only 2 llm_calls at same depth - has_multi_turn is True but llm_call_count < 3
        # So it falls through to next check
        spans = [
            _span("s1", "llm_call"),
            _span("s2", "llm_call"),
        ]
        r = detect_unit_type(spans)
        # has_multi_turn=True but llm_call_count=2 < 3, so not per_turn
        # No tool_calls, no graph_nodes, llm_call_count != 1
        assert r.recommended_unit_type == "outcome"


class TestDetectUnitTypesBatch:
    def test_batch_counts(self):
        traces = [
            [_span("s1", "llm_call")],  # per_call
            [_span("s2", "llm_call")],  # per_call
            [_span("s3", "tool_call")],  # per_tool
        ]
        counts = detect_unit_types_batch(traces)
        assert counts["per_call"] == 2
        assert counts["per_tool"] == 1

    def test_empty_batch(self):
        counts = detect_unit_types_batch([])
        assert all(v == 0 for v in counts.values())

    def test_all_types_present_in_keys(self):
        counts = detect_unit_types_batch([])
        for ut in UNIT_TYPES:
            assert ut in counts

    def test_mixed_batch(self):
        traces = [
            [_span("s1", "llm_call")],  # per_call
            [_span("s2", "llm_call"), _span("s3", "llm_call"), _span("s4", "llm_call")],  # per_turn
            [_span("s5", "node")],  # per_node
            [_span("s6", "custom")],  # outcome
        ]
        counts = detect_unit_types_batch(traces)
        assert counts["per_call"] == 1
        assert counts["per_turn"] == 1
        assert counts["per_node"] == 1
        assert counts["outcome"] == 1
