"""Tests for span dependency graph."""

import pytest
from datetime import datetime, timezone
from evalyn_sdk.models import Span
from evalyn_sdk.analysis.span_dependency import (
    DependencyEdge,
    SpanNode,
    DependencyGraph,
    build_dependency_graph,
    _extract_text,
    _overlap_ratio,
)


def _now():
    return datetime.now(timezone.utc)


def _span(sid, name="s", span_type="custom", parent_id=None, **attrs):
    return Span(
        id=sid, name=name, span_type=span_type,
        parent_id=parent_id, start_time=_now(),
        attributes=attrs,
    )


# =============================================================================
# DependencyEdge
# =============================================================================

class TestDependencyEdge:
    def test_as_dict(self):
        e = DependencyEdge(source_id="a", target_id="b", overlap_ratio=0.75)
        d = e.as_dict()
        assert d["source_id"] == "a"
        assert d["target_id"] == "b"
        assert d["overlap_ratio"] == 0.75


# =============================================================================
# SpanNode
# =============================================================================

class TestSpanNode:
    def test_as_dict(self):
        n = SpanNode(
            span_id="s1", span_name="call", span_type="llm_call",
            upstream=["s0"], downstream=["s2", "s3"],
            is_bottleneck=True,
        )
        d = n.as_dict()
        assert d["span_id"] == "s1"
        assert d["upstream_count"] == 1
        assert d["downstream_count"] == 2
        assert d["is_bottleneck"] is True


# =============================================================================
# DependencyGraph
# =============================================================================

class TestDependencyGraph:
    def test_empty(self):
        g = DependencyGraph()
        assert g.bottlenecks == []
        assert g.roots == []
        assert g.leaves == []

    def test_properties(self):
        g = DependencyGraph(
            nodes={
                "a": SpanNode(span_id="a", span_name="root", span_type="custom",
                              downstream=["b", "c"], is_bottleneck=True),
                "b": SpanNode(span_id="b", span_name="mid", span_type="tool_call",
                              upstream=["a"], downstream=["c"]),
                "c": SpanNode(span_id="c", span_name="leaf", span_type="llm_call",
                              upstream=["a", "b"]),
            },
        )
        assert "a" in g.bottlenecks
        assert "a" in g.roots
        assert "c" in g.leaves

    def test_as_dict(self):
        g = DependencyGraph(
            nodes={"a": SpanNode(span_id="a", span_name="s", span_type="x")},
            edges=[DependencyEdge("a", "b", 0.5)],
        )
        d = g.as_dict()
        assert d["num_nodes"] == 1
        assert d["num_edges"] == 1

    def test_from_dict_roundtrip(self):
        g = DependencyGraph(
            nodes={
                "a": SpanNode(span_id="a", span_name="src", span_type="tool_call"),
                "b": SpanNode(span_id="b", span_name="tgt", span_type="llm_call"),
            },
            edges=[DependencyEdge("a", "b", 0.8)],
        )
        # Manually set upstream/downstream for the original
        g.nodes["a"].downstream = ["b"]
        g.nodes["b"].upstream = ["a"]

        d = g.as_dict()
        restored = DependencyGraph.from_dict(d)
        assert len(restored.nodes) == 2
        assert len(restored.edges) == 1
        assert "b" in restored.nodes["a"].downstream
        assert "a" in restored.nodes["b"].upstream

    def test_to_mermaid(self):
        g = DependencyGraph(
            nodes={
                "a": SpanNode(span_id="a", span_name="query", span_type="llm_call",
                              downstream=["b"], is_bottleneck=True),
                "b": SpanNode(span_id="b", span_name="tool", span_type="tool_call",
                              upstream=["a"]),
            },
            edges=[DependencyEdge("a", "b", 0.5)],
        )
        mermaid = g.to_mermaid()
        assert "graph TD" in mermaid
        assert "query" in mermaid
        assert "50%" in mermaid

    def test_format_text(self):
        g = DependencyGraph(
            nodes={
                "a": SpanNode(span_id="a", span_name="src", span_type="x",
                              downstream=["b"], is_bottleneck=True),
                "b": SpanNode(span_id="b", span_name="tgt", span_type="y",
                              upstream=["a"]),
            },
            edges=[DependencyEdge("a", "b", 0.7)],
        )
        text = g.format_text()
        assert "2 nodes" in text
        assert "Bottlenecks" in text
        assert "src -> tgt" in text


# =============================================================================
# build_dependency_graph
# =============================================================================

class TestBuildDependencyGraph:
    def test_empty(self):
        g = build_dependency_graph([])
        assert len(g.nodes) == 0

    def test_no_content_overlap(self):
        spans = [
            _span("s1", output="hello world"),
            _span("s2", input="completely different"),
        ]
        g = build_dependency_graph(spans)
        assert len(g.nodes) == 2
        # No content-based edges (only structural if parent-child)
        content_edges = [e for e in g.edges if e.overlap_ratio > 0]
        assert len(content_edges) == 0

    def test_content_overlap_creates_edge(self):
        spans = [
            _span("s1", name="query", output="The capital of France is Paris"),
            _span("s2", name="use", input="Based on: The capital of France is Paris"),
        ]
        g = build_dependency_graph(spans, min_overlap=0.5)
        content_edges = [e for e in g.edges if e.overlap_ratio > 0]
        assert len(content_edges) >= 1
        assert content_edges[0].source_id == "s1"
        assert content_edges[0].target_id == "s2"

    def test_structural_edges_from_parent_child(self):
        spans = [
            _span("root", name="root"),
            _span("child", name="child", parent_id="root"),
        ]
        g = build_dependency_graph(spans)
        structural_edges = [e for e in g.edges if e.overlap_ratio == 0.0]
        assert len(structural_edges) == 1
        assert structural_edges[0].source_id == "root"
        assert structural_edges[0].target_id == "child"

    def test_bottleneck_detection(self):
        # Root with 3 children - should be marked as bottleneck with threshold=2
        spans = [
            _span("root", name="root"),
            _span("c1", name="c1", parent_id="root"),
            _span("c2", name="c2", parent_id="root"),
            _span("c3", name="c3", parent_id="root"),
        ]
        g = build_dependency_graph(spans, bottleneck_threshold=2)
        assert g.nodes["root"].is_bottleneck is True
        assert g.nodes["c1"].is_bottleneck is False

    def test_min_overlap_filter(self):
        spans = [
            _span("s1", output="hello world testing output"),
            _span("s2", input="hello different content here"),
        ]
        # With high min_overlap, weak overlap should be filtered
        g_strict = build_dependency_graph(spans, min_overlap=0.9)
        g_loose = build_dependency_graph(spans, min_overlap=0.01)
        strict_content = [e for e in g_strict.edges if e.overlap_ratio > 0]
        loose_content = [e for e in g_loose.edges if e.overlap_ratio > 0]
        assert len(strict_content) <= len(loose_content)

    def test_short_output_skipped(self):
        spans = [
            _span("s1", output="hi"),  # too short (< 5 chars)
            _span("s2", input="hi there"),
        ]
        g = build_dependency_graph(spans)
        content_edges = [e for e in g.edges if e.overlap_ratio > 0]
        assert len(content_edges) == 0


# =============================================================================
# Internal helpers
# =============================================================================

class TestExtractText:
    def test_string(self):
        assert _extract_text({"output": "hello"}, "output") == "hello"

    def test_dict_with_text_key(self):
        assert _extract_text({"output": {"text": "hello"}}, "output") == "hello"

    def test_dict_with_content_key(self):
        assert _extract_text({"input": {"content": "data"}}, "input") == "data"

    def test_missing_key(self):
        assert _extract_text({}, "output") == ""

    def test_none_value(self):
        assert _extract_text({"output": None}, "output") == ""

    def test_numeric(self):
        assert _extract_text({"output": 42}, "output") == "42"


class TestOverlapRatio:
    def test_full_overlap(self):
        assert _overlap_ratio("hello world", "contains hello world here") == 1.0

    def test_no_overlap(self):
        assert _overlap_ratio("abc", "xyz") == 0.0

    def test_partial_overlap(self):
        r = _overlap_ratio("hello world testing", "hello world different")
        assert 0.0 < r < 1.0

    def test_empty_source(self):
        assert _overlap_ratio("", "hello") == 0.0

    def test_empty_target(self):
        assert _overlap_ratio("hello", "") == 0.0

    def test_case_insensitive(self):
        assert _overlap_ratio("Hello World", "contains hello world") == 1.0
