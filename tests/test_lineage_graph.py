"""Tests for trace lineage graph module."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from evalyn_sdk.trace.lineage_graph import (
    LineageEdge,
    LineageNode,
    LineageGraph,
    build_lineage_graph,
    compute_content_hash,
    find_lineage_chain,
    render_lineage_mermaid,
)


def _now():
    return datetime.now(timezone.utc)


def _trace(tid, ts, input_text="", output_text=""):
    return {
        "id": tid,
        "timestamp": ts,
        "input_text": input_text,
        "output_text": output_text,
    }


# -- LineageEdge --


class TestLineageEdge:
    def test_as_dict(self):
        edge = LineageEdge("src", "tgt", 0.85)
        d = edge.as_dict()
        assert d["source_trace_id"] == "src"
        assert d["target_trace_id"] == "tgt"
        assert d["overlap_confidence"] == 0.85


# -- LineageNode --


class TestLineageNode:
    def test_as_dict(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        node = LineageNode(
            trace_id="n1",
            timestamp=ts,
            input_hash="abc123",
            output_hash="def456",
            upstream=["n0"],
            downstream=["n2"],
        )
        d = node.as_dict()
        assert d["trace_id"] == "n1"
        assert d["input_hash"] == "abc123"
        assert d["upstream"] == ["n0"]
        assert d["downstream"] == ["n2"]

    def test_default_fields(self):
        node = LineageNode(trace_id="n1", timestamp=_now())
        assert node.input_hash == ""
        assert node.output_hash == ""
        assert node.upstream == []
        assert node.downstream == []


# -- compute_content_hash --


class TestComputeContentHash:
    def test_deterministic(self):
        h1 = compute_content_hash("hello world")
        h2 = compute_content_hash("hello world")
        assert h1 == h2

    def test_length(self):
        h = compute_content_hash("test")
        assert len(h) == 16

    def test_different_inputs_different_hashes(self):
        h1 = compute_content_hash("aaa")
        h2 = compute_content_hash("bbb")
        assert h1 != h2

    def test_empty_string(self):
        h = compute_content_hash("")
        assert len(h) == 16


# -- build_lineage_graph --


class TestBuildLineageGraph:
    def test_detects_content_flow(self):
        base = _now()
        traces = [
            _trace("t1", base, "query", "result A"),
            _trace("t2", base + timedelta(seconds=1), "result A", "result B"),
        ]
        graph = build_lineage_graph(traces)
        assert len(graph.edges) == 1
        assert graph.edges[0].source_trace_id == "t1"
        assert graph.edges[0].target_trace_id == "t2"
        assert graph.edges[0].overlap_confidence == 1.0

    def test_respects_time_order(self):
        base = _now()
        # t2 has earlier timestamp - so t2's output matching t1's input
        # should NOT create an edge from t1 to t2
        traces = [
            _trace("t1", base + timedelta(seconds=1), "result A", "result B"),
            _trace("t2", base, "query", "result A"),
        ]
        graph = build_lineage_graph(traces)
        assert len(graph.edges) == 1
        # Edge should be t2 -> t1 (earlier to later)
        assert graph.edges[0].source_trace_id == "t2"
        assert graph.edges[0].target_trace_id == "t1"

    def test_no_overlap(self):
        base = _now()
        traces = [
            _trace("t1", base, "input1", "output1"),
            _trace("t2", base + timedelta(seconds=1), "input2", "output2"),
        ]
        graph = build_lineage_graph(traces)
        assert len(graph.edges) == 0

    def test_partial_overlap_substring(self):
        base = _now()
        traces = [
            _trace("t1", base, "query", "hello"),
            _trace("t2", base + timedelta(seconds=1), "hello world", "done"),
        ]
        graph = build_lineage_graph(traces)
        assert len(graph.edges) == 1
        assert graph.edges[0].overlap_confidence < 1.0
        assert graph.edges[0].overlap_confidence > 0.0

    def test_single_trace(self):
        graph = build_lineage_graph([_trace("t1", _now(), "in", "out")])
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0

    def test_empty_traces(self):
        graph = build_lineage_graph([])
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_upstream_downstream_populated(self):
        base = _now()
        traces = [
            _trace("t1", base, "q", "answer"),
            _trace("t2", base + timedelta(seconds=1), "answer", "final"),
        ]
        graph = build_lineage_graph(traces)
        assert "t2" in graph.nodes["t1"].downstream
        assert "t1" in graph.nodes["t2"].upstream

    def test_chain_of_three(self):
        base = _now()
        traces = [
            _trace("t1", base, "q", "step1"),
            _trace("t2", base + timedelta(seconds=1), "step1", "step2"),
            _trace("t3", base + timedelta(seconds=2), "step2", "done"),
        ]
        graph = build_lineage_graph(traces)
        assert len(graph.edges) == 2

    def test_empty_output_no_edge(self):
        base = _now()
        traces = [
            _trace("t1", base, "q", ""),
            _trace("t2", base + timedelta(seconds=1), "", "done"),
        ]
        graph = build_lineage_graph(traces)
        assert len(graph.edges) == 0


# -- roots and leaves properties --


class TestRootsAndLeaves:
    def test_roots(self):
        base = _now()
        traces = [
            _trace("t1", base, "q", "answer"),
            _trace("t2", base + timedelta(seconds=1), "answer", "final"),
        ]
        graph = build_lineage_graph(traces)
        root_ids = [n.trace_id for n in graph.roots]
        assert "t1" in root_ids
        assert "t2" not in root_ids

    def test_leaves(self):
        base = _now()
        traces = [
            _trace("t1", base, "q", "answer"),
            _trace("t2", base + timedelta(seconds=1), "answer", "final"),
        ]
        graph = build_lineage_graph(traces)
        leaf_ids = [n.trace_id for n in graph.leaves]
        assert "t2" in leaf_ids
        assert "t1" not in leaf_ids

    def test_single_node_is_root_and_leaf(self):
        graph = build_lineage_graph([_trace("t1", _now(), "in", "out")])
        assert len(graph.roots) == 1
        assert len(graph.leaves) == 1
        assert graph.roots[0].trace_id == "t1"
        assert graph.leaves[0].trace_id == "t1"


# -- find_lineage_chain --


class TestFindLineageChain:
    def test_follows_upstream(self):
        base = _now()
        traces = [
            _trace("t1", base, "q", "step1"),
            _trace("t2", base + timedelta(seconds=1), "step1", "step2"),
            _trace("t3", base + timedelta(seconds=2), "step2", "done"),
        ]
        graph = build_lineage_graph(traces)
        chain = find_lineage_chain(graph, "t3")
        assert chain == ["t1", "t2", "t3"]

    def test_root_chain_is_single(self):
        base = _now()
        traces = [
            _trace("t1", base, "q", "step1"),
            _trace("t2", base + timedelta(seconds=1), "step1", "done"),
        ]
        graph = build_lineage_graph(traces)
        chain = find_lineage_chain(graph, "t1")
        assert chain == ["t1"]

    def test_missing_trace_id(self):
        graph = LineageGraph()
        chain = find_lineage_chain(graph, "nonexistent")
        assert chain == []

    def test_middle_of_chain(self):
        base = _now()
        traces = [
            _trace("t1", base, "q", "step1"),
            _trace("t2", base + timedelta(seconds=1), "step1", "step2"),
            _trace("t3", base + timedelta(seconds=2), "step2", "done"),
        ]
        graph = build_lineage_graph(traces)
        chain = find_lineage_chain(graph, "t2")
        assert chain == ["t1", "t2"]


# -- LineageGraph serialization --


class TestLineageGraphSerialization:
    def test_as_dict_from_dict_roundtrip(self):
        base = _now()
        traces = [
            _trace("t1", base, "q", "answer"),
            _trace("t2", base + timedelta(seconds=1), "answer", "final"),
        ]
        graph = build_lineage_graph(traces)
        d = graph.as_dict()
        restored = LineageGraph.from_dict(d)
        assert len(restored.nodes) == 2
        assert len(restored.edges) == 1
        assert restored.edges[0].source_trace_id == "t1"
        assert restored.edges[0].target_trace_id == "t2"

    def test_empty_graph_roundtrip(self):
        graph = LineageGraph()
        d = graph.as_dict()
        restored = LineageGraph.from_dict(d)
        assert len(restored.nodes) == 0
        assert len(restored.edges) == 0


# -- format_text --


class TestLineageGraphFormatText:
    def test_format_text(self):
        base = _now()
        traces = [
            _trace("t1", base, "q", "answer"),
            _trace("t2", base + timedelta(seconds=1), "answer", "final"),
        ]
        graph = build_lineage_graph(traces)
        text = graph.format_text()
        assert "2 nodes" in text
        assert "1 edges" in text
        assert "t1" in text
        assert "t2" in text

    def test_format_text_empty(self):
        graph = LineageGraph()
        text = graph.format_text()
        assert "0 nodes" in text
        assert "0 edges" in text


# -- render_lineage_mermaid --


class TestRenderLineageMermaid:
    def test_mermaid_output(self):
        base = _now()
        traces = [
            _trace("t1", base, "q", "answer"),
            _trace("t2", base + timedelta(seconds=1), "answer", "final"),
        ]
        graph = build_lineage_graph(traces)
        mermaid = render_lineage_mermaid(graph)
        assert mermaid.startswith("graph TD")
        assert "t1" in mermaid
        assert "t2" in mermaid
        assert "-->" in mermaid

    def test_mermaid_empty_graph(self):
        graph = LineageGraph()
        mermaid = render_lineage_mermaid(graph)
        assert mermaid.startswith("graph TD")
        # No edges or nodes beyond the header
        lines = mermaid.strip().split("\n")
        assert len(lines) == 1
