from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.analysis.graph_topology import (
    GraphEdge,
    GraphNode,
    GraphTopology,
    compute_critical_path,
    detect_cycles,
    extract_topology,
    find_entry_nodes,
    find_exit_nodes,
    render_topology_mermaid,
)


def _span(sid, name="node", span_type="node", parent_id=None, duration_ms=100):
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Span(
        id=sid,
        name=name,
        span_type=span_type,
        parent_id=parent_id,
        start_time=t,
        end_time=t + timedelta(milliseconds=duration_ms),
    )


# -- extract_topology --


def test_extract_topology_linear():
    spans = [
        _span("a", name="start", span_type="node"),
        _span("b", name="middle", span_type="node", parent_id="a"),
        _span("c", name="end", span_type="node", parent_id="b"),
    ]
    topo = extract_topology(spans)
    assert len(topo.nodes) == 3
    assert len(topo.edges) == 2


def test_extract_topology_branching():
    spans = [
        _span("root", name="root", span_type="graph"),
        _span("a", name="branch_a", span_type="node", parent_id="root"),
        _span("b", name="branch_b", span_type="node", parent_id="root"),
    ]
    topo = extract_topology(spans)
    assert len(topo.nodes) == 3
    assert len(topo.edges) == 2


def test_extract_topology_empty_spans():
    topo = extract_topology([])
    assert len(topo.nodes) == 0
    assert len(topo.edges) == 0


def test_extract_topology_single_span():
    spans = [_span("a", name="only", span_type="node")]
    topo = extract_topology(spans)
    assert len(topo.nodes) == 1
    assert len(topo.edges) == 0


def test_extract_topology_counts_execution():
    spans = [
        _span("a1", name="step", span_type="node"),
        _span("a2", name="step", span_type="node"),
    ]
    topo = extract_topology(spans)
    node = topo.nodes["step:node"]
    assert node.execution_count == 2


def test_extract_topology_avg_duration():
    spans = [
        _span("a1", name="step", span_type="node", duration_ms=100),
        _span("a2", name="step", span_type="node", duration_ms=200),
    ]
    topo = extract_topology(spans)
    node = topo.nodes["step:node"]
    assert node.avg_duration_ms == 150.0


def test_extract_topology_sets_entry_exit():
    spans = [
        _span("a", name="start", span_type="node"),
        _span("b", name="end", span_type="node", parent_id="a"),
    ]
    topo = extract_topology(spans)
    assert "start:node" in topo.entry_nodes
    assert "end:node" in topo.exit_nodes


# -- find_entry_nodes / find_exit_nodes --


def test_find_entry_nodes_correct():
    topo = GraphTopology(
        nodes={
            "a": GraphNode(node_id="a", node_name="a"),
            "b": GraphNode(node_id="b", node_name="b"),
            "c": GraphNode(node_id="c", node_name="c"),
        },
        edges=[GraphEdge(source="a", target="b"), GraphEdge(source="b", target="c")],
    )
    entries = find_entry_nodes(topo)
    assert entries == ["a"]


def test_find_exit_nodes_correct():
    topo = GraphTopology(
        nodes={
            "a": GraphNode(node_id="a", node_name="a"),
            "b": GraphNode(node_id="b", node_name="b"),
            "c": GraphNode(node_id="c", node_name="c"),
        },
        edges=[GraphEdge(source="a", target="b"), GraphEdge(source="b", target="c")],
    )
    exits = find_exit_nodes(topo)
    assert exits == ["c"]


def test_find_entry_nodes_disconnected():
    topo = GraphTopology(
        nodes={
            "a": GraphNode(node_id="a", node_name="a"),
            "b": GraphNode(node_id="b", node_name="b"),
        },
        edges=[],
    )
    entries = find_entry_nodes(topo)
    assert set(entries) == {"a", "b"}


def test_find_exit_nodes_disconnected():
    topo = GraphTopology(
        nodes={
            "a": GraphNode(node_id="a", node_name="a"),
            "b": GraphNode(node_id="b", node_name="b"),
        },
        edges=[],
    )
    exits = find_exit_nodes(topo)
    assert set(exits) == {"a", "b"}


# -- render_topology_mermaid --


def test_render_topology_mermaid_output():
    topo = GraphTopology(
        nodes={
            "a:node": GraphNode(node_id="a:node", node_name="a", execution_count=2),
            "b:node": GraphNode(node_id="b:node", node_name="b", execution_count=1),
        },
        edges=[GraphEdge(source="a:node", target="b:node")],
    )
    mermaid = render_topology_mermaid(topo)
    assert mermaid.startswith("graph TD")
    assert "a_node" in mermaid
    assert "b_node" in mermaid
    assert "-->" in mermaid


def test_render_topology_mermaid_empty():
    topo = GraphTopology()
    mermaid = render_topology_mermaid(topo)
    assert "graph TD" in mermaid


# -- compute_critical_path --


def test_compute_critical_path_longest():
    topo = GraphTopology(
        nodes={
            "a": GraphNode(node_id="a", node_name="a", avg_duration_ms=10),
            "b": GraphNode(node_id="b", node_name="b", avg_duration_ms=100),
            "c": GraphNode(node_id="c", node_name="c", avg_duration_ms=5),
            "d": GraphNode(node_id="d", node_name="d", avg_duration_ms=200),
        },
        edges=[
            GraphEdge(source="a", target="b"),
            GraphEdge(source="a", target="c"),
            GraphEdge(source="b", target="d"),
        ],
    )
    path = compute_critical_path(topo)
    assert path == ["a", "b", "d"]


def test_compute_critical_path_single_node():
    topo = GraphTopology(
        nodes={"a": GraphNode(node_id="a", node_name="a", avg_duration_ms=50)},
    )
    path = compute_critical_path(topo)
    assert path == ["a"]


def test_compute_critical_path_empty():
    topo = GraphTopology()
    assert compute_critical_path(topo) == []


def test_compute_critical_path_linear():
    topo = GraphTopology(
        nodes={
            "a": GraphNode(node_id="a", node_name="a", avg_duration_ms=10),
            "b": GraphNode(node_id="b", node_name="b", avg_duration_ms=20),
            "c": GraphNode(node_id="c", node_name="c", avg_duration_ms=30),
        },
        edges=[
            GraphEdge(source="a", target="b"),
            GraphEdge(source="b", target="c"),
        ],
    )
    path = compute_critical_path(topo)
    assert path == ["a", "b", "c"]


# -- detect_cycles --


def test_detect_cycles_finds_loop():
    topo = GraphTopology(
        nodes={
            "a": GraphNode(node_id="a", node_name="a"),
            "b": GraphNode(node_id="b", node_name="b"),
        },
        edges=[
            GraphEdge(source="a", target="b"),
            GraphEdge(source="b", target="a"),
        ],
    )
    cycles = detect_cycles(topo)
    assert len(cycles) >= 1
    # Flatten all cycle nodes and check both a and b appear
    all_nodes = set()
    for c in cycles:
        all_nodes.update(c)
    assert "a" in all_nodes
    assert "b" in all_nodes


def test_detect_cycles_no_loop():
    topo = GraphTopology(
        nodes={
            "a": GraphNode(node_id="a", node_name="a"),
            "b": GraphNode(node_id="b", node_name="b"),
        },
        edges=[GraphEdge(source="a", target="b")],
    )
    cycles = detect_cycles(topo)
    assert cycles == []


def test_detect_cycles_empty():
    topo = GraphTopology()
    assert detect_cycles(topo) == []


def test_detect_cycles_self_loop():
    topo = GraphTopology(
        nodes={"a": GraphNode(node_id="a", node_name="a")},
        edges=[GraphEdge(source="a", target="a")],
    )
    cycles = detect_cycles(topo)
    assert len(cycles) >= 1


# -- GraphTopology format_text --


def test_format_text():
    topo = GraphTopology(
        nodes={
            "a:node": GraphNode(
                node_id="a:node", node_name="a", node_type="node",
                execution_count=1, avg_duration_ms=50.0,
            ),
        },
        edges=[],
        entry_nodes=["a:node"],
        exit_nodes=["a:node"],
    )
    text = topo.format_text()
    assert "Nodes: 1" in text
    assert "Edges: 0" in text
    assert "Entry: a:node" in text


def test_format_text_with_edges():
    topo = GraphTopology(
        nodes={
            "a": GraphNode(node_id="a", node_name="a"),
            "b": GraphNode(node_id="b", node_name="b"),
        },
        edges=[GraphEdge(source="a", target="b", count=3, avg_latency_ms=12.5)],
    )
    text = topo.format_text()
    assert "a -> b" in text
    assert "count=3" in text


# -- as_dict / from_dict roundtrips --


def test_graph_node_as_dict():
    n = GraphNode(node_id="x", node_name="step", node_type="llm", execution_count=5, avg_duration_ms=42.0)
    d = n.as_dict()
    assert d["node_id"] == "x"
    assert d["avg_duration_ms"] == 42.0


def test_graph_edge_as_dict():
    e = GraphEdge(source="a", target="b", count=2, avg_latency_ms=10.0)
    d = e.as_dict()
    assert d["source"] == "a"
    assert d["count"] == 2


def test_graph_topology_roundtrip():
    topo = GraphTopology(
        nodes={
            "a": GraphNode(node_id="a", node_name="a", node_type="node", execution_count=1, avg_duration_ms=10.0),
            "b": GraphNode(node_id="b", node_name="b", node_type="llm", execution_count=2, avg_duration_ms=20.0),
        },
        edges=[GraphEdge(source="a", target="b", count=1, avg_latency_ms=5.0)],
        entry_nodes=["a"],
        exit_nodes=["b"],
    )
    d = topo.as_dict()
    restored = GraphTopology.from_dict(d)
    assert len(restored.nodes) == 2
    assert len(restored.edges) == 1
    assert restored.entry_nodes == ["a"]
    assert restored.exit_nodes == ["b"]
    assert restored.nodes["b"].node_type == "llm"
    assert restored.edges[0].avg_latency_ms == 5.0
