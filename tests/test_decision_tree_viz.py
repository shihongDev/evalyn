"""Tests for decision_tree_viz module."""

from datetime import datetime, timezone

from evalyn_sdk.models import Span
from evalyn_sdk.analysis.decision_tree_viz import (
    DecisionNode,
    DecisionTree,
    build_decision_tree,
    render_tree_ascii,
    render_tree_mermaid,
    find_decision_paths,
    compute_decision_stats,
)


def _span(sid, name="agent", span_type="llm_call", parent_id=None, **attrs):
    return Span(
        id=sid,
        name=name,
        span_type=span_type,
        parent_id=parent_id,
        start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# --- build_decision_tree ---


def test_build_decision_tree_basic():
    spans = [
        _span("root", name="planner", span_type="llm_call"),
        _span("t1", name="search", span_type="tool_call", parent_id="root"),
        _span("t2", name="calc", span_type="tool_call", parent_id="root"),
    ]
    tree = build_decision_tree(spans)
    assert tree.root_id == "root"
    assert len(tree.nodes) == 3
    assert "t1" in tree.nodes["root"].children
    assert "t2" in tree.nodes["root"].children


def test_build_decision_tree_node_types():
    spans = [
        _span("d1", span_type="llm_call"),
        _span("a1", span_type="tool_call", parent_id="d1"),
        _span("o1", span_type="output_message", parent_id="d1"),
    ]
    tree = build_decision_tree(spans)
    assert tree.nodes["d1"].node_type == "decision"
    assert tree.nodes["a1"].node_type == "action"
    assert tree.nodes["o1"].node_type == "outcome"


def test_build_decision_tree_total_decisions():
    spans = [
        _span("d1", span_type="llm_call"),
        _span("d2", span_type="agent", parent_id="d1"),
        _span("a1", span_type="tool_call", parent_id="d2"),
    ]
    tree = build_decision_tree(spans)
    assert tree.total_decisions == 2


def test_build_decision_tree_metadata_preserved():
    spans = [_span("s1", span_type="llm_call", model="gpt-4")]
    tree = build_decision_tree(spans)
    assert tree.nodes["s1"].metadata.get("model") == "gpt-4"


def test_build_decision_tree_deep_chain():
    spans = [
        _span("a", span_type="llm_call"),
        _span("b", span_type="tool_call", parent_id="a"),
        _span("c", span_type="llm_call", parent_id="b"),
        _span("d", span_type="tool_call", parent_id="c"),
    ]
    tree = build_decision_tree(spans)
    assert tree.root_id == "a"
    assert "b" in tree.nodes["a"].children
    assert "c" in tree.nodes["b"].children
    assert "d" in tree.nodes["c"].children


# --- render_tree_ascii ---


def test_render_tree_ascii_basic():
    spans = [
        _span("root", name="planner", span_type="llm_call"),
        _span("t1", name="search", span_type="tool_call", parent_id="root"),
    ]
    tree = build_decision_tree(spans)
    output = render_tree_ascii(tree)
    assert "[?] planner" in output
    assert "[>] search" in output


def test_render_tree_ascii_indentation():
    spans = [
        _span("root", name="root", span_type="llm_call"),
        _span("c1", name="child", span_type="tool_call", parent_id="root"),
    ]
    tree = build_decision_tree(spans)
    output = render_tree_ascii(tree, indent=4)
    lines = output.split("\n")
    # root at depth 0, child at depth 1 with indent=4
    assert lines[0].startswith("[?]")
    assert lines[1].startswith("    [>]")


def test_render_tree_ascii_custom_indent():
    spans = [
        _span("r", name="r", span_type="llm_call"),
        _span("c", name="c", span_type="tool_call", parent_id="r"),
    ]
    tree = build_decision_tree(spans)
    output = render_tree_ascii(tree, indent=3)
    lines = output.split("\n")
    assert lines[1].startswith("   [>]")


def test_render_tree_ascii_empty():
    tree = DecisionTree()
    assert render_tree_ascii(tree) == ""


def test_render_tree_ascii_outcome_marker():
    spans = [
        _span("r", name="decide", span_type="llm_call"),
        _span("o", name="result", span_type="output_message", parent_id="r"),
    ]
    tree = build_decision_tree(spans)
    output = render_tree_ascii(tree)
    assert "[*] result" in output


# --- render_tree_mermaid ---


def test_render_tree_mermaid_header():
    spans = [_span("s1", name="node1", span_type="llm_call")]
    tree = build_decision_tree(spans)
    output = render_tree_mermaid(tree)
    assert output.startswith("graph TD")


def test_render_tree_mermaid_edges():
    spans = [
        _span("r", name="root", span_type="llm_call"),
        _span("c", name="child", span_type="tool_call", parent_id="r"),
    ]
    tree = build_decision_tree(spans)
    output = render_tree_mermaid(tree)
    assert "-->" in output


def test_render_tree_mermaid_empty():
    tree = DecisionTree()
    assert render_tree_mermaid(tree) == "graph TD"


def test_render_tree_mermaid_node_shapes():
    spans = [
        _span("d1", name="decide", span_type="llm_call"),
        _span("a1", name="act", span_type="tool_call", parent_id="d1"),
        _span("o1", name="done", span_type="output_message", parent_id="d1"),
    ]
    tree = build_decision_tree(spans)
    output = render_tree_mermaid(tree)
    # decision nodes use {" "}
    assert '{"decide"}' in output
    # action nodes use [ ]
    assert "[act]" in output
    # outcome nodes use ([ ])
    assert "([done])" in output


# --- find_decision_paths ---


def test_find_decision_paths_single_path():
    spans = [
        _span("a", span_type="llm_call"),
        _span("b", span_type="tool_call", parent_id="a"),
        _span("c", span_type="output_message", parent_id="b"),
    ]
    tree = build_decision_tree(spans)
    paths = find_decision_paths(tree)
    assert len(paths) == 1
    assert paths[0] == ["a", "b", "c"]


def test_find_decision_paths_branching():
    spans = [
        _span("root", span_type="llm_call"),
        _span("left", span_type="tool_call", parent_id="root"),
        _span("right", span_type="tool_call", parent_id="root"),
    ]
    tree = build_decision_tree(spans)
    paths = find_decision_paths(tree)
    assert len(paths) == 2
    path_sets = [tuple(p) for p in paths]
    assert ("root", "left") in path_sets
    assert ("root", "right") in path_sets


def test_find_decision_paths_empty():
    tree = DecisionTree()
    assert find_decision_paths(tree) == []


def test_find_decision_paths_single_node():
    spans = [_span("solo", span_type="llm_call")]
    tree = build_decision_tree(spans)
    paths = find_decision_paths(tree)
    assert paths == [["solo"]]


# --- compute_decision_stats ---


def test_compute_decision_stats_depth():
    spans = [
        _span("a", span_type="llm_call"),
        _span("b", span_type="tool_call", parent_id="a"),
        _span("c", span_type="output_message", parent_id="b"),
    ]
    tree = build_decision_tree(spans)
    stats = compute_decision_stats(tree)
    assert stats["depth"] == 3


def test_compute_decision_stats_breadth():
    spans = [
        _span("root", span_type="llm_call"),
        _span("c1", span_type="tool_call", parent_id="root"),
        _span("c2", span_type="tool_call", parent_id="root"),
        _span("c3", span_type="tool_call", parent_id="root"),
    ]
    tree = build_decision_tree(spans)
    stats = compute_decision_stats(tree)
    assert stats["breadth"] == 3


def test_compute_decision_stats_nodes_by_type():
    spans = [
        _span("d1", span_type="llm_call"),
        _span("a1", span_type="tool_call", parent_id="d1"),
        _span("a2", span_type="tool_call", parent_id="d1"),
        _span("o1", span_type="output_message", parent_id="d1"),
    ]
    tree = build_decision_tree(spans)
    stats = compute_decision_stats(tree)
    assert stats["nodes_by_type"]["decision"] == 1
    assert stats["nodes_by_type"]["action"] == 2
    assert stats["nodes_by_type"]["outcome"] == 1


def test_compute_decision_stats_avg_branching():
    spans = [
        _span("root", span_type="llm_call"),
        _span("c1", span_type="tool_call", parent_id="root"),
        _span("c2", span_type="tool_call", parent_id="root"),
    ]
    tree = build_decision_tree(spans)
    stats = compute_decision_stats(tree)
    # root has 2 children, only non-leaf node
    assert stats["avg_branching_factor"] == 2.0


def test_compute_decision_stats_total_nodes():
    spans = [
        _span("a", span_type="llm_call"),
        _span("b", span_type="tool_call", parent_id="a"),
    ]
    tree = build_decision_tree(spans)
    stats = compute_decision_stats(tree)
    assert stats["total_nodes"] == 2


def test_compute_decision_stats_empty():
    tree = DecisionTree()
    stats = compute_decision_stats(tree)
    assert stats["depth"] == 0
    assert stats["total_nodes"] == 0
    assert stats["avg_branching_factor"] == 0.0


# --- DecisionNode as_dict / from_dict ---


def test_decision_node_as_dict():
    node = DecisionNode(
        node_id="n1",
        label="planner",
        node_type="decision",
        children=["n2", "n3"],
        metadata={"model": "gpt-4"},
    )
    d = node.as_dict()
    assert d["node_id"] == "n1"
    assert d["children"] == ["n2", "n3"]
    assert d["metadata"]["model"] == "gpt-4"


def test_decision_node_roundtrip():
    node = DecisionNode(
        node_id="n1",
        label="search",
        node_type="action",
        children=["n2"],
        metadata={"tool": "web_search"},
    )
    d = node.as_dict()
    restored = DecisionNode.from_dict(d)
    assert restored.node_id == node.node_id
    assert restored.label == node.label
    assert restored.node_type == node.node_type
    assert restored.children == node.children
    assert restored.metadata == node.metadata


def test_decision_node_from_dict_defaults():
    restored = DecisionNode.from_dict({"node_id": "x", "label": "x"})
    assert restored.node_type == "decision"
    assert restored.children == []
    assert restored.metadata == {}


# --- DecisionTree as_dict / from_dict ---


def test_decision_tree_roundtrip():
    spans = [
        _span("root", name="decide", span_type="llm_call"),
        _span("t1", name="tool", span_type="tool_call", parent_id="root"),
    ]
    original = build_decision_tree(spans)
    d = original.as_dict()
    restored = DecisionTree.from_dict(d)
    assert restored.root_id == original.root_id
    assert restored.total_decisions == original.total_decisions
    assert len(restored.nodes) == len(original.nodes)
    assert "t1" in restored.nodes["root"].children


def test_decision_tree_from_dict_empty():
    restored = DecisionTree.from_dict({})
    assert restored.nodes == {}
    assert restored.root_id == ""


# --- DecisionTree format_text ---


def test_decision_tree_format_text():
    spans = [
        _span("root", name="planner", span_type="llm_call"),
        _span("t1", name="search", span_type="tool_call", parent_id="root"),
    ]
    tree = build_decision_tree(spans)
    text = tree.format_text()
    assert "Decision Tree" in text
    assert "planner" in text
    assert "search" in text


def test_decision_tree_format_text_shows_root():
    spans = [_span("r", name="root_node", span_type="llm_call")]
    tree = build_decision_tree(spans)
    text = tree.format_text()
    assert "Root: r" in text


# --- Edge cases ---


def test_build_decision_tree_empty_spans():
    tree = build_decision_tree([])
    assert tree.nodes == {}
    assert tree.root_id == ""
    assert tree.total_decisions == 0


def test_build_decision_tree_single_span():
    spans = [_span("only", name="sole", span_type="llm_call")]
    tree = build_decision_tree(spans)
    assert len(tree.nodes) == 1
    assert tree.root_id == "only"
    assert tree.nodes["only"].children == []


def test_build_decision_tree_linear_chain():
    spans = [
        _span("a", name="step1", span_type="llm_call"),
        _span("b", name="step2", span_type="tool_call", parent_id="a"),
        _span("c", name="step3", span_type="llm_call", parent_id="b"),
        _span("d", name="step4", span_type="output_message", parent_id="c"),
    ]
    tree = build_decision_tree(spans)
    paths = find_decision_paths(tree)
    assert len(paths) == 1
    assert paths[0] == ["a", "b", "c", "d"]


def test_build_decision_tree_orphan_parent_becomes_root():
    # parent_id refers to a span not in the list
    spans = [_span("x", name="orphan", span_type="llm_call", parent_id="missing")]
    tree = build_decision_tree(spans)
    assert tree.root_id == "x"
