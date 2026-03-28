from __future__ import annotations

from datetime import datetime, timezone, timedelta

from evalyn_sdk.models import Span
from evalyn_sdk.trace.flame_graph import (
    FlameNode,
    FlameGraph,
    build_flame_graph,
    render_ascii,
    render_svg,
    SPAN_TYPE_COLORS,
)


def _now():
    return datetime.now(timezone.utc)


def _span(
    sid,
    name="s",
    span_type="custom",
    parent_id=None,
    start_offset_ms=0,
    duration_ms=100,
):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t = base + timedelta(milliseconds=start_offset_ms)
    return Span(
        id=sid,
        name=name,
        span_type=span_type,
        parent_id=parent_id,
        start_time=t,
        end_time=t + timedelta(milliseconds=duration_ms),
    )


# --- build_flame_graph ---


def test_empty_spans():
    g = build_flame_graph([])
    assert g.root_nodes == []
    assert g.total_duration_ms == 0.0


def test_single_span():
    spans = [_span("a", name="root", duration_ms=200)]
    g = build_flame_graph(spans)
    assert len(g.root_nodes) == 1
    assert g.root_nodes[0].span_name == "root"
    assert g.root_nodes[0].duration_ms == 200.0
    assert g.total_duration_ms == 200.0


def test_parent_child_hierarchy():
    spans = [
        _span("p", name="parent", duration_ms=300),
        _span("c", name="child", parent_id="p", start_offset_ms=50, duration_ms=100),
    ]
    g = build_flame_graph(spans)
    assert len(g.root_nodes) == 1
    root = g.root_nodes[0]
    assert root.span_name == "parent"
    assert len(root.children) == 1
    assert root.children[0].span_name == "child"
    assert root.children[0].depth == 1


def test_multiple_roots():
    spans = [
        _span("a", name="root1", duration_ms=100),
        _span("b", name="root2", start_offset_ms=200, duration_ms=100),
    ]
    g = build_flame_graph(spans)
    assert len(g.root_nodes) == 2


def test_orphan_spans_treated_as_roots():
    spans = [
        _span("a", name="orphan", parent_id="nonexistent", duration_ms=50),
    ]
    g = build_flame_graph(spans)
    assert len(g.root_nodes) == 1
    assert g.root_nodes[0].span_name == "orphan"


def test_depth_correct():
    spans = [
        _span("a", name="root", duration_ms=500),
        _span("b", name="mid", parent_id="a", start_offset_ms=10, duration_ms=400),
        _span("c", name="leaf", parent_id="b", start_offset_ms=20, duration_ms=100),
    ]
    g = build_flame_graph(spans)
    root = g.root_nodes[0]
    assert root.depth == 0
    assert root.children[0].depth == 1
    assert root.children[0].children[0].depth == 2


def test_total_duration_correct():
    spans = [
        _span("a", start_offset_ms=0, duration_ms=100),
        _span("b", start_offset_ms=50, duration_ms=200),
    ]
    g = build_flame_graph(spans)
    # b starts at 50, lasts 200, ends at 250
    assert g.total_duration_ms == 250.0


def test_relative_start_times():
    spans = [
        _span("a", start_offset_ms=100, duration_ms=50),
        _span("b", start_offset_ms=200, duration_ms=50),
    ]
    g = build_flame_graph(spans)
    # Earliest is 100ms offset, so 'a' starts at 0, 'b' at 100
    assert g.root_nodes[0].start_ms == 0.0
    assert g.root_nodes[1].start_ms == 100.0


def test_span_with_no_end_time():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    s = Span(id="x", name="open", span_type="custom", parent_id=None, start_time=base)
    g = build_flame_graph([s])
    assert len(g.root_nodes) == 1
    assert g.root_nodes[0].duration_ms == 0.0


# --- FlameNode as_dict / from_dict ---


def test_flame_node_as_dict():
    node = FlameNode(
        span_id="a", span_name="test", span_type="llm_call",
        start_ms=0.0, duration_ms=100.0, depth=0,
    )
    d = node.as_dict()
    assert d["span_id"] == "a"
    assert d["span_type"] == "llm_call"
    assert d["children"] == []


def test_flame_node_with_children():
    child = FlameNode(
        span_id="b", span_name="child", span_type="tool_call",
        start_ms=10.0, duration_ms=50.0, depth=1,
    )
    parent = FlameNode(
        span_id="a", span_name="parent", span_type="node",
        start_ms=0.0, duration_ms=100.0, depth=0, children=[child],
    )
    d = parent.as_dict()
    assert len(d["children"]) == 1
    assert d["children"][0]["span_id"] == "b"


# --- FlameGraph as_dict / from_dict ---


def test_flame_graph_as_dict():
    node = FlameNode(
        span_id="a", span_name="r", span_type="custom",
        start_ms=0, duration_ms=100, depth=0,
    )
    g = FlameGraph(root_nodes=[node], total_duration_ms=100.0)
    d = g.as_dict()
    assert d["total_duration_ms"] == 100.0
    assert len(d["root_nodes"]) == 1


def test_flame_graph_roundtrip():
    child = FlameNode(
        span_id="b", span_name="child", span_type="tool_call",
        start_ms=10.0, duration_ms=50.0, depth=1,
    )
    root = FlameNode(
        span_id="a", span_name="root", span_type="agent",
        start_ms=0.0, duration_ms=100.0, depth=0, children=[child],
    )
    g = FlameGraph(root_nodes=[root], total_duration_ms=100.0)
    restored = FlameGraph.from_dict(g.as_dict())
    assert len(restored.root_nodes) == 1
    assert restored.root_nodes[0].span_name == "root"
    assert len(restored.root_nodes[0].children) == 1
    assert restored.root_nodes[0].children[0].span_name == "child"
    assert restored.total_duration_ms == 100.0


# --- render_ascii ---


def test_render_ascii_empty():
    g = FlameGraph(root_nodes=[], total_duration_ms=0.0)
    out = render_ascii(g)
    assert "empty" in out.lower()


def test_render_ascii_produces_output():
    spans = [_span("a", name="root", duration_ms=200)]
    g = build_flame_graph(spans)
    out = render_ascii(g)
    assert "root" in out
    assert "[" in out
    assert "]" in out


def test_render_ascii_width_proportional():
    spans = [
        _span("a", name="long", duration_ms=400),
        _span("b", name="short", start_offset_ms=0, parent_id="a", duration_ms=100),
    ]
    g = build_flame_graph(spans)
    out = render_ascii(g, width=80)
    lines = out.strip().split("\n")
    # Parent bar should be wider than child bar
    parent_line = lines[0]
    child_line = lines[1]
    parent_bar_len = parent_line.count("=")
    child_bar_len = child_line.count("=")
    assert parent_bar_len > child_bar_len


def test_render_ascii_indentation():
    spans = [
        _span("a", name="root", duration_ms=200),
        _span("b", name="child", parent_id="a", duration_ms=100),
    ]
    g = build_flame_graph(spans)
    out = render_ascii(g)
    lines = out.strip().split("\n")
    # Child should be indented more than root
    assert lines[1].startswith("  ")
    assert not lines[0].startswith("  ")


# --- render_svg ---


def test_render_svg_empty():
    g = FlameGraph(root_nodes=[], total_duration_ms=0.0)
    svg = render_svg(g)
    assert "<svg" in svg
    assert 'width="0"' in svg


def test_render_svg_produces_valid_svg():
    spans = [_span("a", name="root", span_type="llm_call", duration_ms=200)]
    g = build_flame_graph(spans)
    svg = render_svg(g)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert "<rect" in svg


def test_render_svg_has_color():
    spans = [_span("a", name="root", span_type="llm_call", duration_ms=200)]
    g = build_flame_graph(spans)
    svg = render_svg(g)
    assert SPAN_TYPE_COLORS["llm_call"] in svg


def test_render_svg_multiple_types():
    spans = [
        _span("a", name="agent", span_type="agent", duration_ms=300),
        _span("b", name="tool", span_type="tool_call", parent_id="a", duration_ms=100),
    ]
    g = build_flame_graph(spans)
    svg = render_svg(g)
    assert SPAN_TYPE_COLORS["agent"] in svg
    assert SPAN_TYPE_COLORS["tool_call"] in svg


def test_render_svg_contains_span_name():
    spans = [_span("a", name="my_func", span_type="custom", duration_ms=200)]
    g = build_flame_graph(spans)
    svg = render_svg(g)
    assert "my_func" in svg


def test_render_svg_custom_dimensions():
    spans = [_span("a", name="r", duration_ms=100)]
    g = build_flame_graph(spans)
    svg = render_svg(g, width=800, row_height=30)
    assert 'width="800"' in svg


# --- SPAN_TYPE_COLORS ---


def test_span_type_colors_has_expected_keys():
    expected = {"llm_call", "tool_call", "retrieval", "node", "agent", "custom", "scorer"}
    assert set(SPAN_TYPE_COLORS.keys()) == expected
