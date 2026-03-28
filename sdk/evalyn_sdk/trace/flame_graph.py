from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from html import escape as html_escape

from ..models import Span

SPAN_TYPE_COLORS = {
    "llm_call": "#E94560",
    "tool_call": "#2ECC71",
    "retrieval": "#3498DB",
    "node": "#F39C12",
    "agent": "#9B59B6",
    "custom": "#95A5A6",
    "scorer": "#1ABC9C",
}


@dataclass
class FlameNode:
    """A node in the flame graph tree."""

    span_id: str
    span_name: str
    span_type: str
    start_ms: float
    duration_ms: float
    depth: int
    children: List[FlameNode] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "span_name": self.span_name,
            "span_type": self.span_type,
            "start_ms": self.start_ms,
            "duration_ms": self.duration_ms,
            "depth": self.depth,
            "children": [c.as_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FlameNode:
        return cls(
            span_id=data["span_id"],
            span_name=data["span_name"],
            span_type=data["span_type"],
            start_ms=data["start_ms"],
            duration_ms=data["duration_ms"],
            depth=data["depth"],
            children=[cls.from_dict(c) for c in data.get("children", [])],
        )


@dataclass
class FlameGraph:
    """A flame graph built from spans."""

    root_nodes: List[FlameNode] = field(default_factory=list)
    total_duration_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "root_nodes": [n.as_dict() for n in self.root_nodes],
            "total_duration_ms": self.total_duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FlameGraph:
        return cls(
            root_nodes=[FlameNode.from_dict(n) for n in data.get("root_nodes", [])],
            total_duration_ms=data.get("total_duration_ms", 0.0),
        )


def build_flame_graph(spans: List[Span]) -> FlameGraph:
    """Build a flame graph from a list of spans.

    Computes relative start times from the earliest span and builds a tree
    from parent_id relationships. Orphan spans (parent not found) are treated
    as roots.
    """
    if not spans:
        return FlameGraph(root_nodes=[], total_duration_ms=0.0)

    # Find the earliest start time as the reference
    earliest = min(s.start_time for s in spans)

    # Build span info keyed by id
    span_info: Dict[str, Dict[str, Any]] = {}
    for s in spans:
        start_ms = (s.start_time - earliest).total_seconds() * 1000
        duration_ms = s.duration_ms if s.duration_ms is not None else 0.0
        span_info[s.id] = {
            "span": s,
            "start_ms": start_ms,
            "duration_ms": duration_ms,
        }

    # Build tree - compute depth and collect children
    children_map: Dict[str, List[str]] = {s.id: [] for s in spans}
    root_ids: List[str] = []

    for s in spans:
        if s.parent_id and s.parent_id in span_info:
            children_map[s.parent_id].append(s.id)
        else:
            root_ids.append(s.id)

    def _build_node(span_id: str, depth: int) -> FlameNode:
        info = span_info[span_id]
        s = info["span"]
        child_nodes = [
            _build_node(cid, depth + 1) for cid in children_map[span_id]
        ]
        return FlameNode(
            span_id=s.id,
            span_name=s.name,
            span_type=s.span_type,
            start_ms=info["start_ms"],
            duration_ms=info["duration_ms"],
            depth=depth,
            children=child_nodes,
        )

    roots = [_build_node(rid, 0) for rid in root_ids]

    # Total duration: from earliest start to latest end
    latest_end_ms = 0.0
    for info in span_info.values():
        end_ms = info["start_ms"] + info["duration_ms"]
        if end_ms > latest_end_ms:
            latest_end_ms = end_ms

    return FlameGraph(root_nodes=roots, total_duration_ms=latest_end_ms)


def render_ascii(graph: FlameGraph, width: int = 80) -> str:
    """Render a flame graph as ASCII art.

    Each span is a [====name====] bar where width is proportional to duration.
    Depth is shown by indentation.
    """
    if not graph.root_nodes:
        return "(empty flame graph)"

    total = graph.total_duration_ms
    if total <= 0:
        total = 1.0  # avoid division by zero

    lines: List[str] = []

    def _render_node(node: FlameNode) -> None:
        indent = "  " * node.depth
        available = width - len(indent) - 2  # subtract brackets
        if available < 1:
            available = 1

        bar_width = max(1, int((node.duration_ms / total) * available))
        name = node.span_name
        # Truncate name if needed
        if len(name) > bar_width - 2:
            name = name[: max(0, bar_width - 2)]
        # Center name in bar
        padding = bar_width - len(name)
        left_pad = padding // 2
        right_pad = padding - left_pad
        bar = "[" + "=" * left_pad + name + "=" * right_pad + "]"
        lines.append(f"{indent}{bar}")

        for child in node.children:
            _render_node(child)

    for root in graph.root_nodes:
        _render_node(root)

    return "\n".join(lines)


def render_svg(
    graph: FlameGraph, width: int = 1200, row_height: int = 24
) -> str:
    """Render a flame graph as an SVG string.

    Each span is a colored rectangle. Width is proportional to duration.
    Color is determined by SPAN_TYPE_COLORS.
    """
    if not graph.root_nodes:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0"></svg>'

    total = graph.total_duration_ms
    if total <= 0:
        total = 1.0

    rects: List[str] = []
    max_depth = 0

    def _render_node(node: FlameNode) -> None:
        nonlocal max_depth
        if node.depth > max_depth:
            max_depth = node.depth

        x = (node.start_ms / total) * width
        w = max(1, (node.duration_ms / total) * width)
        y = node.depth * row_height
        color = SPAN_TYPE_COLORS.get(node.span_type, SPAN_TYPE_COLORS["custom"])
        name = html_escape(node.span_name)
        dur_text = html_escape(f"{node.duration_ms:.1f}ms")

        rects.append(
            f'  <g>'
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{row_height - 2}" '
            f'fill="{color}" rx="2" />'
            f'<text x="{x + 3:.1f}" y="{y + row_height - 6}" '
            f'font-size="11" fill="#fff" clip-path="url(#clip)">'
            f'{name} ({dur_text})</text>'
            f'</g>'
        )

        for child in node.children:
            _render_node(child)

    for root in graph.root_nodes:
        _render_node(root)

    height = (max_depth + 1) * row_height + 4

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'  <style>text {{ font-family: monospace; }}</style>',
    ]
    svg_parts.extend(rects)
    svg_parts.append("</svg>")

    return "\n".join(svg_parts)
