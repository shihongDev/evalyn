"""Agent decision tree visualization.

Render agent tool-selection choices as a tree from span traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Span

# Span types that map to each node type
_DECISION_TYPES = {"llm_call", "agent"}
_ACTION_TYPES = {"tool_call", "tool_use", "tool_result", "retrieval"}
_OUTCOME_TYPES = {"output_message"}


def _classify_span(span: Span) -> str:
    """Classify a span into decision/action/outcome."""
    if span.span_type in _DECISION_TYPES:
        return "decision"
    if span.span_type in _ACTION_TYPES:
        return "action"
    if span.span_type in _OUTCOME_TYPES:
        return "outcome"
    # Default: treat as decision for unknown types
    return "decision"


@dataclass
class DecisionNode:
    """A node in the decision tree."""

    node_id: str
    label: str
    node_type: str = "decision"  # decision / action / outcome
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type,
            "children": list(self.children),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionNode:
        return cls(
            node_id=data["node_id"],
            label=data["label"],
            node_type=data.get("node_type", "decision"),
            children=data.get("children", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DecisionTree:
    """Complete decision tree extracted from spans."""

    nodes: dict[str, DecisionNode] = field(default_factory=dict)
    root_id: str = ""
    total_decisions: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": {nid: n.as_dict() for nid, n in self.nodes.items()},
            "root_id": self.root_id,
            "total_decisions": self.total_decisions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionTree:
        nodes: dict[str, DecisionNode] = {}
        for nid, ndata in data.get("nodes", {}).items():
            nodes[nid] = DecisionNode.from_dict(ndata)
        return cls(
            nodes=nodes,
            root_id=data.get("root_id", ""),
            total_decisions=data.get("total_decisions", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(f"Decision Tree ({len(self.nodes)} nodes)")
        lines.append(f"Root: {self.root_id}")
        lines.append(f"Total decisions: {self.total_decisions}")
        lines.append("")
        for node in self.nodes.values():
            children_str = ", ".join(node.children) if node.children else "none"
            lines.append(
                f"  [{node.node_id}] {node.label} "
                f"(type={node.node_type}, children={children_str})"
            )
        return "\n".join(lines)


def build_decision_tree(spans: list[Span]) -> DecisionTree:
    """Build a decision tree from span sequence.

    LLM calls become decision nodes, tool calls become action nodes,
    final outputs become outcome nodes. Parent-child links are
    derived from span parent_id relationships.
    """
    if not spans:
        return DecisionTree()

    span_by_id = {s.id: s for s in spans}
    nodes: dict[str, DecisionNode] = {}

    for s in spans:
        ntype = _classify_span(s)
        nodes[s.id] = DecisionNode(
            node_id=s.id,
            label=s.name,
            node_type=ntype,
            children=[],
            metadata=dict(s.attributes),
        )

    # Wire parent -> child relationships
    for s in spans:
        if s.parent_id and s.parent_id in nodes:
            nodes[s.parent_id].children.append(s.id)

    # Find root: spans with no parent or whose parent is not in our set
    roots = [
        s.id for s in spans
        if s.parent_id is None or s.parent_id not in span_by_id
    ]
    root_id = roots[0] if roots else ""

    total_decisions = sum(
        1 for n in nodes.values() if n.node_type == "decision"
    )

    return DecisionTree(
        nodes=nodes,
        root_id=root_id,
        total_decisions=total_decisions,
    )


def render_tree_ascii(tree: DecisionTree, indent: int = 2) -> str:
    """Render the decision tree as an ASCII tree with indentation."""
    if not tree.nodes or not tree.root_id:
        return ""

    lines: list[str] = []

    def _render(node_id: str, depth: int) -> None:
        node = tree.nodes.get(node_id)
        if node is None:
            return
        prefix = " " * (depth * indent)
        type_marker = {"decision": "?", "action": ">", "outcome": "*"}.get(
            node.node_type, "-"
        )
        lines.append(f"{prefix}[{type_marker}] {node.label}")
        for child_id in node.children:
            _render(child_id, depth + 1)

    _render(tree.root_id, 0)
    return "\n".join(lines)


def render_tree_mermaid(tree: DecisionTree) -> str:
    """Render the decision tree as a Mermaid flowchart."""
    if not tree.nodes:
        return "graph TD"

    lines: list[str] = ["graph TD"]

    shape_map = {
        "decision": ('{"', '"}'),   # diamond-ish via double brace
        "action": ("[", "]"),        # rectangle
        "outcome": ("([", "])"),     # stadium
    }

    for node in tree.nodes.values():
        left, right = shape_map.get(node.node_type, ("[", "]"))
        safe_id = node.node_id.replace("-", "_")
        lines.append(f"    {safe_id}{left}{node.label}{right}")

    for node in tree.nodes.values():
        src = node.node_id.replace("-", "_")
        for child_id in node.children:
            tgt = child_id.replace("-", "_")
            lines.append(f"    {src} --> {tgt}")

    return "\n".join(lines)


def find_decision_paths(tree: DecisionTree) -> list[list[str]]:
    """Find all root-to-leaf paths in the decision tree."""
    if not tree.nodes or not tree.root_id:
        return []

    paths: list[list[str]] = []

    def _walk(node_id: str, current_path: list[str]) -> None:
        node = tree.nodes.get(node_id)
        if node is None:
            return
        current_path.append(node_id)
        if not node.children:
            paths.append(list(current_path))
        else:
            for child_id in node.children:
                _walk(child_id, current_path)
        current_path.pop()

    _walk(tree.root_id, [])
    return paths


def compute_decision_stats(tree: DecisionTree) -> dict[str, Any]:
    """Compute tree statistics: depth, breadth, nodes by type, avg branching factor."""
    if not tree.nodes:
        return {
            "depth": 0,
            "breadth": 0,
            "total_nodes": 0,
            "nodes_by_type": {},
            "avg_branching_factor": 0.0,
        }

    # Nodes by type
    nodes_by_type: dict[str, int] = {}
    for node in tree.nodes.values():
        nodes_by_type[node.node_type] = nodes_by_type.get(node.node_type, 0) + 1

    # Depth via BFS from root
    depth = 0
    if tree.root_id and tree.root_id in tree.nodes:
        level = [tree.root_id]
        while level:
            depth += 1
            next_level: list[str] = []
            for nid in level:
                node = tree.nodes.get(nid)
                if node:
                    next_level.extend(node.children)
            level = next_level

    # Breadth: max children count across all nodes
    breadth = max(
        (len(n.children) for n in tree.nodes.values()), default=0
    )

    # Avg branching factor: average children count of non-leaf nodes
    non_leaf = [n for n in tree.nodes.values() if n.children]
    avg_branching = (
        sum(len(n.children) for n in non_leaf) / len(non_leaf)
        if non_leaf
        else 0.0
    )

    return {
        "depth": depth,
        "breadth": breadth,
        "total_nodes": len(tree.nodes),
        "nodes_by_type": nodes_by_type,
        "avg_branching_factor": avg_branching,
    }
