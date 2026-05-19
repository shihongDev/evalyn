"""Span dependency graph construction.

Detects causal data flow between spans within a trace by analyzing
content overlap between span outputs and inputs. Builds a directed
dependency graph and identifies bottleneck spans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Span


@dataclass
class DependencyEdge:
    """A directed edge from source span to target span."""

    source_id: str
    target_id: str
    overlap_ratio: float  # 0-1: how much of source output appears in target input

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "overlap_ratio": round(self.overlap_ratio, 4),
        }


@dataclass
class SpanNode:
    """A span with dependency context."""

    span_id: str
    span_name: str
    span_type: str
    upstream: list[str] = field(default_factory=list)  # span IDs this depends on
    downstream: list[str] = field(default_factory=list)  # span IDs that depend on this
    is_bottleneck: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "span_name": self.span_name,
            "span_type": self.span_type,
            "upstream_count": len(self.upstream),
            "downstream_count": len(self.downstream),
            "is_bottleneck": self.is_bottleneck,
        }


@dataclass
class DependencyGraph:
    """Directed dependency graph of spans within a trace."""

    nodes: dict[str, SpanNode] = field(default_factory=dict)
    edges: list[DependencyEdge] = field(default_factory=list)

    @property
    def bottlenecks(self) -> list[str]:
        return [nid for nid, n in self.nodes.items() if n.is_bottleneck]

    @property
    def roots(self) -> list[str]:
        """Spans with no upstream dependencies."""
        return [nid for nid, n in self.nodes.items() if not n.upstream]

    @property
    def leaves(self) -> list[str]:
        """Spans with no downstream dependents."""
        return [nid for nid, n in self.nodes.items() if not n.downstream]

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "bottlenecks": self.bottlenecks,
            "roots": self.roots,
            "leaves": self.leaves,
            "edges": [e.as_dict() for e in self.edges],
            "nodes": {nid: n.as_dict() for nid, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DependencyGraph:
        edges = [
            DependencyEdge(
                source_id=e["source_id"],
                target_id=e["target_id"],
                overlap_ratio=e.get("overlap_ratio", 0.0),
            )
            for e in data.get("edges", [])
        ]
        nodes = {}
        for nid, nd in data.get("nodes", {}).items():
            nodes[nid] = SpanNode(
                span_id=nd["span_id"],
                span_name=nd["span_name"],
                span_type=nd["span_type"],
                is_bottleneck=nd.get("is_bottleneck", False),
            )
        # Rebuild upstream/downstream from edges
        for e in edges:
            if e.source_id in nodes:
                nodes[e.source_id].downstream.append(e.target_id)
            if e.target_id in nodes:
                nodes[e.target_id].upstream.append(e.source_id)
        return cls(nodes=nodes, edges=edges)

    def to_mermaid(self) -> str:
        """Render as Mermaid graph."""
        lines = ["graph TD"]
        for nid, node in self.nodes.items():
            label = f"{node.span_name} ({node.span_type})"
            if node.is_bottleneck:
                lines.append(f"    {nid}[[\"{label}\"]]")
            else:
                lines.append(f"    {nid}[\"{label}\"]")
        for e in self.edges:
            pct = int(e.overlap_ratio * 100)
            lines.append(f"    {e.source_id} -->|{pct}%| {e.target_id}")
        return "\n".join(lines)

    def format_text(self) -> str:
        lines = [f"Dependency Graph: {len(self.nodes)} nodes, {len(self.edges)} edges"]
        if self.bottlenecks:
            bnames = [self.nodes[b].span_name for b in self.bottlenecks]
            lines.append(f"  Bottlenecks: {', '.join(bnames)}")
        for e in self.edges:
            src = self.nodes.get(e.source_id)
            tgt = self.nodes.get(e.target_id)
            src_name = src.span_name if src else e.source_id
            tgt_name = tgt.span_name if tgt else e.target_id
            lines.append(f"  {src_name} -> {tgt_name} ({e.overlap_ratio:.0%})")
        return "\n".join(lines)


def build_dependency_graph(
    spans: list[Span],
    min_overlap: float = 0.1,
    bottleneck_threshold: int = 2,
) -> DependencyGraph:
    """Build a dependency graph from span content overlap.

    Analyzes span attributes for "output"/"input" keys and detects
    when one span's output text appears in another span's input.

    Args:
        spans: Spans from a single trace.
        min_overlap: Minimum overlap ratio to create an edge (0-1).
        bottleneck_threshold: Minimum downstream count to mark as bottleneck.

    Returns:
        DependencyGraph with nodes, edges, and bottleneck flags.
    """
    graph = DependencyGraph()

    # Create nodes
    for s in spans:
        graph.nodes[s.id] = SpanNode(
            span_id=s.id,
            span_name=s.name,
            span_type=s.span_type,
        )

    # Extract output/input text from attributes
    outputs: dict[str, str] = {}
    inputs: dict[str, str] = {}
    for s in spans:
        out_text = _extract_text(s.attributes, "output")
        if out_text:
            outputs[s.id] = out_text
        in_text = _extract_text(s.attributes, "input")
        if in_text:
            inputs[s.id] = in_text

    # Detect content overlap
    for src_id, src_output in outputs.items():
        if len(src_output) < 5:  # skip trivially short outputs
            continue
        for tgt_id, tgt_input in inputs.items():
            if src_id == tgt_id:
                continue
            ratio = _overlap_ratio(src_output, tgt_input)
            if ratio >= min_overlap:
                graph.edges.append(DependencyEdge(
                    source_id=src_id,
                    target_id=tgt_id,
                    overlap_ratio=ratio,
                ))
                graph.nodes[src_id].downstream.append(tgt_id)
                graph.nodes[tgt_id].upstream.append(src_id)

    # Also add parent-child structural edges (weaker signal)
    existing_pairs: set[tuple[str, str]] = {
        (e.source_id, e.target_id) for e in graph.edges
    }
    for s in spans:
        if s.parent_id and s.parent_id in graph.nodes:
            pair = (s.parent_id, s.id)
            if pair not in existing_pairs:
                graph.edges.append(DependencyEdge(
                    source_id=s.parent_id,
                    target_id=s.id,
                    overlap_ratio=0.0,  # structural, not content-based
                ))
                graph.nodes[s.parent_id].downstream.append(s.id)
                graph.nodes[s.id].upstream.append(s.parent_id)

    # Mark bottlenecks
    for nid, node in graph.nodes.items():
        if len(node.downstream) >= bottleneck_threshold:
            node.is_bottleneck = True

    return graph


def _extract_text(attributes: dict[str, Any], key: str) -> str:
    """Extract text from span attributes, handling various formats."""
    val = attributes.get(key)
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        # Try common sub-keys
        for subkey in ("text", "content", "message", "value"):
            if subkey in val:
                return str(val[subkey])
        return str(val)
    return str(val)


def _overlap_ratio(source: str, target: str) -> float:
    """Compute what fraction of source appears in target.

    Uses substring matching with a sliding window approach.
    Returns ratio of source characters found in target.
    """
    if not source or not target:
        return 0.0

    source_lower = source.lower()
    target_lower = target.lower()

    # Direct substring check
    if source_lower in target_lower:
        return 1.0

    # Check for significant substrings (word-level)
    source_words = set(source_lower.split())
    target_words = set(target_lower.split())

    if not source_words:
        return 0.0

    common = source_words & target_words
    # Filter out very short words (articles, prepositions)
    meaningful_common = {w for w in common if len(w) > 3}
    meaningful_source = {w for w in source_words if len(w) > 3}

    if not meaningful_source:
        return 0.0

    return len(meaningful_common) / len(meaningful_source)
