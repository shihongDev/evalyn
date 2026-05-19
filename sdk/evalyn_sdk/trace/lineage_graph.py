from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse_datetime(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


@dataclass
class LineageEdge:
    """A directed edge from source trace to target trace."""

    source_trace_id: str
    target_trace_id: str
    overlap_confidence: float  # 0-1

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_trace_id": self.source_trace_id,
            "target_trace_id": self.target_trace_id,
            "overlap_confidence": self.overlap_confidence,
        }


@dataclass
class LineageNode:
    """A node in the lineage graph representing a single trace."""

    trace_id: str
    timestamp: datetime
    input_hash: str = ""
    output_hash: str = ""
    upstream: list[str] = field(default_factory=list)
    downstream: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": _iso(self.timestamp),
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "upstream": list(self.upstream),
            "downstream": list(self.downstream),
        }


@dataclass
class LineageGraph:
    """A directed acyclic graph of trace lineage relationships."""

    nodes: dict[str, LineageNode] = field(default_factory=dict)
    edges: list[LineageEdge] = field(default_factory=list)

    @property
    def roots(self) -> list[LineageNode]:
        """Nodes with no upstream connections."""
        return [n for n in self.nodes.values() if not n.upstream]

    @property
    def leaves(self) -> list[LineageNode]:
        """Nodes with no downstream connections."""
        return [n for n in self.nodes.values() if not n.downstream]

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": {tid: node.as_dict() for tid, node in self.nodes.items()},
            "edges": [e.as_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageGraph:
        nodes: dict[str, LineageNode] = {}
        for tid, nd in data.get("nodes", {}).items():
            nodes[tid] = LineageNode(
                trace_id=nd["trace_id"],
                timestamp=_parse_datetime(nd["timestamp"]) or datetime.now(timezone.utc),
                input_hash=nd.get("input_hash", ""),
                output_hash=nd.get("output_hash", ""),
                upstream=nd.get("upstream", []),
                downstream=nd.get("downstream", []),
            )
        edges = [
            LineageEdge(
                source_trace_id=e["source_trace_id"],
                target_trace_id=e["target_trace_id"],
                overlap_confidence=e["overlap_confidence"],
            )
            for e in data.get("edges", [])
        ]
        return cls(nodes=nodes, edges=edges)

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(
            f"Lineage graph: {len(self.nodes)} nodes, {len(self.edges)} edges"
        )
        root_ids = [n.trace_id for n in self.roots]
        leaf_ids = [n.trace_id for n in self.leaves]
        lines.append(f"  roots: {root_ids}")
        lines.append(f"  leaves: {leaf_ids}")
        for edge in self.edges:
            lines.append(
                f"  {edge.source_trace_id} -> {edge.target_trace_id} "
                f"(confidence={edge.overlap_confidence:.2f})"
            )
        return "\n".join(lines)


def compute_content_hash(text: str) -> str:
    """SHA-256 hash of text, first 16 hex chars."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_lineage_graph(traces: list[dict[str, Any]]) -> LineageGraph:
    """Build a lineage graph from traces.

    Each trace dict has "id", "timestamp" (datetime), "input_text" (str),
    "output_text" (str). Connect traces where one's output_text appears as
    another's input_text (content overlap). Only consider source timestamp
    < target timestamp.
    """
    graph = LineageGraph()

    # Create nodes for all traces
    for trace in traces:
        tid = trace["id"]
        graph.nodes[tid] = LineageNode(
            trace_id=tid,
            timestamp=trace["timestamp"],
            input_hash=compute_content_hash(trace["input_text"]),
            output_hash=compute_content_hash(trace["output_text"]),
        )

    # Sort traces by timestamp for efficient comparison
    sorted_traces = sorted(traces, key=lambda t: t["timestamp"])

    # Find edges: source output_text matches target input_text
    for i, source in enumerate(sorted_traces):
        for j in range(i + 1, len(sorted_traces)):
            target = sorted_traces[j]
            if source["timestamp"] >= target["timestamp"]:
                continue

            source_output = source["output_text"]
            target_input = target["input_text"]

            if not source_output or not target_input:
                continue

            # Check for content overlap
            if source_output == target_input:
                confidence = 1.0
            elif source_output in target_input or target_input in source_output:
                shorter = min(len(source_output), len(target_input))
                longer = max(len(source_output), len(target_input))
                confidence = shorter / longer if longer > 0 else 0.0
            else:
                continue

            edge = LineageEdge(
                source_trace_id=source["id"],
                target_trace_id=target["id"],
                overlap_confidence=confidence,
            )
            graph.edges.append(edge)
            graph.nodes[source["id"]].downstream.append(target["id"])
            graph.nodes[target["id"]].upstream.append(source["id"])

    return graph


def find_lineage_chain(graph: LineageGraph, trace_id: str) -> list[str]:
    """Follow upstream chain to root. Return list from root to given trace."""
    if trace_id not in graph.nodes:
        return []

    chain: list[str] = [trace_id]
    visited: set = {trace_id}
    current = trace_id

    while True:
        node = graph.nodes.get(current)
        if node is None or not node.upstream:
            break
        # Follow first upstream link (primary lineage)
        parent = node.upstream[0]
        if parent in visited:
            break
        chain.append(parent)
        visited.add(parent)
        current = parent

    chain.reverse()
    return chain


def render_lineage_mermaid(graph: LineageGraph) -> str:
    """Render the lineage graph as a Mermaid diagram."""
    lines: list[str] = ["graph TD"]
    for tid, node in graph.nodes.items():
        label = tid[:8]
        lines.append(f"    {tid}[\"{label}\"]")
    for edge in graph.edges:
        conf = f"{edge.overlap_confidence:.0%}"
        lines.append(f"    {edge.source_trace_id} -->|{conf}| {edge.target_trace_id}")
    return "\n".join(lines)
