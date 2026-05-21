from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Span


@dataclass
class GraphNode:
    """A node in the execution graph."""

    node_id: str
    node_name: str
    node_type: str = "node"
    execution_count: int = 0
    avg_duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "node_type": self.node_type,
            "execution_count": self.execution_count,
            "avg_duration_ms": self.avg_duration_ms,
        }


@dataclass
class GraphEdge:
    """A directed edge between two nodes."""

    source: str
    target: str
    count: int = 1
    avg_latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "count": self.count,
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class GraphTopology:
    """Complete execution graph extracted from traces."""

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    entry_nodes: list[str] = field(default_factory=list)
    exit_nodes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": {nid: n.as_dict() for nid, n in self.nodes.items()},
            "edges": [e.as_dict() for e in self.edges],
            "entry_nodes": list(self.entry_nodes),
            "exit_nodes": list(self.exit_nodes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphTopology:
        nodes = {}
        for nid, ndata in data.get("nodes", {}).items():
            nodes[nid] = GraphNode(
                node_id=ndata["node_id"],
                node_name=ndata["node_name"],
                node_type=ndata.get("node_type", "node"),
                execution_count=ndata.get("execution_count", 0),
                avg_duration_ms=ndata.get("avg_duration_ms", 0.0),
            )
        edges = []
        for edata in data.get("edges", []):
            edges.append(
                GraphEdge(
                    source=edata["source"],
                    target=edata["target"],
                    count=edata.get("count", 1),
                    avg_latency_ms=edata.get("avg_latency_ms", 0.0),
                )
            )
        return cls(
            nodes=nodes,
            edges=edges,
            entry_nodes=data.get("entry_nodes", []),
            exit_nodes=data.get("exit_nodes", []),
        )

    def format_text(self) -> str:
        """Human-readable summary of the topology."""
        lines: list[str] = []
        lines.append(f"Nodes: {len(self.nodes)}")
        lines.append(f"Edges: {len(self.edges)}")
        if self.entry_nodes:
            lines.append(f"Entry: {', '.join(self.entry_nodes)}")
        if self.exit_nodes:
            lines.append(f"Exit: {', '.join(self.exit_nodes)}")
        lines.append("")
        for node in self.nodes.values():
            lines.append(
                f"  [{node.node_id}] {node.node_name} "
                f"(type={node.node_type}, count={node.execution_count}, "
                f"avg={node.avg_duration_ms:.1f}ms)"
            )
        for edge in self.edges:
            lines.append(
                f"  {edge.source} -> {edge.target} "
                f"(count={edge.count}, latency={edge.avg_latency_ms:.1f}ms)"
            )
        return "\n".join(lines)


def _node_key(span: Span) -> str:
    """Derive a stable node key from a span."""
    return f"{span.name}:{span.span_type}"


def extract_topology(spans: list[Span]) -> GraphTopology:
    """Build a graph topology from span parent-child relationships.

    Each unique (name, span_type) pair becomes a node.
    Parent-child links become directed edges.
    """
    if not spans:
        return GraphTopology()

    # Build node stats
    node_durations: dict[str, list[float]] = {}
    span_to_key: dict[str, str] = {}

    for span in spans:
        key = _node_key(span)
        span_to_key[span.id] = key
        node_durations.setdefault(key, [])
        dur = span.duration_ms
        if dur is not None:
            node_durations[key].append(dur)

    # Create nodes
    nodes: dict[str, GraphNode] = {}
    for key, durations in node_durations.items():
        name, stype = key.rsplit(":", 1)
        avg_dur = sum(durations) / len(durations) if durations else 0.0
        nodes[key] = GraphNode(
            node_id=key,
            node_name=name,
            node_type=stype,
            execution_count=len(durations) if durations else 1,
            avg_duration_ms=avg_dur,
        )

    # Build edges from parent-child
    edge_counts: dict[tuple, int] = {}
    edge_latencies: dict[tuple, list[float]] = {}

    span_by_id = {s.id: s for s in spans}
    for span in spans:
        if span.parent_id and span.parent_id in span_by_id:
            src = span_to_key[span.parent_id]
            tgt = span_to_key[span.id]
            if src == tgt:
                continue
            pair = (src, tgt)
            edge_counts[pair] = edge_counts.get(pair, 0) + 1
            parent = span_by_id[span.parent_id]
            if parent.end_time and span.start_time:
                latency = (span.start_time - parent.start_time).total_seconds() * 1000
                edge_latencies.setdefault(pair, []).append(latency)

    edges: list[GraphEdge] = []
    for (src, tgt), count in edge_counts.items():
        lats = edge_latencies.get((src, tgt), [])
        avg_lat = sum(lats) / len(lats) if lats else 0.0
        edges.append(GraphEdge(source=src, target=tgt, count=count, avg_latency_ms=avg_lat))

    topo = GraphTopology(nodes=nodes, edges=edges)
    topo.entry_nodes = find_entry_nodes(topo)
    topo.exit_nodes = find_exit_nodes(topo)
    return topo


def find_entry_nodes(topology: GraphTopology) -> list[str]:
    """Nodes with no incoming edges."""
    targets = {e.target for e in topology.edges}
    return [nid for nid in topology.nodes if nid not in targets]


def find_exit_nodes(topology: GraphTopology) -> list[str]:
    """Nodes with no outgoing edges."""
    sources = {e.source for e in topology.edges}
    return [nid for nid in topology.nodes if nid not in sources]


def render_topology_mermaid(topology: GraphTopology) -> str:
    """Render topology as a Mermaid graph diagram."""
    lines: list[str] = ["graph TD"]
    for node in topology.nodes.values():
        label = f"{node.node_name} ({node.execution_count}x)"
        lines.append(f'    {node.node_id.replace(":", "_")}["{label}"]')
    for edge in topology.edges:
        src = edge.source.replace(":", "_")
        tgt = edge.target.replace(":", "_")
        lines.append(f"    {src} --> {tgt}")
    return "\n".join(lines)


def compute_critical_path(topology: GraphTopology) -> list[str]:
    """Longest path by avg_duration_ms through the graph (DAG assumed for main path).

    Uses dynamic programming on a topological sort. Falls back to DFS
    when cycles are present.
    """
    if not topology.nodes:
        return []

    # Build adjacency
    adj: dict[str, list[str]] = {nid: [] for nid in topology.nodes}
    for edge in topology.edges:
        adj[edge.source].append(edge.target)

    # Topological sort via Kahn's algorithm
    in_degree: dict[str, int] = dict.fromkeys(topology.nodes, 0)
    for edge in topology.edges:
        in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

    queue = [nid for nid, d in in_degree.items() if d == 0]
    topo_order: list[str] = []
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If not all nodes processed, there is a cycle - use DFS fallback
    if len(topo_order) != len(topology.nodes):
        return _critical_path_dfs(topology, adj)

    # DP: dist[node] = longest path weight ending at node
    dist: dict[str, float] = {nid: topology.nodes[nid].avg_duration_ms for nid in topology.nodes}
    prev: dict[str, str | None] = dict.fromkeys(topology.nodes)

    for node in topo_order:
        for neighbor in adj[node]:
            new_dist = dist[node] + topology.nodes[neighbor].avg_duration_ms
            if new_dist > dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = node

    # Find the node with the largest distance
    end_node = max(dist, key=lambda n: dist[n])
    path: list[str] = []
    current: str | None = end_node
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()
    return path


def _critical_path_dfs(topology: GraphTopology, adj: dict[str, list[str]]) -> list[str]:
    """DFS fallback for critical path when graph has cycles."""
    best_path: list[str] = []
    best_weight = 0.0

    for start in topology.nodes:
        stack = [(start, [start], topology.nodes[start].avg_duration_ms, {start})]
        while stack:
            node, path, weight, visited = stack.pop()
            if weight > best_weight:
                best_weight = weight
                best_path = list(path)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    stack.append(
                        (
                            neighbor,
                            path + [neighbor],
                            weight + topology.nodes[neighbor].avg_duration_ms,
                            visited | {neighbor},
                        )
                    )

    return best_path


def detect_cycles(topology: GraphTopology) -> list[list[str]]:
    """Detect cycles in the graph using DFS.

    Returns a list of cycles, each cycle represented as a list of node IDs.
    """
    adj: dict[str, list[str]] = {nid: [] for nid in topology.nodes}
    for edge in topology.edges:
        adj[edge.source].append(edge.target)

    cycles: list[list[str]] = []
    visited: set = set()
    rec_stack: set = set()
    path: list[str] = []

    def dfs(node: str) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                # Found a cycle: extract from neighbor to current position
                idx = path.index(neighbor)
                cycle = path[idx:] + [neighbor]
                cycles.append(cycle)

        path.pop()
        rec_stack.discard(node)

    for node in topology.nodes:
        if node not in visited:
            dfs(node)

    return cycles
