"""
SAMMO-style structural prompt optimization.

Treats prompts as symbolic DAGs with structural mutations.
Pure Python, no external dependencies, no LLM calls - this is the
structural framework for prompt manipulation and optimization.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PromptSection:
    """A single section of a structured prompt."""

    section_id: str
    section_type: str  # "instruction", "context", "examples", "rubric", "constraint"
    content: str
    weight: float = 1.0
    optional: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "section_type": self.section_type,
            "content": self.content,
            "weight": self.weight,
            "optional": self.optional,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PromptSection:
        return cls(
            section_id=data["section_id"],
            section_type=data["section_type"],
            content=data["content"],
            weight=data.get("weight", 1.0),
            optional=data.get("optional", False),
        )


@dataclass
class PromptDAG:
    """Directed acyclic graph of prompt sections with dependency edges."""

    sections: List[PromptSection] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sections": [s.as_dict() for s in self.sections],
            "edges": [list(e) for e in self.edges],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PromptDAG:
        return cls(
            sections=[PromptSection.from_dict(s) for s in data.get("sections", [])],
            edges=[tuple(e) for e in data.get("edges", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class MutationOp:
    """A mutation operation to apply to a PromptDAG."""

    op_type: str  # "paraphrase", "drop", "reformat", "reorder", "merge", "split"
    target_section: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "target_section": self.target_section,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MutationOp:
        return cls(
            op_type=data["op_type"],
            target_section=data["target_section"],
            parameters=data.get("parameters", {}),
        )


@dataclass
class OptimizationResult:
    """Result of applying a mutation to a PromptDAG."""

    original: PromptDAG
    mutated: PromptDAG
    mutation: MutationOp
    score_delta: float = 0.0
    cost_delta: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original.as_dict(),
            "mutated": self.mutated.as_dict(),
            "mutation": self.mutation.as_dict(),
            "score_delta": self.score_delta,
            "cost_delta": self.cost_delta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OptimizationResult:
        return cls(
            original=PromptDAG.from_dict(data["original"]),
            mutated=PromptDAG.from_dict(data["mutated"]),
            mutation=MutationOp.from_dict(data["mutation"]),
            score_delta=data.get("score_delta", 0.0),
            cost_delta=data.get("cost_delta", 0.0),
        )


# ---------------------------------------------------------------------------
# Graph Utilities
# ---------------------------------------------------------------------------


def detect_cycles(edges: List[Tuple[str, str]]) -> bool:
    """Return True if the edge list contains a cycle."""
    # Build adjacency list
    graph: Dict[str, List[str]] = {}
    nodes: set[str] = set()
    for src, dst in edges:
        graph.setdefault(src, []).append(dst)
        nodes.add(src)
        nodes.add(dst)

    # Kahn's algorithm: if we can't consume all nodes, there's a cycle
    in_degree: Dict[str, int] = {n: 0 for n in nodes}
    for src, dst in edges:
        in_degree[dst] += 1

    queue = deque(n for n in nodes if in_degree[n] == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return visited < len(nodes)


def topological_sort(dag: PromptDAG) -> List[str]:
    """Return section_ids in dependency order (Kahn's algorithm).

    Sections with no edges appear in their original list order.
    """
    section_ids = [s.section_id for s in dag.sections]
    id_set = set(section_ids)

    # Build adjacency and in-degree from edges
    graph: Dict[str, List[str]] = {sid: [] for sid in section_ids}
    in_degree: Dict[str, int] = {sid: 0 for sid in section_ids}
    for src, dst in dag.edges:
        if src in id_set and dst in id_set:
            graph[src].append(dst)
            in_degree[dst] += 1

    # Use index-based tie-breaking for stable ordering
    index_map = {sid: i for i, sid in enumerate(section_ids)}
    queue: List[str] = sorted(
        [sid for sid in section_ids if in_degree[sid] == 0],
        key=lambda s: index_map[s],
    )

    result: List[str] = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                # Insert in sorted position by original index
                inserted = False
                for i, q_node in enumerate(queue):
                    if index_map[neighbor] < index_map[q_node]:
                        queue.insert(i, neighbor)
                        inserted = True
                        break
                if not inserted:
                    queue.append(neighbor)

    return result


# ---------------------------------------------------------------------------
# DAG Construction
# ---------------------------------------------------------------------------


def build_prompt_dag(
    sections: List[PromptSection],
    edges: List[Tuple[str, str]] | None = None,
) -> PromptDAG:
    """Construct a PromptDAG with validation.

    Raises ValueError if edges reference unknown sections or create cycles.
    """
    edges = edges or []
    section_ids = {s.section_id for s in sections}

    # Validate edge references
    for src, dst in edges:
        if src not in section_ids:
            raise ValueError(f"edge source '{src}' not found in sections")
        if dst not in section_ids:
            raise ValueError(f"edge target '{dst}' not found in sections")

    # Validate no cycles
    if detect_cycles(edges):
        raise ValueError("edges contain a cycle")

    return PromptDAG(
        sections=list(sections),
        edges=list(edges),
        metadata={},
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_prompt(dag: PromptDAG) -> str:
    """Render sections in dependency order into a single prompt string."""
    order = topological_sort(dag)
    section_map = {s.section_id: s for s in dag.sections}
    parts: List[str] = []
    for sid in order:
        section = section_map[sid]
        parts.append(section.content)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


def apply_drop_mutation(dag: PromptDAG, section_id: str) -> PromptDAG:
    """Remove an optional section from the DAG.

    Raises ValueError if the section is not optional or does not exist.
    """
    section_map = {s.section_id: s for s in dag.sections}
    if section_id not in section_map:
        raise ValueError(f"section '{section_id}' not found")
    if not section_map[section_id].optional:
        raise ValueError(f"section '{section_id}' is not optional")

    new_sections = [s for s in dag.sections if s.section_id != section_id]
    new_edges = [
        (src, dst)
        for src, dst in dag.edges
        if src != section_id and dst != section_id
    ]
    return PromptDAG(
        sections=new_sections,
        edges=new_edges,
        metadata=copy.deepcopy(dag.metadata),
    )


def apply_reorder_mutation(
    dag: PromptDAG, new_order: List[str]
) -> PromptDAG:
    """Reorder sections according to new_order.

    Raises ValueError if new_order does not match the existing section ids.
    """
    existing_ids = {s.section_id for s in dag.sections}
    given_ids = set(new_order)
    if existing_ids != given_ids:
        raise ValueError("new_order must contain exactly the existing section ids")

    section_map = {s.section_id: s for s in dag.sections}
    new_sections = [section_map[sid] for sid in new_order]
    return PromptDAG(
        sections=new_sections,
        edges=list(dag.edges),
        metadata=copy.deepcopy(dag.metadata),
    )


def apply_merge_mutation(
    dag: PromptDAG, section_ids: List[str], new_id: str
) -> PromptDAG:
    """Merge multiple sections into one combined section.

    The merged section inherits the section_type of the first section.
    Content is concatenated with double newlines. Edges referencing any
    of the merged sections are rewritten to reference new_id.
    """
    section_map = {s.section_id: s for s in dag.sections}
    for sid in section_ids:
        if sid not in section_map:
            raise ValueError(f"section '{sid}' not found")

    merge_set = set(section_ids)
    merged_content = "\n\n".join(
        section_map[sid].content for sid in section_ids
    )
    first = section_map[section_ids[0]]
    merged_section = PromptSection(
        section_id=new_id,
        section_type=first.section_type,
        content=merged_content,
        weight=sum(section_map[sid].weight for sid in section_ids) / len(section_ids),
        optional=all(section_map[sid].optional for sid in section_ids),
    )

    # Build new sections list, replacing first occurrence with merged
    new_sections: List[PromptSection] = []
    inserted = False
    for s in dag.sections:
        if s.section_id in merge_set:
            if not inserted:
                new_sections.append(merged_section)
                inserted = True
            # skip subsequent merged sections
        else:
            new_sections.append(s)

    # Rewrite edges
    new_edges: List[Tuple[str, str]] = []
    seen_edges: set[Tuple[str, str]] = set()
    for src, dst in dag.edges:
        new_src = new_id if src in merge_set else src
        new_dst = new_id if dst in merge_set else dst
        if new_src == new_dst:
            continue  # skip self-loops
        edge = (new_src, new_dst)
        if edge not in seen_edges:
            new_edges.append(edge)
            seen_edges.add(edge)

    return PromptDAG(
        sections=new_sections,
        edges=new_edges,
        metadata=copy.deepcopy(dag.metadata),
    )


def apply_split_mutation(
    dag: PromptDAG, section_id: str, split_point: int
) -> PromptDAG:
    """Split a section at a character position into two sections.

    Creates section_id + "_a" and section_id + "_b".
    Edges are updated: incoming edges go to _a, outgoing go from _b,
    and an edge _a -> _b is added.
    """
    section_map = {s.section_id: s for s in dag.sections}
    if section_id not in section_map:
        raise ValueError(f"section '{section_id}' not found")

    original = section_map[section_id]
    if split_point <= 0 or split_point >= len(original.content):
        raise ValueError(
            f"split_point must be between 1 and {len(original.content) - 1}"
        )

    id_a = section_id + "_a"
    id_b = section_id + "_b"

    part_a = PromptSection(
        section_id=id_a,
        section_type=original.section_type,
        content=original.content[:split_point],
        weight=original.weight,
        optional=original.optional,
    )
    part_b = PromptSection(
        section_id=id_b,
        section_type=original.section_type,
        content=original.content[split_point:],
        weight=original.weight,
        optional=original.optional,
    )

    new_sections: List[PromptSection] = []
    for s in dag.sections:
        if s.section_id == section_id:
            new_sections.append(part_a)
            new_sections.append(part_b)
        else:
            new_sections.append(s)

    # Rewrite edges
    new_edges: List[Tuple[str, str]] = [(id_a, id_b)]
    for src, dst in dag.edges:
        if src == section_id and dst == section_id:
            continue  # skip self-loops on original
        new_src = id_b if src == section_id else src
        new_dst = id_a if dst == section_id else dst
        new_edges.append((new_src, new_dst))

    return PromptDAG(
        sections=new_sections,
        edges=new_edges,
        metadata=copy.deepcopy(dag.metadata),
    )


# ---------------------------------------------------------------------------
# Cost Estimation
# ---------------------------------------------------------------------------


def estimate_cost(dag: PromptDAG, cost_per_char: float = 0.00001) -> float:
    """Estimate prompt cost based on total character count and weight."""
    total = 0.0
    for section in dag.sections:
        total += len(section.content) * section.weight * cost_per_char
    return total


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_dag_summary(dag: PromptDAG) -> str:
    """Return a human-readable summary of the DAG."""
    lines: List[str] = []
    lines.append(f"PromptDAG: {len(dag.sections)} sections, {len(dag.edges)} edges")
    for s in dag.sections:
        opt = " (optional)" if s.optional else ""
        lines.append(
            f"  [{s.section_id}] {s.section_type} - "
            f"{len(s.content)} chars, weight={s.weight}{opt}"
        )
    if dag.edges:
        lines.append("  Edges:")
        for src, dst in dag.edges:
            lines.append(f"    {src} -> {dst}")
    cost = estimate_cost(dag)
    lines.append(f"  Estimated cost: {cost:.6f}")
    return "\n".join(lines)
