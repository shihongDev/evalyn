"""Knowledge graph test generation from document text.

Build a simple knowledge graph from text via entity/relation extraction,
then generate evaluation questions from the graph structure. Pure Python,
no external dependencies, no LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class KGNode:
    """A node in the knowledge graph."""

    node_id: str
    label: str
    node_type: str  # "entity", "concept", "fact"
    properties: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type,
            "properties": dict(self.properties),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KGNode:
        return cls(
            node_id=data["node_id"],
            label=data["label"],
            node_type=data["node_type"],
            properties=data.get("properties", {}),
        )


@dataclass
class KGEdge:
    """A directed edge in the knowledge graph."""

    source: str
    target: str
    relation: str
    weight: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KGEdge:
        return cls(
            source=data["source"],
            target=data["target"],
            relation=data["relation"],
            weight=data.get("weight", 1.0),
        )


@dataclass
class KnowledgeGraph:
    """A collection of nodes and edges forming a knowledge graph."""

    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.as_dict() for n in self.nodes],
            "edges": [e.as_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        return cls(
            nodes=[KGNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[KGEdge.from_dict(e) for e in data.get("edges", [])],
        )


@dataclass
class GeneratedQuestion:
    """A question generated from the knowledge graph."""

    question: str
    expected_answer: str
    source_nodes: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    question_type: str = "factual"

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "source_nodes": list(self.source_nodes),
            "difficulty": self.difficulty,
            "question_type": self.question_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneratedQuestion:
        return cls(
            question=data["question"],
            expected_answer=data["expected_answer"],
            source_nodes=data.get("source_nodes", []),
            difficulty=data.get("difficulty", "medium"),
            question_type=data.get("question_type", "factual"),
        )


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

_CAPITALIZED_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_LEADING_ARTICLE_RE = re.compile(r"^(?:The|A|An)\s+", re.IGNORECASE)
_QUOTED_TERM_RE = re.compile(r'"([^"]+)"')
_IS_A_RE = re.compile(r"\b(?:is\s+a|is\s+an|are)\s+([A-Za-z][A-Za-z\s]*[A-Za-z])", re.IGNORECASE)


def _make_node_id(label: str) -> str:
    """Deterministic node id from label."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def extract_entities(text: str) -> list[KGNode]:
    """Extract entities from text using simple heuristics.

    Finds capitalized multi-word phrases, quoted terms, and terms
    following "is a"/"is an"/"are" patterns.
    """
    seen: dict[str, KGNode] = {}

    # Capitalized multi-word phrases -> entity
    for match in _CAPITALIZED_PHRASE_RE.finditer(text):
        label = match.group(1).strip()
        # Strip leading articles like "The", "A", "An"
        label = _LEADING_ARTICLE_RE.sub("", label).strip()
        if not label or " " not in label:
            continue
        nid = _make_node_id(label)
        if nid not in seen:
            seen[nid] = KGNode(node_id=nid, label=label, node_type="entity")

    # Quoted terms -> concept
    for match in _QUOTED_TERM_RE.finditer(text):
        label = match.group(1).strip()
        if not label:
            continue
        nid = _make_node_id(label)
        if nid not in seen:
            seen[nid] = KGNode(node_id=nid, label=label, node_type="concept")

    # "is a/are" patterns -> fact
    for match in _IS_A_RE.finditer(text):
        label = match.group(1).strip()
        if not label:
            continue
        nid = _make_node_id(label)
        if nid not in seen:
            seen[nid] = KGNode(node_id=nid, label=label, node_type="fact")

    return list(seen.values())


# ---------------------------------------------------------------------------
# Relation extraction
# ---------------------------------------------------------------------------

_VERB_PHRASE_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+"
    r"((?:is|are|was|were|has|have|had|can|will|does|do|did|"
    r"uses|creates|builds|provides|contains|requires|supports|"
    r"enables|includes|produces|generates|manages|controls|"
    r"connects|links|runs|operates|depends\s+on|relates\s+to|"
    r"belongs\s+to|consists\s+of|leads\s+to|results\s+in|"
    r"works\s+with|interacts\s+with|derives\s+from)"
    r"(?:\s+\w+)*?)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)


def extract_relations(text: str, entities: list[KGNode]) -> list[KGEdge]:
    """Find verb phrases connecting entities and create edges.

    Scans text for patterns where a capitalized phrase is connected to
    another capitalized phrase via a verb, then checks whether both
    phrases match known entities.
    """
    entity_labels = {e.label.lower(): e.node_id for e in entities}
    edges: list[KGEdge] = []
    seen: set[tuple[str, str, str]] = set()

    for match in _VERB_PHRASE_RE.finditer(text):
        src_label = match.group(1).strip().lower()
        relation = match.group(2).strip().lower()
        tgt_label = match.group(3).strip().lower()

        src_id = entity_labels.get(src_label)
        tgt_id = entity_labels.get(tgt_label)

        if src_id and tgt_id and src_id != tgt_id:
            key = (src_id, tgt_id, relation)
            if key not in seen:
                seen.add(key)
                edges.append(KGEdge(source=src_id, target=tgt_id, relation=relation))

    return edges


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_knowledge_graph(texts: list[str]) -> KnowledgeGraph:
    """Build a knowledge graph from multiple text passages.

    Extracts entities and relations from each text and merges them
    into a single graph, deduplicating nodes by id.
    """
    all_nodes: dict[str, KGNode] = {}
    all_edges: list[KGEdge] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for text in texts:
        nodes = extract_entities(text)
        for node in nodes:
            if node.node_id not in all_nodes:
                all_nodes[node.node_id] = node
        edges = extract_relations(text, list(all_nodes.values()))
        for edge in edges:
            key = (edge.source, edge.target, edge.relation)
            if key not in seen_edges:
                seen_edges.add(key)
                all_edges.append(edge)

    return KnowledgeGraph(nodes=list(all_nodes.values()), edges=all_edges)


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------


def generate_factual_questions(graph: KnowledgeGraph, n: int = 5) -> list[GeneratedQuestion]:
    """Generate factual questions from graph nodes and edges.

    Produces "What is X?" questions from nodes and "What is the
    relationship between X and Y?" questions from edges.
    """
    questions: list[GeneratedQuestion] = []

    # Node-based questions
    for node in graph.nodes:
        if len(questions) >= n:
            break
        q = GeneratedQuestion(
            question=f"What is {node.label}?",
            expected_answer=f"{node.label} is a {node.node_type}.",
            source_nodes=[node.node_id],
            difficulty="easy",
            question_type="factual",
        )
        questions.append(q)

    # Edge-based questions
    node_map = {nd.node_id: nd for nd in graph.nodes}
    for edge in graph.edges:
        if len(questions) >= n:
            break
        src = node_map.get(edge.source)
        tgt = node_map.get(edge.target)
        if src and tgt:
            q = GeneratedQuestion(
                question=(f"What is the relationship between {src.label} and {tgt.label}?"),
                expected_answer=f"{src.label} {edge.relation} {tgt.label}.",
                source_nodes=[edge.source, edge.target],
                difficulty="medium",
                question_type="factual",
            )
            questions.append(q)

    return questions[:n]


def generate_reasoning_questions(graph: KnowledgeGraph, n: int = 3) -> list[GeneratedQuestion]:
    """Generate multi-hop reasoning questions from graph paths.

    Finds two-hop paths (A->B->C) and produces questions of the form
    "If A relates to B and B relates to C, what can be inferred?"
    """
    questions: list[GeneratedQuestion] = []
    node_map = {nd.node_id: nd for nd in graph.nodes}

    # Build adjacency: source -> list of (target, relation)
    adj: dict[str, list[tuple[str, str]]] = {}
    for edge in graph.edges:
        adj.setdefault(edge.source, []).append((edge.target, edge.relation))

    seen: set[tuple[str, str, str]] = set()

    for edge in graph.edges:
        if len(questions) >= n:
            break
        mid_id = edge.target
        neighbors = adj.get(mid_id, [])
        for end_id, rel2 in neighbors:
            if len(questions) >= n:
                break
            if end_id == edge.source:
                continue
            key = (edge.source, mid_id, end_id)
            if key in seen:
                continue
            seen.add(key)

            src = node_map.get(edge.source)
            mid = node_map.get(mid_id)
            end = node_map.get(end_id)
            if not (src and mid and end):
                continue

            q = GeneratedQuestion(
                question=(
                    f"If {src.label} {edge.relation} {mid.label} "
                    f"and {mid.label} {rel2} {end.label}, "
                    f"what can be inferred about {src.label} and {end.label}?"
                ),
                expected_answer=(f"{src.label} is connected to {end.label} through {mid.label}."),
                source_nodes=[edge.source, mid_id, end_id],
                difficulty="hard",
                question_type="reasoning",
            )
            questions.append(q)

    return questions[:n]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_question_set(questions: list[GeneratedQuestion]) -> str:
    """Format questions as a human-readable numbered Q&A list."""
    if not questions:
        return "No questions generated."

    lines: list[str] = []
    for i, q in enumerate(questions, 1):
        lines.append(f"Q{i}: {q.question}")
        lines.append(f"A{i}: {q.expected_answer}")
        lines.append(f"    [difficulty={q.difficulty}, type={q.question_type}]")
        lines.append("")

    return "\n".join(lines).rstrip()
