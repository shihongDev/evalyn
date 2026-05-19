"""DAG-based deterministic scoring.

Decision-tree scoring as a middle ground between simple rules and LLM judges.
Each DAGMetric is a directed acyclic graph of condition nodes. Walking the graph
from the root produces a score based on which leaf is reached.

Safety: all condition checking uses safe parsing with regex only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DAGNode:
    """A single node in a decision DAG."""

    id: str
    condition: str  # e.g. 'len(output) > 10' or '"keyword" in output'
    true_branch: Optional[str] = None   # next node id when condition is True
    false_branch: Optional[str] = None  # next node id when condition is False
    score: Optional[float] = None       # leaf score (used when no branches)
    label: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "condition": self.condition,
            "true_branch": self.true_branch,
            "false_branch": self.false_branch,
            "score": self.score,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DAGNode:
        return cls(
            id=data["id"],
            condition=data.get("condition", ""),
            true_branch=data.get("true_branch"),
            false_branch=data.get("false_branch"),
            score=data.get("score"),
            label=data.get("label", ""),
        )


@dataclass
class DAGMetric:
    """A complete decision-tree metric."""

    name: str
    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    root_id: str = ""
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": {nid: node.as_dict() for nid, node in self.nodes.items()},
            "root_id": self.root_id,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DAGMetric:
        nodes = {
            nid: DAGNode.from_dict(ndata)
            for nid, ndata in data.get("nodes", {}).items()
        }
        return cls(
            name=data["name"],
            nodes=nodes,
            root_id=data.get("root_id", ""),
            description=data.get("description", ""),
        )


@dataclass
class DAGResult:
    """Result of walking a DAG for a single assessment."""

    metric_name: str
    score: float
    path: List[str]                          # node ids traversed
    decisions: List[Tuple[str, bool]]        # (condition, result) pairs

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "path": list(self.path),
            "decisions": [[cond, result] for cond, result in self.decisions],
        }


# ---------------------------------------------------------------------------
# Condition parser (safe - regex-based, no dynamic code running)
# ---------------------------------------------------------------------------

def _compare(actual: int, op: str, expected: int) -> bool:
    if op == "<":
        return actual < expected
    if op == "<=":
        return actual <= expected
    if op == ">":
        return actual > expected
    if op == ">=":
        return actual >= expected
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    return False


def _check_condition(condition: str, context: Dict[str, Any]) -> bool:
    """Parse and check a simple condition against context.

    Supported forms:
      - len(output) > N / len(input) < N  (any comparison operator)
      - output_length > N / input_length > N
      - "keyword" in output / "keyword" in input
      - "keyword" not in output / "keyword" not in input

    Returns False for unparseable conditions.
    """
    cond = condition.strip()

    output_text = str(context.get("output", ""))
    input_text = str(context.get("input", ""))
    output_length = context.get("output_length", len(output_text))
    input_length = context.get("input_length", len(input_text))

    # len(output) op N
    m = re.match(r'^len\(output\)\s*([<>=!]+)\s*(\d+)$', cond)
    if m:
        return _compare(len(output_text), m.group(1), int(m.group(2)))

    # len(input) op N
    m = re.match(r'^len\(input\)\s*([<>=!]+)\s*(\d+)$', cond)
    if m:
        return _compare(len(input_text), m.group(1), int(m.group(2)))

    # output_length op N
    m = re.match(r'^output_length\s*([<>=!]+)\s*(\d+)$', cond)
    if m:
        return _compare(int(output_length), m.group(1), int(m.group(2)))

    # input_length op N
    m = re.match(r'^input_length\s*([<>=!]+)\s*(\d+)$', cond)
    if m:
        return _compare(int(input_length), m.group(1), int(m.group(2)))

    # "keyword" not in output  (must check before "in")
    m = re.match(r'^"([^"]+)"\s+not\s+in\s+output$', cond)
    if m:
        return m.group(1).lower() not in output_text.lower()

    # "keyword" not in input
    m = re.match(r'^"([^"]+)"\s+not\s+in\s+input$', cond)
    if m:
        return m.group(1).lower() not in input_text.lower()

    # "keyword" in output
    m = re.match(r'^"([^"]+)"\s+in\s+output$', cond)
    if m:
        return m.group(1).lower() in output_text.lower()

    # "keyword" in input
    m = re.match(r'^"([^"]+)"\s+in\s+input$', cond)
    if m:
        return m.group(1).lower() in input_text.lower()

    # Unparseable - return False
    return False


# Public alias matching the spec name
_evaluate_condition = _check_condition


# ---------------------------------------------------------------------------
# DAG walker
# ---------------------------------------------------------------------------

def evaluate_dag(dag: DAGMetric, context: Dict[str, Any]) -> DAGResult:
    """Walk the DAG from root and produce a score.

    At each node the condition is checked against *context*. The walk
    follows true_branch or false_branch until a leaf (no branch / missing
    node) is reached.

    Context keys: "input", "output", "output_length", "input_length".
    """
    path: List[str] = []
    decisions: List[Tuple[str, bool]] = []
    score = 0.0

    current_id = dag.root_id
    visited: set = set()

    while current_id and current_id in dag.nodes and current_id not in visited:
        visited.add(current_id)
        node = dag.nodes[current_id]
        path.append(current_id)

        # Leaf node: no branches to follow
        if node.true_branch is None and node.false_branch is None:
            score = node.score if node.score is not None else 0.0
            break

        result = _check_condition(node.condition, context)
        decisions.append((node.condition, result))

        next_id = node.true_branch if result else node.false_branch

        # If the chosen branch is None or points to a missing node, use this node's score
        if next_id is None or next_id not in dag.nodes:
            score = node.score if node.score is not None else 0.0
            break

        current_id = next_id
    else:
        # Reached a missing node or empty root - use last visited node's score
        if path:
            last_node = dag.nodes[path[-1]]
            score = last_node.score if last_node.score is not None else 0.0

    return DAGResult(
        metric_name=dag.name,
        score=score,
        path=path,
        decisions=decisions,
    )


# ---------------------------------------------------------------------------
# Builder helper
# ---------------------------------------------------------------------------

def build_simple_dag(
    name: str,
    conditions: List[Tuple[str, float, float]],
) -> DAGMetric:
    """Build a linear chain DAG from a list of (condition, true_score, false_score).

    Each condition creates a decision node whose true branch leads to a leaf
    with *true_score* and whose false branch leads to a leaf with *false_score*
    (unless there is a subsequent condition, in which case the true branch
    chains to the next decision node).

    For N conditions:
      - N decision nodes (cond_0 .. cond_{N-1})
      - N false-leaf nodes (false_0 .. false_{N-1})
      - 1 true-leaf node (true_{N-1}) for the last condition

    The true branch of each decision node (except the last) chains to the
    next decision node, so the DAG checks conditions in sequence.
    """
    nodes: Dict[str, DAGNode] = {}

    if not conditions:
        # Edge case: no conditions, single leaf with score 0
        leaf = DAGNode(id="leaf", condition="", score=0.0)
        nodes["leaf"] = leaf
        return DAGMetric(name=name, nodes=nodes, root_id="leaf")

    n = len(conditions)

    for i, (cond, true_score, false_score) in enumerate(conditions):
        cond_id = f"cond_{i}"
        false_id = f"false_{i}"

        # False leaf for this condition
        nodes[false_id] = DAGNode(
            id=false_id,
            condition="",
            score=false_score,
            label=f"false leaf {i}",
        )

        if i == n - 1:
            # Last condition: true branch goes to a true leaf
            true_id = f"true_{i}"
            nodes[true_id] = DAGNode(
                id=true_id,
                condition="",
                score=true_score,
                label=f"true leaf {i}",
            )
            nodes[cond_id] = DAGNode(
                id=cond_id,
                condition=cond,
                true_branch=true_id,
                false_branch=false_id,
                label=f"check {i}",
            )
        else:
            # Chain true branch to next condition node
            next_cond_id = f"cond_{i + 1}"
            nodes[cond_id] = DAGNode(
                id=cond_id,
                condition=cond,
                true_branch=next_cond_id,
                false_branch=false_id,
                label=f"check {i}",
            )

    return DAGMetric(name=name, nodes=nodes, root_id="cond_0")
