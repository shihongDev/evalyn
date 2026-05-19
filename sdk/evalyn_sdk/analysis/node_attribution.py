"""
Node-level metric attribution.

Attributes eval failures to specific graph nodes, identifying which nodes
contribute most to overall failure and surfacing hotspots for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class NodeScore:
    """Score for a single node on a single metric."""

    node_id: str
    node_name: str
    metric_id: str
    score: float
    passed: bool
    contribution: float = 0.0  # how much this node contributed to overall failure

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "metric_id": self.metric_id,
            "score": self.score,
            "passed": self.passed,
            "contribution": self.contribution,
        }


@dataclass
class AttributionResult:
    """Attribution result for a single dataset item."""

    item_id: str
    overall_score: float
    node_scores: list[NodeScore] = field(default_factory=list)
    primary_failure_node: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "overall_score": self.overall_score,
            "node_scores": [ns.as_dict() for ns in self.node_scores],
            "primary_failure_node": self.primary_failure_node,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttributionResult:
        node_scores = [
            NodeScore(
                node_id=ns["node_id"],
                node_name=ns["node_name"],
                metric_id=ns["metric_id"],
                score=ns["score"],
                passed=ns["passed"],
                contribution=ns.get("contribution", 0.0),
            )
            for ns in data.get("node_scores", [])
        ]
        return cls(
            item_id=data["item_id"],
            overall_score=data["overall_score"],
            node_scores=node_scores,
            primary_failure_node=data.get("primary_failure_node"),
        )


@dataclass
class AttributionReport:
    """Aggregated attribution report across items."""

    results: list[AttributionResult] = field(default_factory=list)
    failure_hotspots: dict[str, int] = field(default_factory=dict)  # node_name -> failure count
    total_items: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "failure_hotspots": dict(self.failure_hotspots),
            "total_items": self.total_items,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttributionReport:
        results = [AttributionResult.from_dict(r) for r in data.get("results", [])]
        return cls(
            results=results,
            failure_hotspots=data.get("failure_hotspots", {}),
            total_items=data.get("total_items", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("NODE ATTRIBUTION REPORT")
        lines.append(f"Total items: {self.total_items}")
        lines.append("")

        if self.failure_hotspots:
            lines.append("FAILURE HOTSPOTS:")
            for node_name, count in sorted(
                self.failure_hotspots.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {node_name}: {count} failures")
            lines.append("")

        for result in self.results:
            lines.append(f"Item: {result.item_id} (score: {result.overall_score:.2f})")
            if result.primary_failure_node:
                lines.append(f"  Primary failure: {result.primary_failure_node}")
            for ns in result.node_scores:
                status = "PASS" if ns.passed else "FAIL"
                lines.append(
                    f"  {ns.node_name} [{ns.metric_id}]: {ns.score:.2f} "
                    f"({status}, contribution: {ns.contribution:.2f})"
                )
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def attribute_to_nodes(
    item_id: str,
    overall_score: float,
    node_results: list[dict[str, Any]],
) -> AttributionResult:
    """Attribute an item's overall score to individual graph nodes.

    Each node_result dict must have: node_id, node_name, metric_id, score, passed.
    Contribution is computed as (1 - node_score) / sum(1 - all_scores).
    Primary failure node is the node with the lowest score.

    Args:
        item_id: The dataset item identifier.
        overall_score: The item's overall evaluation score.
        node_results: Per-node metric results.

    Returns:
        AttributionResult with per-node scores and contribution weights.
    """
    if not node_results:
        return AttributionResult(item_id=item_id, overall_score=overall_score)

    deficits = [1.0 - nr["score"] for nr in node_results]
    total_deficit = sum(deficits)

    node_scores: list[NodeScore] = []
    lowest_score = float("inf")
    primary_failure: str | None = None

    for nr, deficit in zip(node_results, deficits):
        contribution = deficit / total_deficit if total_deficit > 0 else 0.0
        ns = NodeScore(
            node_id=nr["node_id"],
            node_name=nr["node_name"],
            metric_id=nr["metric_id"],
            score=nr["score"],
            passed=nr["passed"],
            contribution=contribution,
        )
        node_scores.append(ns)

        if nr["score"] < lowest_score:
            lowest_score = nr["score"]
            primary_failure = nr["node_name"]

    return AttributionResult(
        item_id=item_id,
        overall_score=overall_score,
        node_scores=node_scores,
        primary_failure_node=primary_failure,
    )


def build_attribution_report(results: list[AttributionResult]) -> AttributionReport:
    """Aggregate attribution results into a report with failure hotspots.

    Counts how many times each node appears as a failure (passed=False)
    across all items.

    Args:
        results: List of per-item attribution results.

    Returns:
        AttributionReport with hotspot counts.
    """
    hotspots: dict[str, int] = {}
    for result in results:
        for ns in result.node_scores:
            if not ns.passed:
                hotspots[ns.node_name] = hotspots.get(ns.node_name, 0) + 1

    return AttributionReport(
        results=results,
        failure_hotspots=hotspots,
        total_items=len(results),
    )


def find_failure_hotspots(
    report: AttributionReport,
    min_failures: int = 2,
) -> list[tuple[str, int]]:
    """Return nodes with >= min_failures occurrences, sorted descending.

    Args:
        report: The attribution report to filter.
        min_failures: Minimum failure count to include a node.

    Returns:
        List of (node_name, failure_count) tuples sorted by count descending.
    """
    return sorted(
        [(name, count) for name, count in report.failure_hotspots.items() if count >= min_failures],
        key=lambda x: x[1],
        reverse=True,
    )


def suggest_fixes(report: AttributionReport) -> list[str]:
    """Generate fix suggestions based on failure hotspots.

    Args:
        report: The attribution report to analyze.

    Returns:
        List of human-readable suggestion strings.
    """
    if not report.failure_hotspots:
        return []

    suggestions: list[str] = []
    sorted_hotspots = sorted(
        report.failure_hotspots.items(), key=lambda x: x[1], reverse=True
    )

    for node_name, count in sorted_hotspots:
        suggestions.append(
            f"Node '{node_name}' fails most ({count} failures) - review prompt/logic"
        )

    return suggestions
