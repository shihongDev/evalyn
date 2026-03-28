"""Trace complexity scoring.

Computes a single numeric score summarizing trace complexity based on
span depth, breadth, total span count, and tool call count. Useful for
quick triage and filtering.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models import Span


@dataclass
class ComplexityScore:
    """Complexity analysis for a set of spans (one trace)."""

    total_spans: int = 0
    max_depth: int = 0
    max_breadth: int = 0  # max children of any single span
    tool_call_count: int = 0
    llm_call_count: int = 0
    unique_span_types: int = 0
    score: float = 0.0  # weighted composite 0-100

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_spans": self.total_spans,
            "max_depth": self.max_depth,
            "max_breadth": self.max_breadth,
            "tool_call_count": self.tool_call_count,
            "llm_call_count": self.llm_call_count,
            "unique_span_types": self.unique_span_types,
            "score": round(self.score, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplexityScore":
        return cls(
            total_spans=data.get("total_spans", 0),
            max_depth=data.get("max_depth", 0),
            max_breadth=data.get("max_breadth", 0),
            tool_call_count=data.get("tool_call_count", 0),
            llm_call_count=data.get("llm_call_count", 0),
            unique_span_types=data.get("unique_span_types", 0),
            score=data.get("score", 0.0),
        )

    def format_text(self) -> str:
        return (
            f"Complexity: {self.score:.1f}/100 "
            f"(spans={self.total_spans}, depth={self.max_depth}, "
            f"breadth={self.max_breadth}, tools={self.tool_call_count}, "
            f"llm={self.llm_call_count})"
        )


@dataclass
class ComplexityThreshold:
    """Alert when trace exceeds expected complexity."""

    max_score: float = 70.0
    max_depth: int = 10
    max_spans: int = 50


def compute_complexity(
    spans: List[Span],
    weights: Optional[Dict[str, float]] = None,
) -> ComplexityScore:
    """Compute complexity score for a list of spans.

    Args:
        spans: Spans belonging to a single trace.
        weights: Optional custom weights for score components.
            Keys: "depth", "breadth", "count", "tools", "types".
            Default: depth=30, breadth=20, count=25, tools=15, types=10.

    Returns:
        ComplexityScore with weighted composite score.
    """
    if not spans:
        return ComplexityScore()

    w = weights or {
        "depth": 30.0,
        "breadth": 20.0,
        "count": 25.0,
        "tools": 15.0,
        "types": 10.0,
    }

    # Build parent -> children mapping
    children: Dict[Optional[str], List[str]] = defaultdict(list)
    span_map: Dict[str, Span] = {}
    for s in spans:
        span_map[s.id] = s
        children[s.parent_id].append(s.id)

    # Compute max depth via DFS
    roots = [s.id for s in spans if s.parent_id is None or s.parent_id not in span_map]
    max_depth = _max_depth(roots, children)

    # Max breadth (most children of any single parent)
    max_breadth = max((len(ch) for ch in children.values()), default=0)

    # Count span types
    span_types = {s.span_type for s in spans}
    tool_count = sum(1 for s in spans if s.span_type == "tool_call")
    llm_count = sum(1 for s in spans if s.span_type == "llm_call")

    total = len(spans)

    # Normalize each component to 0-1 scale using tanh-like saturation
    depth_norm = min(max_depth / 10.0, 1.0)
    breadth_norm = min(max_breadth / 10.0, 1.0)
    count_norm = min(total / 50.0, 1.0)
    tools_norm = min(tool_count / 20.0, 1.0)
    types_norm = min(len(span_types) / 8.0, 1.0)

    # Weighted sum, scaled to 0-100
    total_weight = sum(w.values())
    if total_weight == 0:
        score = 0.0
    else:
        raw = (
            w.get("depth", 0) * depth_norm
            + w.get("breadth", 0) * breadth_norm
            + w.get("count", 0) * count_norm
            + w.get("tools", 0) * tools_norm
            + w.get("types", 0) * types_norm
        )
        score = (raw / total_weight) * 100.0

    return ComplexityScore(
        total_spans=total,
        max_depth=max_depth,
        max_breadth=max_breadth,
        tool_call_count=tool_count,
        llm_call_count=llm_count,
        unique_span_types=len(span_types),
        score=score,
    )


def check_complexity_alerts(
    score: ComplexityScore,
    threshold: Optional[ComplexityThreshold] = None,
) -> List[str]:
    """Check if complexity exceeds thresholds.

    Returns list of alert messages (empty if within bounds).
    """
    t = threshold or ComplexityThreshold()
    alerts = []
    if score.score > t.max_score:
        alerts.append(
            f"Complexity score {score.score:.1f} exceeds threshold {t.max_score}"
        )
    if score.max_depth > t.max_depth:
        alerts.append(
            f"Span depth {score.max_depth} exceeds threshold {t.max_depth}"
        )
    if score.total_spans > t.max_spans:
        alerts.append(
            f"Span count {score.total_spans} exceeds threshold {t.max_spans}"
        )
    return alerts


def _max_depth(roots: List[str], children: Dict[Optional[str], List[str]]) -> int:
    """Compute max depth of the span tree via iterative DFS."""
    if not roots:
        return 0
    max_d = 0
    stack = [(r, 1) for r in roots]
    while stack:
        node, depth = stack.pop()
        max_d = max(max_d, depth)
        for child in children.get(node, []):
            stack.append((child, depth + 1))
    return max_d
