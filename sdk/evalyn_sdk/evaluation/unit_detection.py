"""Unit type auto-detection: infer best EvalUnit type from trace structure.

Analyzes span patterns in a trace to recommend the most appropriate
evaluation unit granularity (outcome, per_call, per_turn, per_tool, per_node).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..models import Span

UNIT_TYPES = ["outcome", "per_call", "per_turn", "per_tool", "per_node"]


@dataclass
class TracePattern:
    """Structural features extracted from a trace's spans."""

    has_multi_turn: bool
    has_tool_calls: bool
    has_retrieval: bool
    has_graph_nodes: bool
    llm_call_count: int
    tool_call_count: int
    max_depth: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "has_multi_turn": self.has_multi_turn,
            "has_tool_calls": self.has_tool_calls,
            "has_retrieval": self.has_retrieval,
            "has_graph_nodes": self.has_graph_nodes,
            "llm_call_count": self.llm_call_count,
            "tool_call_count": self.tool_call_count,
            "max_depth": self.max_depth,
        }


@dataclass
class DetectionResult:
    """Recommended unit type with confidence and reasoning."""

    recommended_unit_type: str
    confidence: float  # 0-1
    pattern: TracePattern
    reasoning: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "recommended_unit_type": self.recommended_unit_type,
            "confidence": self.confidence,
            "pattern": self.pattern.as_dict(),
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DetectionResult:
        return cls(
            recommended_unit_type=data["recommended_unit_type"],
            confidence=data["confidence"],
            pattern=TracePattern(**data["pattern"]),
            reasoning=data["reasoning"],
        )


def _compute_depth(span: Span, parent_map: Dict[str, str]) -> int:
    """Compute the depth of a span in the hierarchy."""
    depth = 0
    current = span.parent_id
    while current and current in parent_map:
        depth += 1
        current = parent_map[current]
    if span.parent_id is not None:
        depth += 1
    return depth


def analyze_trace_pattern(spans: List[Span]) -> TracePattern:
    """Analyze span types and structure to extract a TracePattern.

    has_multi_turn: multiple llm_call spans at the same depth.
    has_tool_calls: any span with span_type "tool_call".
    has_retrieval: any span with span_type "retrieval".
    has_graph_nodes: any span with span_type "node".
    """
    if not spans:
        return TracePattern(
            has_multi_turn=False,
            has_tool_calls=False,
            has_retrieval=False,
            has_graph_nodes=False,
            llm_call_count=0,
            tool_call_count=0,
            max_depth=0,
        )

    # Build parent_id -> parent_id mapping for depth computation
    parent_map: Dict[str, str] = {}
    for s in spans:
        if s.parent_id is not None:
            parent_map[s.id] = s.parent_id

    llm_call_count = 0
    tool_call_count = 0
    has_tool_calls = False
    has_retrieval = False
    has_graph_nodes = False

    llm_depths: List[int] = []
    max_depth = 0

    for s in spans:
        depth = _compute_depth(s, parent_map)
        if depth > max_depth:
            max_depth = depth

        if s.span_type == "llm_call":
            llm_call_count += 1
            llm_depths.append(depth)
        elif s.span_type == "tool_call":
            tool_call_count += 1
            has_tool_calls = True
        elif s.span_type == "retrieval":
            has_retrieval = True
        elif s.span_type == "node":
            has_graph_nodes = True

    # has_multi_turn: multiple llm_calls at the same depth
    has_multi_turn = False
    if llm_depths:
        from collections import Counter
        depth_counts = Counter(llm_depths)
        has_multi_turn = any(count >= 2 for count in depth_counts.values())

    return TracePattern(
        has_multi_turn=has_multi_turn,
        has_tool_calls=has_tool_calls,
        has_retrieval=has_retrieval,
        has_graph_nodes=has_graph_nodes,
        llm_call_count=llm_call_count,
        tool_call_count=tool_call_count,
        max_depth=max_depth,
    )


def detect_unit_type(spans: List[Span]) -> DetectionResult:
    """Detect the recommended evaluation unit type based on trace structure.

    Priority:
    1. If has_multi_turn and llm_call_count >= 3: "per_turn" (0.85 confidence)
    2. If has_tool_calls: "per_tool" (0.80 confidence)
    3. If has_graph_nodes: "per_node" (0.75 confidence)
    4. If llm_call_count == 1: "per_call" (0.90 confidence)
    5. Default: "outcome" (0.70 confidence)
    """
    pattern = analyze_trace_pattern(spans)

    if pattern.has_multi_turn and pattern.llm_call_count >= 3:
        return DetectionResult(
            recommended_unit_type="per_turn",
            confidence=0.85,
            pattern=pattern,
            reasoning="Multiple LLM calls at same depth suggest multi-turn conversation",
        )

    if pattern.has_tool_calls:
        return DetectionResult(
            recommended_unit_type="per_tool",
            confidence=0.80,
            pattern=pattern,
            reasoning="Tool call spans detected - per-tool evaluation recommended",
        )

    if pattern.has_graph_nodes:
        return DetectionResult(
            recommended_unit_type="per_node",
            confidence=0.75,
            pattern=pattern,
            reasoning="Graph node spans detected - per-node evaluation recommended",
        )

    if pattern.llm_call_count == 1:
        return DetectionResult(
            recommended_unit_type="per_call",
            confidence=0.90,
            pattern=pattern,
            reasoning="Single LLM call - per-call evaluation recommended",
        )

    return DetectionResult(
        recommended_unit_type="outcome",
        confidence=0.70,
        pattern=pattern,
        reasoning="Default to outcome-level evaluation",
    )


def detect_unit_types_batch(traces: List[List[Span]]) -> Dict[str, int]:
    """Count how many traces recommend each unit type.

    Returns a dict mapping unit type names to counts.
    """
    counts: Dict[str, int] = {ut: 0 for ut in UNIT_TYPES}
    for spans in traces:
        result = detect_unit_type(spans)
        counts[result.recommended_unit_type] = counts.get(result.recommended_unit_type, 0) + 1
    return counts
