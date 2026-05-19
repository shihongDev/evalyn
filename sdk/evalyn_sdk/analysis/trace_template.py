"""Trace template matching.

Matches traces against known execution patterns (e.g. RAG, retry loop,
fan-out). Supports built-in and custom pattern definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models import Span


@dataclass
class TracePattern:
    """A named span-type sequence pattern."""

    name: str
    description: str
    sequence: List[str]  # ordered span types to match
    allow_gaps: bool = True  # if True, extra spans between pattern elements are ok

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "sequence": list(self.sequence),
            "allow_gaps": self.allow_gaps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TracePattern:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            sequence=data["sequence"],
            allow_gaps=data.get("allow_gaps", True),
        )


@dataclass
class MatchResult:
    """Result of matching a trace against a pattern."""

    pattern_name: str
    matched: bool
    matched_span_ids: List[str] = field(default_factory=list)
    coverage: float = 0.0  # fraction of trace spans involved in the match

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "matched": self.matched,
            "matched_span_count": len(self.matched_span_ids),
            "coverage": round(self.coverage, 4),
        }


@dataclass
class ClassificationReport:
    """Classification of traces against known patterns."""

    trace_count: int = 0
    classified: int = 0
    unclassified: int = 0
    pattern_counts: Dict[str, int] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)  # per-trace results

    @property
    def coverage_rate(self) -> float:
        return self.classified / self.trace_count if self.trace_count > 0 else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trace_count": self.trace_count,
            "classified": self.classified,
            "unclassified": self.unclassified,
            "coverage_rate": round(self.coverage_rate, 4),
            "pattern_counts": dict(self.pattern_counts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClassificationReport:
        return cls(
            trace_count=data.get("trace_count", 0),
            classified=data.get("classified", 0),
            unclassified=data.get("unclassified", 0),
            pattern_counts=data.get("pattern_counts", {}),
        )

    def format_text(self) -> str:
        lines = [
            f"Trace Classification: {self.classified}/{self.trace_count} "
            f"classified ({self.coverage_rate:.0%})"
        ]
        for name, count in sorted(
            self.pattern_counts.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  {name}: {count}")
        if self.unclassified > 0:
            lines.append(f"  (unclassified): {self.unclassified}")
        return "\n".join(lines)


# ---- Built-in patterns ----

RAG_PATTERN = TracePattern(
    name="rag",
    description="Retrieve then generate: retrieval followed by LLM call",
    sequence=["retrieval", "llm_call"],
)

RETRY_LOOP = TracePattern(
    name="retry_loop",
    description="Repeated LLM calls (retry or multi-turn)",
    sequence=["llm_call", "llm_call"],
)

TOOL_USE = TracePattern(
    name="tool_use",
    description="LLM call followed by tool execution",
    sequence=["llm_call", "tool_call"],
)

FAN_OUT = TracePattern(
    name="fan_out",
    description="LLM call followed by multiple tool calls",
    sequence=["llm_call", "tool_call", "tool_call"],
)

AGENT_LOOP = TracePattern(
    name="agent_loop",
    description="LLM reason then act loop",
    sequence=["llm_call", "tool_call", "llm_call"],
)

BUILTIN_PATTERNS: Dict[str, TracePattern] = {
    "rag": RAG_PATTERN,
    "retry_loop": RETRY_LOOP,
    "tool_use": TOOL_USE,
    "fan_out": FAN_OUT,
    "agent_loop": AGENT_LOOP,
}


def match_trace(
    spans: List[Span],
    pattern: TracePattern,
) -> MatchResult:
    """Check if a trace matches a pattern.

    Args:
        spans: Spans of a single trace, in any order.
        pattern: Pattern to match against.

    Returns:
        MatchResult indicating match status and coverage.
    """
    ordered = sorted(spans, key=lambda s: s.start_time)
    types = [s.span_type for s in ordered]

    if pattern.allow_gaps:
        matched_indices = _subsequence_match(types, pattern.sequence)
    else:
        matched_indices = _contiguous_match(types, pattern.sequence)

    if matched_indices is not None:
        matched_ids = [ordered[i].id for i in matched_indices]
        coverage = len(matched_ids) / len(spans) if spans else 0.0
        return MatchResult(
            pattern_name=pattern.name,
            matched=True,
            matched_span_ids=matched_ids,
            coverage=coverage,
        )

    return MatchResult(pattern_name=pattern.name, matched=False)


def classify_traces(
    traces: List[List[Span]],
    patterns: Optional[List[TracePattern]] = None,
    custom_patterns: Optional[List[Dict[str, Any]]] = None,
) -> ClassificationReport:
    """Classify multiple traces against known patterns.

    Each trace is matched against all patterns; the first match wins.
    Patterns are checked in order, so put more specific patterns first.

    Args:
        traces: List of traces (each a list of spans).
        patterns: Patterns to match. Defaults to built-in patterns.
        custom_patterns: Additional patterns from config (list of dicts).

    Returns:
        ClassificationReport with pattern counts and coverage.
    """
    all_patterns = list(patterns or BUILTIN_PATTERNS.values())
    if custom_patterns:
        for cp in custom_patterns:
            all_patterns.append(TracePattern.from_dict(cp))

    report = ClassificationReport(trace_count=len(traces))
    pattern_counts: Dict[str, int] = {}

    for trace_spans in traces:
        matched_any = False
        trace_result = {"matched_patterns": []}

        for pat in all_patterns:
            result = match_trace(trace_spans, pat)
            if result.matched:
                trace_result["matched_patterns"].append(result.as_dict())
                if not matched_any:
                    pattern_counts[pat.name] = pattern_counts.get(pat.name, 0) + 1
                    matched_any = True

        if matched_any:
            report.classified += 1
        else:
            report.unclassified += 1

        report.results.append(trace_result)

    report.pattern_counts = pattern_counts
    return report


def _subsequence_match(
    types: List[str],
    pattern: List[str],
) -> Optional[List[int]]:
    """Find pattern as a subsequence in types (gaps allowed)."""
    indices = []
    j = 0
    for i, t in enumerate(types):
        if j < len(pattern) and t == pattern[j]:
            indices.append(i)
            j += 1
    return indices if j == len(pattern) else None


def _contiguous_match(
    types: List[str],
    pattern: List[str],
) -> Optional[List[int]]:
    """Find pattern as a contiguous substring in types."""
    plen = len(pattern)
    for i in range(len(types) - plen + 1):
        if types[i:i + plen] == pattern:
            return list(range(i, i + plen))
    return None
