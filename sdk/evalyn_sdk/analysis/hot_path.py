"""Hot path detection across traces.

Extracts sequential span-type patterns, ranks by frequency and
cumulative duration, and highlights optimization opportunities.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..models import Span


@dataclass
class PathPattern:
    """A frequently occurring span-type sequence."""

    sequence: tuple[str, ...]  # e.g. ("llm_call", "tool_call", "llm_call")
    count: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    example_span_ids: list[list[str]] = field(default_factory=list)  # up to 3 examples

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": list(self.sequence),
            "count": self.count,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "example_count": len(self.example_span_ids),
        }

    def format_text(self) -> str:
        seq_str = " -> ".join(self.sequence)
        return (
            f"{seq_str} (x{self.count}, "
            f"avg={self.avg_duration_ms:.1f}ms, "
            f"total={self.total_duration_ms:.1f}ms)"
        )


@dataclass
class HotPathReport:
    """Report of frequently occurring span patterns."""

    patterns: list[PathPattern] = field(default_factory=list)
    total_traces: int = 0

    @property
    def top_by_frequency(self) -> list[PathPattern]:
        return sorted(self.patterns, key=lambda p: p.count, reverse=True)

    @property
    def top_by_cost(self) -> list[PathPattern]:
        return sorted(self.patterns, key=lambda p: p.total_duration_ms, reverse=True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_traces": self.total_traces,
            "num_patterns": len(self.patterns),
            "patterns": [p.as_dict() for p in self.top_by_frequency],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HotPathReport:
        patterns = []
        for pd in data.get("patterns", []):
            patterns.append(PathPattern(
                sequence=tuple(pd["sequence"]),
                count=pd.get("count", 0),
                total_duration_ms=pd.get("total_duration_ms", 0.0),
                avg_duration_ms=pd.get("avg_duration_ms", 0.0),
            ))
        return cls(
            patterns=patterns,
            total_traces=data.get("total_traces", 0),
        )

    def format_text(self) -> str:
        lines = [f"Hot Paths ({len(self.patterns)} patterns from {self.total_traces} traces):"]
        for p in self.top_by_frequency[:10]:
            lines.append(f"  {p.format_text()}")
        return "\n".join(lines)


def detect_hot_paths(
    traces: list[list[Span]],
    window_size: int = 3,
    min_count: int = 2,
    max_patterns: int = 20,
) -> HotPathReport:
    """Detect frequently occurring span-type sequences across traces.

    Args:
        traces: List of traces, each trace is a list of spans.
        window_size: Length of span-type sequence to extract (2-5).
        min_count: Minimum occurrences to include a pattern.
        max_patterns: Maximum number of patterns to return.

    Returns:
        HotPathReport with ranked patterns.
    """
    if window_size < 2:
        window_size = 2
    if window_size > 5:
        window_size = 5

    # Count patterns and collect durations
    pattern_counter: Counter = Counter()
    pattern_durations: dict[tuple[str, ...], list[float]] = defaultdict(list)
    pattern_examples: dict[tuple[str, ...], list[list[str]]] = defaultdict(list)

    for trace_spans in traces:
        # Order spans by start time
        ordered = sorted(trace_spans, key=lambda s: s.start_time)
        types = [s.span_type for s in ordered]
        ids = [s.id for s in ordered]

        # Extract sliding windows
        for i in range(len(types) - window_size + 1):
            seq = tuple(types[i:i + window_size])
            pattern_counter[seq] += 1

            # Sum durations of spans in this window
            dur = sum(
                s.duration_ms or 0.0
                for s in ordered[i:i + window_size]
            )
            pattern_durations[seq].append(dur)

            # Save example (up to 3)
            if len(pattern_examples[seq]) < 3:
                pattern_examples[seq].append(ids[i:i + window_size])

    # Build patterns list
    patterns = []
    for seq, count in pattern_counter.most_common():
        if count < min_count:
            break
        durations = pattern_durations[seq]
        total_dur = sum(durations)
        avg_dur = total_dur / len(durations) if durations else 0.0
        patterns.append(PathPattern(
            sequence=seq,
            count=count,
            total_duration_ms=total_dur,
            avg_duration_ms=avg_dur,
            example_span_ids=pattern_examples[seq],
        ))

    return HotPathReport(
        patterns=patterns[:max_patterns],
        total_traces=len(traces),
    )
