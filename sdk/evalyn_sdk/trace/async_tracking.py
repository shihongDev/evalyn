from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from ..models import Span


@dataclass
class ConcurrentCall:
    """A single call within a concurrency group."""

    call_id: str
    start_ms: float
    end_ms: float
    duration_ms: float
    group_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "group_id": self.group_id,
        }


@dataclass
class ConcurrencyGroup:
    """A group of overlapping concurrent calls."""

    group_id: str
    calls: list[ConcurrentCall] = field(default_factory=list)
    max_concurrent: int = 0
    total_duration_ms: float = 0.0
    wall_clock_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "calls": [c.as_dict() for c in self.calls],
            "max_concurrent": self.max_concurrent,
            "total_duration_ms": self.total_duration_ms,
            "wall_clock_ms": self.wall_clock_ms,
            "parallelism_ratio": self.parallelism_ratio,
            "speedup": self.speedup,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConcurrencyGroup:
        calls = [
            ConcurrentCall(
                call_id=c["call_id"],
                start_ms=c["start_ms"],
                end_ms=c["end_ms"],
                duration_ms=c["duration_ms"],
                group_id=c.get("group_id", ""),
            )
            for c in data.get("calls", [])
        ]
        return cls(
            group_id=data["group_id"],
            calls=calls,
            max_concurrent=data.get("max_concurrent", 0),
            total_duration_ms=data.get("total_duration_ms", 0.0),
            wall_clock_ms=data.get("wall_clock_ms", 0.0),
        )

    @property
    def parallelism_ratio(self) -> float:
        """wall_clock / total_duration. Lower means more parallel."""
        if self.total_duration_ms > 0:
            return self.wall_clock_ms / self.total_duration_ms
        return 0.0

    @property
    def speedup(self) -> float:
        """total_duration / wall_clock. Higher means more parallel."""
        if self.wall_clock_ms > 0:
            return self.total_duration_ms / self.wall_clock_ms
        return 0.0


@dataclass
class ConcurrencyReport:
    """Full concurrency analysis report."""

    groups: list[ConcurrencyGroup] = field(default_factory=list)
    total_calls: int = 0
    max_concurrent: int = 0
    avg_parallelism: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "groups": [g.as_dict() for g in self.groups],
            "total_calls": self.total_calls,
            "max_concurrent": self.max_concurrent,
            "avg_parallelism": self.avg_parallelism,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConcurrencyReport:
        return cls(
            groups=[ConcurrencyGroup.from_dict(g) for g in data.get("groups", [])],
            total_calls=data.get("total_calls", 0),
            max_concurrent=data.get("max_concurrent", 0),
            avg_parallelism=data.get("avg_parallelism", 0.0),
        )

    def format_text(self) -> str:
        """Human-readable concurrency report."""
        lines = [
            f"Total calls: {self.total_calls}",
            f"Max concurrent: {self.max_concurrent}",
            f"Avg parallelism ratio: {self.avg_parallelism:.3f}",
            f"Concurrency groups: {len(self.groups)}",
        ]
        for g in self.groups:
            lines.append(
                f"  Group {g.group_id}: {len(g.calls)} calls, "
                f"max_concurrent={g.max_concurrent}, "
                f"speedup={g.speedup:.2f}x"
            )
        return "\n".join(lines)


def _span_to_interval(span: Span) -> tuple:
    """Convert span to (start_ms, end_ms) using epoch milliseconds."""
    start_ms = span.start_time.timestamp() * 1000
    end_ms = span.end_time.timestamp() * 1000 if span.end_time else start_ms
    return (start_ms, end_ms)


def detect_concurrent_calls(spans: list[Span]) -> list[ConcurrencyGroup]:
    """Group overlapping spans into concurrency groups.

    Two spans overlap if span_a.start < span_b.end and span_b.start < span_a.end.
    Uses union-find to cluster all transitively overlapping spans.
    """
    if not spans:
        return []

    intervals = [_span_to_interval(s) for s in spans]
    n = len(spans)

    # Union-find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Find all overlapping pairs
    for i in range(n):
        for j in range(i + 1, n):
            si, ei = intervals[i]
            sj, ej = intervals[j]
            if si < ej and sj < ei:
                union(i, j)

    # Build groups
    groups_map: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups_map.setdefault(root, []).append(i)

    result: list[ConcurrencyGroup] = []
    for indices in groups_map.values():
        if len(indices) < 2:
            continue
        gid = str(uuid.uuid4())[:8]
        calls: list[ConcurrentCall] = []
        for idx in indices:
            s_ms, e_ms = intervals[idx]
            calls.append(
                ConcurrentCall(
                    call_id=spans[idx].id,
                    start_ms=s_ms,
                    end_ms=e_ms,
                    duration_ms=e_ms - s_ms,
                    group_id=gid,
                )
            )
        total_dur = sum(c.duration_ms for c in calls)
        wc = compute_wall_clock(calls)
        mc = _max_concurrency_from_calls(calls)
        result.append(
            ConcurrencyGroup(
                group_id=gid,
                calls=calls,
                max_concurrent=mc,
                total_duration_ms=total_dur,
                wall_clock_ms=wc,
            )
        )

    return result


def _max_concurrency_from_calls(calls: list[ConcurrentCall]) -> int:
    """Sweep line over ConcurrentCall list."""
    events: list[tuple] = []
    for c in calls:
        events.append((c.start_ms, 1))
        events.append((c.end_ms, -1))
    events.sort(key=lambda e: (e[0], e[1]))
    current = 0
    best = 0
    for _, delta in events:
        current += delta
        if current > best:
            best = current
    return best


def compute_max_concurrency(spans: list[Span]) -> int:
    """Maximum number of spans active at any point in time. Uses sweep line."""
    if not spans:
        return 0
    events: list[tuple] = []
    for s in spans:
        s_ms, e_ms = _span_to_interval(s)
        events.append((s_ms, 1))
        events.append((e_ms, -1))
    events.sort(key=lambda e: (e[0], e[1]))
    current = 0
    best = 0
    for _, delta in events:
        current += delta
        if current > best:
            best = current
    return best


def compute_wall_clock(calls: list[ConcurrentCall]) -> float:
    """Actual wall clock time covering all calls (merge overlapping intervals)."""
    if not calls:
        return 0.0
    intervals = sorted((c.start_ms, c.end_ms) for c in calls)
    merged: list[tuple] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return sum(e - s for s, e in merged)


def analyze_concurrency(spans: list[Span]) -> ConcurrencyReport:
    """Full concurrency analysis."""
    if not spans:
        return ConcurrencyReport()

    groups = detect_concurrent_calls(spans)
    max_conc = compute_max_concurrency(spans)
    total_calls = len(spans)

    ratios = [g.parallelism_ratio for g in groups if g.parallelism_ratio > 0]
    avg_par = sum(ratios) / len(ratios) if ratios else 0.0

    return ConcurrencyReport(
        groups=groups,
        total_calls=total_calls,
        max_concurrent=max_conc,
        avg_parallelism=avg_par,
    )


def identify_sequential_bottlenecks(spans: list[Span]) -> list[str]:
    """Find spans that could have been parallel but were sequential.

    Detects consecutive non-overlapping spans of the same type.
    Returns list of span IDs that form sequential bottlenecks.
    """
    if len(spans) < 2:
        return []

    # Sort by start time
    sorted_spans = sorted(spans, key=lambda s: s.start_time)
    bottleneck_ids: list[str] = []

    for i in range(len(sorted_spans) - 1):
        a = sorted_spans[i]
        b = sorted_spans[i + 1]
        if a.span_type != b.span_type:
            continue
        _a_start, a_end = _span_to_interval(a)
        b_start, _b_end = _span_to_interval(b)
        # Non-overlapping: b starts at or after a ends
        if b_start >= a_end:
            if a.id not in bottleneck_ids:
                bottleneck_ids.append(a.id)
            if b.id not in bottleneck_ids:
                bottleneck_ids.append(b.id)

    return bottleneck_ids
