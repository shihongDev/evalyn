"""Judge latency optimization: reduce judge call overhead for large-scale evaluation.

Profiles latency per metric, generates optimization suggestions, and
ranks metrics by slowness to focus tuning effort.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyProfile:
    """Latency statistics for a single metric's judge calls."""

    metric_id: str
    avg_latency_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    call_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "call_count": self.call_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatencyProfile:
        return cls(
            metric_id=data["metric_id"],
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            p50_ms=data.get("p50_ms", 0.0),
            p95_ms=data.get("p95_ms", 0.0),
            p99_ms=data.get("p99_ms", 0.0),
            call_count=data.get("call_count", 0),
        )


@dataclass
class OptimizationSuggestion:
    """A suggested optimization with estimated impact."""

    suggestion: str
    estimated_savings_pct: float = 0.0
    priority: str = "medium"  # "low" / "medium" / "high"

    def as_dict(self) -> dict[str, Any]:
        return {
            "suggestion": self.suggestion,
            "estimated_savings_pct": self.estimated_savings_pct,
            "priority": self.priority,
        }


@dataclass
class LatencyReport:
    """Aggregated latency report with optimization suggestions."""

    profiles: list[LatencyProfile] = field(default_factory=list)
    suggestions: list[OptimizationSuggestion] = field(default_factory=list)
    total_calls: int = 0
    total_latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "profiles": [p.as_dict() for p in self.profiles],
            "suggestions": [s.as_dict() for s in self.suggestions],
            "total_calls": self.total_calls,
            "total_latency_ms": self.total_latency_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatencyReport:
        return cls(
            profiles=[LatencyProfile.from_dict(p) for p in data.get("profiles", [])],
            suggestions=[],
            total_calls=data.get("total_calls", 0),
            total_latency_ms=data.get("total_latency_ms", 0.0),
        )

    def format_text(self) -> str:
        """Human-readable text summary."""
        lines: list[str] = []
        lines.append("Latency Report")
        lines.append(f"  Total calls: {self.total_calls}")
        lines.append(f"  Total latency: {self.total_latency_ms:.1f} ms")
        if self.profiles:
            lines.append("  Profiles:")
            for p in self.profiles:
                lines.append(
                    f"    {p.metric_id}: avg={p.avg_latency_ms:.1f}ms "
                    f"p50={p.p50_ms:.1f}ms p95={p.p95_ms:.1f}ms "
                    f"p99={p.p99_ms:.1f}ms calls={p.call_count}"
                )
        if self.suggestions:
            lines.append("  Suggestions:")
            for s in self.suggestions:
                lines.append(
                    f"    [{s.priority}] {s.suggestion} (~{s.estimated_savings_pct:.0f}% savings)"
                )
        return "\n".join(lines)


def compute_percentile(values: list[float], percentile: float) -> float:
    """Compute percentile (0-100) from a list of values.

    Uses linear interpolation between adjacent ranks.
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    # Rank index (0-based)
    k = (percentile / 100.0) * (n - 1)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return sorted_values[lower]
    frac = k - lower
    return sorted_values[lower] + frac * (sorted_values[upper] - sorted_values[lower])


def build_latency_profile(metric_id: str, latencies: list[float]) -> LatencyProfile:
    """Compute avg, p50, p95, p99 from raw latency measurements."""
    if not latencies:
        return LatencyProfile(metric_id=metric_id)
    avg = sum(latencies) / len(latencies)
    return LatencyProfile(
        metric_id=metric_id,
        avg_latency_ms=avg,
        p50_ms=compute_percentile(latencies, 50),
        p95_ms=compute_percentile(latencies, 95),
        p99_ms=compute_percentile(latencies, 99),
        call_count=len(latencies),
    )


def analyze_latency(profiles: list[LatencyProfile]) -> LatencyReport:
    """Analyze all profiles and generate optimization suggestions."""
    total_calls = sum(p.call_count for p in profiles)
    total_latency = sum(p.avg_latency_ms * p.call_count for p in profiles)

    suggestions: list[OptimizationSuggestion] = []

    # Suggest batching if many calls
    if total_calls >= 10:
        suggestions.append(
            OptimizationSuggestion(
                suggestion="Batch calls to reduce per-call overhead",
                estimated_savings_pct=30.0,
                priority="high" if total_calls >= 50 else "medium",
            )
        )

    # Suggest cheaper model if any metric has high p95
    for p in profiles:
        if p.p95_ms > 5000:
            suggestions.append(
                OptimizationSuggestion(
                    suggestion=f"Use cheaper model for {p.metric_id} (p95={p.p95_ms:.0f}ms)",
                    estimated_savings_pct=40.0,
                    priority="high",
                )
            )

    # Suggest caching if repeated inputs likely
    if total_calls >= 5:
        suggestions.append(
            OptimizationSuggestion(
                suggestion="Enable caching for repeated inputs",
                estimated_savings_pct=50.0,
                priority="medium",
            )
        )

    return LatencyReport(
        profiles=list(profiles),
        suggestions=suggestions,
        total_calls=total_calls,
        total_latency_ms=total_latency,
    )


def estimate_optimization_impact(profile: LatencyProfile, optimization: str) -> float:
    """Estimate latency reduction percentage for an optimization type.

    Returns a fraction (0-1) representing the expected reduction.
    """
    reductions = {
        "caching": 0.50,
        "batching": 0.30,
        "cheaper_model": 0.40,
    }
    return reductions.get(optimization, 0.0)


def rank_metrics_by_latency(profiles: list[LatencyProfile]) -> list[LatencyProfile]:
    """Sort profiles by avg_latency descending (slowest first)."""
    return sorted(profiles, key=lambda p: p.avg_latency_ms, reverse=True)
