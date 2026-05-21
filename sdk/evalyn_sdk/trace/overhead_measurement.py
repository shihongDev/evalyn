from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Span


@dataclass
class OverheadSample:
    """Single overhead measurement sample."""

    operation: str
    with_tracing_ms: float
    without_tracing_ms: float
    overhead_ms: float
    overhead_pct: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "with_tracing_ms": self.with_tracing_ms,
            "without_tracing_ms": self.without_tracing_ms,
            "overhead_ms": self.overhead_ms,
            "overhead_pct": self.overhead_pct,
        }


@dataclass
class OverheadReport:
    """Aggregated overhead measurement report."""

    samples: list[OverheadSample] = field(default_factory=list)
    avg_overhead_ms: float = 0.0
    avg_overhead_pct: float = 0.0
    max_overhead_ms: float = 0.0
    recommendation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": [s.as_dict() for s in self.samples],
            "avg_overhead_ms": self.avg_overhead_ms,
            "avg_overhead_pct": self.avg_overhead_pct,
            "max_overhead_ms": self.max_overhead_ms,
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OverheadReport:
        samples = [
            OverheadSample(
                operation=s["operation"],
                with_tracing_ms=s["with_tracing_ms"],
                without_tracing_ms=s["without_tracing_ms"],
                overhead_ms=s["overhead_ms"],
                overhead_pct=s["overhead_pct"],
            )
            for s in data.get("samples", [])
        ]
        return cls(
            samples=samples,
            avg_overhead_ms=data.get("avg_overhead_ms", 0.0),
            avg_overhead_pct=data.get("avg_overhead_pct", 0.0),
            max_overhead_ms=data.get("max_overhead_ms", 0.0),
            recommendation=data.get("recommendation", ""),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Instrumentation Overhead Report")
        lines.append("-" * 40)
        for s in self.samples:
            lines.append(f"  {s.operation}: +{s.overhead_ms:.2f}ms ({s.overhead_pct:.1f}%)")
        lines.append("-" * 40)
        lines.append(f"Avg overhead: {self.avg_overhead_ms:.2f}ms ({self.avg_overhead_pct:.1f}%)")
        lines.append(f"Max overhead: {self.max_overhead_ms:.2f}ms")
        if self.recommendation:
            lines.append(f"Recommendation: {self.recommendation}")
        return "\n".join(lines)


def compute_overhead(with_tracing_ms: float, without_tracing_ms: float) -> tuple[float, float]:
    """Return (overhead_ms, overhead_pct).

    overhead_ms = with_tracing_ms - without_tracing_ms
    overhead_pct = overhead_ms / without_tracing_ms * 100 if without > 0, else 0.0
    """
    overhead_ms = with_tracing_ms - without_tracing_ms
    if without_tracing_ms > 0:
        overhead_pct = overhead_ms / without_tracing_ms * 100
    else:
        overhead_pct = 0.0
    return overhead_ms, overhead_pct


def record_sample(
    operation: str, with_tracing_ms: float, without_tracing_ms: float
) -> OverheadSample:
    """Create an OverheadSample with computed overhead."""
    overhead_ms, overhead_pct = compute_overhead(with_tracing_ms, without_tracing_ms)
    return OverheadSample(
        operation=operation,
        with_tracing_ms=with_tracing_ms,
        without_tracing_ms=without_tracing_ms,
        overhead_ms=overhead_ms,
        overhead_pct=overhead_pct,
    )


def analyze_overhead(
    samples: list[OverheadSample], max_acceptable_pct: float = 5.0
) -> OverheadReport:
    """Compute averages, max. Recommend disabling tracing if avg > max_acceptable."""
    if not samples:
        return OverheadReport(
            samples=[],
            avg_overhead_ms=0.0,
            avg_overhead_pct=0.0,
            max_overhead_ms=0.0,
            recommendation="No samples to analyze",
        )

    avg_ms = sum(s.overhead_ms for s in samples) / len(samples)
    avg_pct = sum(s.overhead_pct for s in samples) / len(samples)
    max_ms = max(s.overhead_ms for s in samples)

    recommendation = ""
    if avg_pct > max_acceptable_pct:
        recommendation = (
            f"Average overhead {avg_pct:.1f}% exceeds threshold {max_acceptable_pct:.1f}%. "
            "Consider disabling tracing in production."
        )

    return OverheadReport(
        samples=list(samples),
        avg_overhead_ms=avg_ms,
        avg_overhead_pct=avg_pct,
        max_overhead_ms=max_ms,
        recommendation=recommendation,
    )


def estimate_overhead_from_spans(spans: list[Span], baseline_duration_ms: float) -> OverheadReport:
    """Estimate overhead from actual span durations vs a baseline.

    Each span's overhead = (span.duration_ms - baseline) if span.duration_ms > baseline.
    Spans without duration_ms are skipped.
    """
    samples: list[OverheadSample] = []
    for s in spans:
        dur = s.duration_ms
        if dur is None:
            continue
        overhead_ms = dur - baseline_duration_ms if dur > baseline_duration_ms else 0.0
        if baseline_duration_ms > 0:
            overhead_pct = overhead_ms / baseline_duration_ms * 100
        else:
            overhead_pct = 0.0
        samples.append(
            OverheadSample(
                operation=s.name,
                with_tracing_ms=dur,
                without_tracing_ms=baseline_duration_ms,
                overhead_ms=overhead_ms,
                overhead_pct=overhead_pct,
            )
        )

    if not samples:
        return OverheadReport(
            samples=[],
            avg_overhead_ms=0.0,
            avg_overhead_pct=0.0,
            max_overhead_ms=0.0,
            recommendation="No spans with duration to analyze",
        )

    avg_ms = sum(s.overhead_ms for s in samples) / len(samples)
    avg_pct = sum(s.overhead_pct for s in samples) / len(samples)
    max_ms = max(s.overhead_ms for s in samples)

    return OverheadReport(
        samples=samples,
        avg_overhead_ms=avg_ms,
        avg_overhead_pct=avg_pct,
        max_overhead_ms=max_ms,
    )
