"""
Session-level analysis: aggregate metrics across all calls within an eval session.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..models import Span


@dataclass
class SessionSummary:
    """Aggregated metrics for a single eval session."""

    session_id: str
    span_count: int
    total_cost_usd: float
    total_duration_ms: float
    total_tokens: int
    pass_rate: Optional[float]
    error_count: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "span_count": self.span_count,
            "total_cost_usd": self.total_cost_usd,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "pass_rate": self.pass_rate,
            "error_count": self.error_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionSummary:
        return cls(
            session_id=data["session_id"],
            span_count=data["span_count"],
            total_cost_usd=data["total_cost_usd"],
            total_duration_ms=data["total_duration_ms"],
            total_tokens=data["total_tokens"],
            pass_rate=data.get("pass_rate"),
            error_count=data["error_count"],
        )


@dataclass
class SessionComparisonResult:
    """Delta between two session summaries."""

    session_a_id: str
    session_b_id: str
    cost_delta: float
    duration_delta: float
    pass_rate_delta: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "session_a_id": self.session_a_id,
            "session_b_id": self.session_b_id,
            "cost_delta": self.cost_delta,
            "duration_delta": self.duration_delta,
            "pass_rate_delta": self.pass_rate_delta,
        }

    def format_text(self) -> str:
        lines = [
            f"Session comparison: {self.session_a_id} vs {self.session_b_id}",
            f"  Cost delta: {self.cost_delta:+.4f} USD",
            f"  Duration delta: {self.duration_delta:+.1f} ms",
        ]
        if self.pass_rate_delta is not None:
            lines.append(f"  Pass rate delta: {self.pass_rate_delta:+.2%}")
        return "\n".join(lines)


@dataclass
class SessionReport:
    """Report across multiple sessions."""

    sessions: List[SessionSummary]
    total_sessions: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sessions": [s.as_dict() for s in self.sessions],
            "total_sessions": self.total_sessions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionReport:
        return cls(
            sessions=[SessionSummary.from_dict(s) for s in data.get("sessions", [])],
            total_sessions=data["total_sessions"],
        )

    def format_text(self) -> str:
        lines = [f"Session report: {self.total_sessions} session(s)"]
        for s in self.sessions:
            pr = f"{s.pass_rate:.2%}" if s.pass_rate is not None else "N/A"
            lines.append(
                f"  {s.session_id}: {s.span_count} spans, "
                f"${s.total_cost_usd:.4f}, {s.total_duration_ms:.1f} ms, "
                f"pass rate {pr}, {s.error_count} error(s)"
            )
        return "\n".join(lines)


def analyze_session(session_id: str, spans: List[Span]) -> SessionSummary:
    """Aggregate metrics for one session."""
    total_cost = 0.0
    total_tokens = 0
    total_duration = 0.0
    error_count = 0
    passed_values: List[bool] = []

    for span in spans:
        total_cost += span.attributes.get("cost", 0)
        total_tokens += span.attributes.get("tokens", 0)
        dur = span.duration_ms
        if dur is not None:
            total_duration += dur
        if span.status == "error":
            error_count += 1
        p = span.attributes.get("passed", None)
        if p is not None:
            passed_values.append(bool(p))

    pass_rate: Optional[float] = None
    if passed_values:
        pass_rate = sum(1 for v in passed_values if v) / len(passed_values)

    return SessionSummary(
        session_id=session_id,
        span_count=len(spans),
        total_cost_usd=total_cost,
        total_duration_ms=total_duration,
        total_tokens=total_tokens,
        pass_rate=pass_rate,
        error_count=error_count,
    )


def analyze_sessions(spans: List[Span]) -> SessionReport:
    """Group spans by session attribute and analyze each."""
    groups: Dict[str, List[Span]] = defaultdict(list)
    for span in spans:
        sid = span.attributes.get("session_id", "unknown")
        groups[sid].append(span)

    summaries = [analyze_session(sid, group) for sid, group in groups.items()]
    return SessionReport(sessions=summaries, total_sessions=len(summaries))


def compare_sessions(
    a: SessionSummary, b: SessionSummary
) -> SessionComparisonResult:
    """Compute deltas between two session summaries (b - a)."""
    pass_rate_delta: Optional[float] = None
    if a.pass_rate is not None and b.pass_rate is not None:
        pass_rate_delta = b.pass_rate - a.pass_rate

    return SessionComparisonResult(
        session_a_id=a.session_id,
        session_b_id=b.session_id,
        cost_delta=b.total_cost_usd - a.total_cost_usd,
        duration_delta=b.total_duration_ms - a.total_duration_ms,
        pass_rate_delta=pass_rate_delta,
    )
