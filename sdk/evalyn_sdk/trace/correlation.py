from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse_datetime(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


@dataclass
class ExternalEvent:
    """An external event such as a deployment, incident, or config change."""

    event_id: str
    event_type: str  # "deployment", "incident", "config_change", "release"
    timestamp: datetime
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": _iso(self.timestamp),
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalEvent:
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            timestamp=_parse_datetime(data["timestamp"]) or datetime.now(timezone.utc),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CorrelationMatch:
    """A match between a trace and an external event."""

    trace_id: str
    event_id: str
    time_delta_ms: float
    correlation_type: str = "temporal"

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "event_id": self.event_id,
            "time_delta_ms": self.time_delta_ms,
            "correlation_type": self.correlation_type,
        }


@dataclass
class CorrelationReport:
    """Summary of trace-to-event correlation analysis."""

    matches: list[CorrelationMatch] = field(default_factory=list)
    total_traces: int = 0
    total_events: int = 0
    correlated_traces: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "matches": [m.as_dict() for m in self.matches],
            "total_traces": self.total_traces,
            "total_events": self.total_events,
            "correlated_traces": self.correlated_traces,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorrelationReport:
        return cls(
            matches=[
                CorrelationMatch(**m) for m in data.get("matches", [])
            ],
            total_traces=data.get("total_traces", 0),
            total_events=data.get("total_events", 0),
            correlated_traces=data.get("correlated_traces", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(
            f"Correlation report: {self.correlated_traces}/{self.total_traces} "
            f"traces correlated across {self.total_events} events"
        )
        for m in self.matches:
            lines.append(
                f"  trace={m.trace_id} -> event={m.event_id} "
                f"delta={m.time_delta_ms:.1f}ms type={m.correlation_type}"
            )
        return "\n".join(lines)


def correlate_by_time(
    traces: list[dict[str, Any]],
    events: list[ExternalEvent],
    max_gap_ms: float = 60000.0,
) -> list[CorrelationMatch]:
    """Match traces to events by timestamp proximity.

    Each trace dict has "id" and "timestamp" (datetime).
    Each trace is matched to the nearest event within max_gap_ms.
    """
    if not traces or not events:
        return []

    matches: list[CorrelationMatch] = []
    for trace in traces:
        trace_id = trace["id"]
        trace_ts = trace["timestamp"]
        best_event: ExternalEvent | None = None
        best_delta_ms: float = float("inf")

        for event in events:
            delta = abs((trace_ts - event.timestamp).total_seconds() * 1000.0)
            if delta <= max_gap_ms and delta < best_delta_ms:
                best_delta_ms = delta
                best_event = event

        if best_event is not None:
            matches.append(
                CorrelationMatch(
                    trace_id=trace_id,
                    event_id=best_event.event_id,
                    time_delta_ms=best_delta_ms,
                )
            )

    return matches


def find_events_in_window(
    events: list[ExternalEvent],
    start: datetime,
    end: datetime,
) -> list[ExternalEvent]:
    """Filter events within a time window (inclusive)."""
    return [e for e in events if start <= e.timestamp <= end]


def build_correlation_report(
    traces: list[dict[str, Any]],
    events: list[ExternalEvent],
    max_gap_ms: float = 60000.0,
) -> CorrelationReport:
    """Full correlation analysis between traces and external events."""
    matches = correlate_by_time(traces, events, max_gap_ms)
    correlated_ids = {m.trace_id for m in matches}
    return CorrelationReport(
        matches=matches,
        total_traces=len(traces),
        total_events=len(events),
        correlated_traces=len(correlated_ids),
    )


def annotate_trace_with_events(
    trace_id: str,
    matches: list[CorrelationMatch],
    events: list[ExternalEvent],
) -> dict[str, Any]:
    """Return dict with trace_id and list of correlated events."""
    event_map = {e.event_id: e for e in events}
    correlated: list[dict[str, Any]] = []
    for m in matches:
        if m.trace_id == trace_id and m.event_id in event_map:
            ev = event_map[m.event_id]
            correlated.append({
                "event_id": ev.event_id,
                "event_type": ev.event_type,
                "description": ev.description,
                "time_delta_ms": m.time_delta_ms,
            })
    return {
        "trace_id": trace_id,
        "correlated_events": correlated,
    }
