"""Evaluation timeout per item.

Prevent single slow items from blocking the entire run. Tracks elapsed
time per item, flags timeouts, and suggests adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimeoutConfig:
    """Timeout configuration for evaluation items."""

    default_timeout_seconds: float = 30.0
    per_metric_timeouts: dict[str, float] = field(default_factory=dict)
    abort_on_timeout: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "default_timeout_seconds": self.default_timeout_seconds,
            "per_metric_timeouts": dict(self.per_metric_timeouts),
            "abort_on_timeout": self.abort_on_timeout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeoutConfig:
        return cls(
            default_timeout_seconds=data.get("default_timeout_seconds", 30.0),
            per_metric_timeouts=dict(data.get("per_metric_timeouts", {})),
            abort_on_timeout=data.get("abort_on_timeout", False),
        )


@dataclass
class TimeoutEvent:
    """Record of a single item's timing."""

    item_id: str
    metric_id: str = ""
    elapsed_ms: float = 0.0
    limit_ms: float = 0.0
    timed_out: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "metric_id": self.metric_id,
            "elapsed_ms": self.elapsed_ms,
            "limit_ms": self.limit_ms,
            "timed_out": self.timed_out,
        }


@dataclass
class TimeoutReport:
    """Summary report of timeout events."""

    events: list[TimeoutEvent] = field(default_factory=list)
    total_items: int = 0
    timed_out_count: int = 0
    timeout_rate: float = 0.0
    avg_elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "events": [e.as_dict() for e in self.events],
            "total_items": self.total_items,
            "timed_out_count": self.timed_out_count,
            "timeout_rate": self.timeout_rate,
            "avg_elapsed_ms": self.avg_elapsed_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeoutReport:
        events = [
            TimeoutEvent(**e) for e in data.get("events", [])
        ]
        return cls(
            events=events,
            total_items=data.get("total_items", 0),
            timed_out_count=data.get("timed_out_count", 0),
            timeout_rate=data.get("timeout_rate", 0.0),
            avg_elapsed_ms=data.get("avg_elapsed_ms", 0.0),
        )

    def format_text(self) -> str:
        lines = [
            f"Total items: {self.total_items}",
            f"Timed out: {self.timed_out_count}",
            f"Timeout rate: {self.timeout_rate:.1%}",
            f"Avg elapsed: {self.avg_elapsed_ms:.1f}ms",
        ]
        timed_out_events = [e for e in self.events if e.timed_out]
        if timed_out_events:
            lines.append("Timeout events:")
            for e in timed_out_events:
                parts = [f"  - {e.item_id}"]
                if e.metric_id:
                    parts.append(f"[{e.metric_id}]")
                parts.append(f"{e.elapsed_ms:.1f}ms / {e.limit_ms:.1f}ms")
                lines.append(" ".join(parts))
        return "\n".join(lines)


def get_timeout_for_metric(metric_id: str, config: TimeoutConfig) -> float:
    """Return the timeout in seconds for a given metric.

    Uses per-metric override if configured, otherwise the default.
    """
    return config.per_metric_timeouts.get(
        metric_id, config.default_timeout_seconds
    )


def check_timeout(elapsed_seconds: float, limit_seconds: float) -> bool:
    """Return True if elapsed time exceeds the limit."""
    return elapsed_seconds > limit_seconds


def record_timeout_event(
    item_id: str,
    metric_id: str,
    elapsed_ms: float,
    limit_ms: float,
) -> TimeoutEvent:
    """Create a TimeoutEvent, computing timed_out from elapsed vs limit."""
    timed_out = elapsed_ms > limit_ms
    return TimeoutEvent(
        item_id=item_id,
        metric_id=metric_id,
        elapsed_ms=elapsed_ms,
        limit_ms=limit_ms,
        timed_out=timed_out,
    )


def build_timeout_report(events: list[TimeoutEvent]) -> TimeoutReport:
    """Aggregate a list of timeout events into a report."""
    total = len(events)
    timed_out_count = sum(1 for e in events if e.timed_out)
    timeout_rate = timed_out_count / total if total > 0 else 0.0
    avg_elapsed = (
        sum(e.elapsed_ms for e in events) / total if total > 0 else 0.0
    )
    return TimeoutReport(
        events=list(events),
        total_items=total,
        timed_out_count=timed_out_count,
        timeout_rate=timeout_rate,
        avg_elapsed_ms=avg_elapsed,
    )


def identify_slow_items(
    report: TimeoutReport, threshold_pct: float = 0.8
) -> list[str]:
    """Return item IDs using more than threshold_pct of their timeout budget.

    Args:
        report: The timeout report to analyze.
        threshold_pct: Fraction of timeout budget (0.0 to 1.0). Items using
            more than this fraction are considered slow.

    Returns:
        List of item IDs that are slow but may not have timed out.
    """
    slow: list[str] = []
    for event in report.events:
        if event.limit_ms <= 0:
            continue
        usage = event.elapsed_ms / event.limit_ms
        if usage > threshold_pct:
            slow.append(event.item_id)
    return slow


def suggest_timeout_adjustments(report: TimeoutReport) -> list[str]:
    """Suggest timeout adjustments based on observed timing data."""
    if not report.events:
        return []

    suggestions: list[str] = []

    # High timeout rate - suggest increasing
    if report.timeout_rate > 0.5:
        suggestions.append(
            f"Timeout rate is {report.timeout_rate:.0%} - consider increasing default timeout"
        )

    # All items are fast - suggest decreasing
    if report.total_items > 0 and report.timed_out_count == 0:
        max_elapsed = max(e.elapsed_ms for e in report.events)
        limits = [e.limit_ms for e in report.events if e.limit_ms > 0]
        if limits:
            min_limit = min(limits)
            if max_elapsed < min_limit * 0.3:
                suggestions.append(
                    f"All items finish well under budget (max {max_elapsed:.0f}ms vs "
                    f"{min_limit:.0f}ms limit) - consider reducing timeout"
                )

    # Per-metric analysis: find metrics with high timeout rates
    metric_events: dict[str, list[TimeoutEvent]] = {}
    for event in report.events:
        key = event.metric_id or "(default)"
        if key not in metric_events:
            metric_events[key] = []
        metric_events[key].append(event)

    for metric_id, events in metric_events.items():
        metric_timeouts = sum(1 for e in events if e.timed_out)
        if len(events) >= 2 and metric_timeouts / len(events) > 0.5:
            suggestions.append(
                f"Metric '{metric_id}' has high timeout rate "
                f"({metric_timeouts}/{len(events)}) - consider a longer per-metric timeout"
            )

    return suggestions
