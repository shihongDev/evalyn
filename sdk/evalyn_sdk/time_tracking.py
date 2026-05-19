"""CLI time tracking: aggregate timing stats per command type."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class TimingEntry:
    """A single timing measurement for a CLI command."""

    command: str
    duration_seconds: float
    timestamp: str  # ISO 8601

    def as_dict(self) -> dict:
        return {
            "command": self.command,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TimingEntry:
        return cls(
            command=data["command"],
            duration_seconds=data["duration_seconds"],
            timestamp=data["timestamp"],
        )


@dataclass
class TimingStats:
    """Aggregated timing statistics for a single command type."""

    command: str
    total_seconds: float
    count: int
    avg_seconds: float
    min_seconds: float
    max_seconds: float

    def as_dict(self) -> dict:
        return {
            "command": self.command,
            "total_seconds": self.total_seconds,
            "count": self.count,
            "avg_seconds": self.avg_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TimingStats:
        return cls(
            command=data["command"],
            total_seconds=data["total_seconds"],
            count=data["count"],
            avg_seconds=data["avg_seconds"],
            min_seconds=data["min_seconds"],
            max_seconds=data["max_seconds"],
        )


@dataclass
class TimingReport:
    """Full timing report with per-command stats and totals."""

    stats: list[TimingStats]
    total_commands: int
    total_time_seconds: float

    def as_dict(self) -> dict:
        return {
            "stats": [s.as_dict() for s in self.stats],
            "total_commands": self.total_commands,
            "total_time_seconds": self.total_time_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TimingReport:
        return cls(
            stats=[TimingStats.from_dict(s) for s in data.get("stats", [])],
            total_commands=data.get("total_commands", 0),
            total_time_seconds=data.get("total_time_seconds", 0.0),
        )


def create_timing_entry(command: str, duration: float) -> TimingEntry:
    """Factory: create a TimingEntry with the current UTC timestamp."""
    ts = datetime.now(timezone.utc).isoformat()
    return TimingEntry(command=command, duration_seconds=duration, timestamp=ts)


def log_timing(log_path: str, entry: TimingEntry) -> None:
    """Append a TimingEntry as a JSON line to the timing log file."""
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry.as_dict()) + "\n")


def read_timing_log(log_path: str) -> list[TimingEntry]:
    """Read all TimingEntry records from a JSONL timing log."""
    if not os.path.exists(log_path):
        return []
    entries: list[TimingEntry] = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(TimingEntry.from_dict(json.loads(line)))
    return entries


def compute_stats(entries: list[TimingEntry]) -> list[TimingStats]:
    """Aggregate per-command timing statistics. Sorted by total_seconds descending."""
    buckets: dict[str, list[float]] = {}
    for e in entries:
        buckets.setdefault(e.command, []).append(e.duration_seconds)
    result: list[TimingStats] = []
    for cmd, durations in buckets.items():
        total = sum(durations)
        count = len(durations)
        result.append(
            TimingStats(
                command=cmd,
                total_seconds=total,
                count=count,
                avg_seconds=total / count,
                min_seconds=min(durations),
                max_seconds=max(durations),
            )
        )
    result.sort(key=lambda s: s.total_seconds, reverse=True)
    return result


def build_timing_report(entries: list[TimingEntry]) -> TimingReport:
    """Build a full TimingReport from a list of entries."""
    stats = compute_stats(entries)
    total_commands = len(entries)
    total_time = sum(e.duration_seconds for e in entries)
    return TimingReport(
        stats=stats,
        total_commands=total_commands,
        total_time_seconds=total_time,
    )


def identify_slowest(stats: list[TimingStats], n: int = 5) -> list[TimingStats]:
    """Return the top N slowest commands by average duration."""
    ranked = sorted(stats, key=lambda s: s.avg_seconds, reverse=True)
    return ranked[:n]


def format_timing_report(report: TimingReport) -> str:
    """Format a TimingReport as a human-readable table."""
    lines: list[str] = []
    lines.append("Timing Report")
    lines.append("=" * 72)
    header = f"{'Command':<20} {'Count':>6} {'Total(s)':>10} {'Avg(s)':>10} {'Min(s)':>10} {'Max(s)':>10}"
    lines.append(header)
    lines.append("-" * 72)
    for s in report.stats:
        row = (
            f"{s.command:<20} {s.count:>6} {s.total_seconds:>10.3f} "
            f"{s.avg_seconds:>10.3f} {s.min_seconds:>10.3f} {s.max_seconds:>10.3f}"
        )
        lines.append(row)
    lines.append("-" * 72)
    lines.append(
        f"Total: {report.total_commands} commands, {report.total_time_seconds:.3f}s"
    )
    return "\n".join(lines)
