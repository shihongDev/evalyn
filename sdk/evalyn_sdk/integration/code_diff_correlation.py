"""
Source code diff correlation: track agent code changes alongside metric changes.

Provides dataclasses and pure functions for correlating code commits with
metric movements, identifying impactful changes, rendering ASCII timelines,
and suggesting root causes based on files changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class CodeChange:
    """A single code change (commit) with metadata."""

    commit_hash: str = ""
    files_changed: list[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    description: str = ""
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "commit_hash": self.commit_hash,
            "files_changed": list(self.files_changed),
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "description": self.description,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeChange:
        return cls(
            commit_hash=data.get("commit_hash", ""),
            files_changed=list(data.get("files_changed", [])),
            lines_added=data.get("lines_added", 0),
            lines_removed=data.get("lines_removed", 0),
            description=data.get("description", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class MetricChange:
    """A metric's before/after scores for a single change."""

    metric_id: str = ""
    before_score: float = 0.0
    after_score: float = 0.0
    delta: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "before_score": self.before_score,
            "after_score": self.after_score,
            "delta": self.delta,
        }


@dataclass
class CorrelationEntry:
    """Links a code change to its metric impacts."""

    code_change: CodeChange = field(default_factory=CodeChange)
    metric_changes: list[MetricChange] = field(default_factory=list)
    correlation_strength: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "code_change": self.code_change.as_dict(),
            "metric_changes": [m.as_dict() for m in self.metric_changes],
            "correlation_strength": self.correlation_strength,
        }


@dataclass
class CodeMetricReport:
    """Aggregate report of code-metric correlations."""

    entries: list[CorrelationEntry] = field(default_factory=list)
    strongest_correlation: str | None = None
    total_changes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.as_dict() for e in self.entries],
            "strongest_correlation": self.strongest_correlation,
            "total_changes": self.total_changes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeMetricReport:
        entries = []
        for entry_data in data.get("entries", []):
            code_change = CodeChange.from_dict(entry_data.get("code_change", {}))
            metric_changes = [
                MetricChange(
                    metric_id=m.get("metric_id", ""),
                    before_score=m.get("before_score", 0.0),
                    after_score=m.get("after_score", 0.0),
                    delta=m.get("delta", 0.0),
                )
                for m in entry_data.get("metric_changes", [])
            ]
            entries.append(
                CorrelationEntry(
                    code_change=code_change,
                    metric_changes=metric_changes,
                    correlation_strength=entry_data.get("correlation_strength", 0.0),
                )
            )
        return cls(
            entries=entries,
            strongest_correlation=data.get("strongest_correlation"),
            total_changes=data.get("total_changes", 0),
        )

    def format_text(self) -> str:
        lines = [
            "Code-Metric Correlation Report",
            f"  total_changes: {self.total_changes}",
            f"  strongest_correlation: {self.strongest_correlation or 'none'}",
        ]
        for entry in self.entries:
            desc = entry.code_change.description or entry.code_change.commit_hash
            lines.append(f"  [{desc}] strength={entry.correlation_strength:.3f}")
            for mc in entry.metric_changes:
                lines.append(
                    f"    {mc.metric_id}: {mc.before_score:.2f} -> "
                    f"{mc.after_score:.2f} (delta={mc.delta:+.3f})"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def correlate_change(
    code_change: CodeChange,
    metric_changes: list[MetricChange],
) -> CorrelationEntry:
    """Compute correlation strength as avg abs(delta) of metric changes."""
    if not metric_changes:
        return CorrelationEntry(
            code_change=code_change,
            metric_changes=[],
            correlation_strength=0.0,
        )
    total = sum(abs(m.delta) for m in metric_changes)
    strength = total / len(metric_changes)
    return CorrelationEntry(
        code_change=code_change,
        metric_changes=list(metric_changes),
        correlation_strength=strength,
    )


def build_correlation_report(
    entries: list[CorrelationEntry],
) -> CodeMetricReport:
    """Build a report and find the strongest correlation."""
    if not entries:
        return CodeMetricReport(entries=[], strongest_correlation=None, total_changes=0)

    strongest = max(entries, key=lambda e: e.correlation_strength)
    label = strongest.code_change.description or strongest.code_change.commit_hash
    return CodeMetricReport(
        entries=list(entries),
        strongest_correlation=label if strongest.correlation_strength > 0 else None,
        total_changes=len(entries),
    )


def identify_impactful_changes(
    report: CodeMetricReport,
    threshold: float = 0.1,
) -> list[CorrelationEntry]:
    """Return entries whose correlation strength exceeds the threshold."""
    return [e for e in report.entries if e.correlation_strength > threshold]


def render_correlation_timeline(entries: list[CorrelationEntry]) -> str:
    """ASCII timeline of changes and their metric impacts."""
    if not entries:
        return "No entries to display."

    lines: list[str] = ["Timeline of Code-Metric Correlations", ""]
    for i, entry in enumerate(entries):
        cc = entry.code_change
        marker = f"[{i + 1}]"
        desc = cc.description or cc.commit_hash or "unknown"
        ts = cc.timestamp or "no timestamp"
        lines.append(f"{marker} {ts} - {desc}")
        lines.append(f"    files: {', '.join(cc.files_changed) or 'none'}")
        lines.append(f"    +{cc.lines_added} -{cc.lines_removed}")
        lines.append(f"    correlation_strength: {entry.correlation_strength:.3f}")
        for mc in entry.metric_changes:
            direction = "+" if mc.delta >= 0 else ""
            lines.append(f"      {mc.metric_id}: {direction}{mc.delta:.3f}")
        lines.append("")
    return "\n".join(lines)


def suggest_root_causes(entry: CorrelationEntry) -> list[str]:
    """Suggest root causes based on files changed in the code change."""
    suggestions: list[str] = []
    files = entry.code_change.files_changed

    if not files:
        suggestions.append("No files recorded - unable to suggest root causes")
        return suggestions

    for f in files:
        lower = f.lower()
        if "prompt" in lower:
            suggestions.append(f"{f} changed - likely prompt-related")
        elif "model" in lower or "llm" in lower:
            suggestions.append(f"{f} changed - likely model configuration change")
        elif "metric" in lower or "scorer" in lower or "eval" in lower:
            suggestions.append(f"{f} changed - likely evaluation logic change")
        elif "config" in lower or "settings" in lower:
            suggestions.append(f"{f} changed - likely configuration change")
        elif "test" in lower:
            suggestions.append(f"{f} changed - likely test-related change")
        elif "data" in lower or "dataset" in lower:
            suggestions.append(f"{f} changed - likely data pipeline change")
        else:
            suggestions.append(f"{f} changed - review for potential impact")

    return suggestions
