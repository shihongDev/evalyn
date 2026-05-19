"""Calibration comparison dashboard.

Side-by-side view of multiple calibration attempts, grouped by metric,
with optimizer ranking and ASCII rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class CalibrationAttemptSummary:
    """Summary of a single calibration attempt."""

    attempt_id: str
    metric_id: str
    optimizer: str
    alignment_score: float
    cost_tokens: int = 0
    prompt_length: int = 0
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "metric_id": self.metric_id,
            "optimizer": self.optimizer,
            "alignment_score": self.alignment_score,
            "cost_tokens": self.cost_tokens,
            "prompt_length": self.prompt_length,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationAttemptSummary:
        return cls(
            attempt_id=data["attempt_id"],
            metric_id=data["metric_id"],
            optimizer=data["optimizer"],
            alignment_score=data["alignment_score"],
            cost_tokens=data.get("cost_tokens", 0),
            prompt_length=data.get("prompt_length", 0),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class ComparisonEntry:
    """Comparison entry for a single metric across attempts."""

    metric_id: str
    attempts: list[CalibrationAttemptSummary] = field(default_factory=list)
    best_attempt_id: str = ""
    improvement_over_baseline: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "attempts": [a.as_dict() for a in self.attempts],
            "best_attempt_id": self.best_attempt_id,
            "improvement_over_baseline": self.improvement_over_baseline,
        }


@dataclass
class ComparisonDashboard:
    """Full dashboard with all comparison entries."""

    entries: list[ComparisonEntry] = field(default_factory=list)
    total_metrics: int = 0
    total_attempts: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.as_dict() for e in self.entries],
            "total_metrics": self.total_metrics,
            "total_attempts": self.total_attempts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComparisonDashboard:
        entries = []
        for e in data.get("entries", []):
            attempts = [
                CalibrationAttemptSummary.from_dict(a) for a in e.get("attempts", [])
            ]
            entries.append(
                ComparisonEntry(
                    metric_id=e["metric_id"],
                    attempts=attempts,
                    best_attempt_id=e.get("best_attempt_id", ""),
                    improvement_over_baseline=e.get("improvement_over_baseline", 0.0),
                )
            )
        return cls(
            entries=entries,
            total_metrics=data.get("total_metrics", 0),
            total_attempts=data.get("total_attempts", 0),
        )

    def format_text(self) -> str:
        lines = [
            f"Comparison Dashboard: {self.total_metrics} metrics, {self.total_attempts} attempts",
        ]
        for entry in self.entries:
            lines.append(f"  Metric: {entry.metric_id}")
            lines.append(f"    Best: {entry.best_attempt_id}")
            lines.append(
                f"    Improvement over baseline: {entry.improvement_over_baseline:+.4f}"
            )
            for a in entry.attempts:
                marker = " *" if a.attempt_id == entry.best_attempt_id else ""
                lines.append(
                    f"    [{a.optimizer}] score={a.alignment_score:.4f}"
                    f" tokens={a.cost_tokens}{marker}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def build_comparison_entry(
    metric_id: str, attempts: list[CalibrationAttemptSummary]
) -> ComparisonEntry:
    """Find best attempt and compute improvement over first attempt (baseline)."""
    if not attempts:
        return ComparisonEntry(metric_id=metric_id)

    best = max(attempts, key=lambda a: a.alignment_score)
    baseline_score = attempts[0].alignment_score
    improvement = best.alignment_score - baseline_score

    return ComparisonEntry(
        metric_id=metric_id,
        attempts=list(attempts),
        best_attempt_id=best.attempt_id,
        improvement_over_baseline=improvement,
    )


def build_dashboard(
    attempts: list[CalibrationAttemptSummary],
) -> ComparisonDashboard:
    """Group attempts by metric_id and build comparison entries."""
    groups: dict[str, list[CalibrationAttemptSummary]] = {}
    for a in attempts:
        groups.setdefault(a.metric_id, []).append(a)

    entries = []
    for metric_id in sorted(groups.keys()):
        entries.append(build_comparison_entry(metric_id, groups[metric_id]))

    return ComparisonDashboard(
        entries=entries,
        total_metrics=len(entries),
        total_attempts=len(attempts),
    )


def render_dashboard_ascii(dashboard: ComparisonDashboard) -> str:
    """ASCII table showing metric, optimizer, score, cost for each attempt.

    Best attempt per metric is highlighted with an asterisk.
    """
    header = f"{'Metric':<20} {'Optimizer':<15} {'Score':>8} {'Tokens':>8}  Best"
    separator = "-" * len(header)
    lines = [header, separator]

    for entry in dashboard.entries:
        best_id = entry.best_attempt_id
        for a in entry.attempts:
            marker = "  *" if a.attempt_id == best_id else ""
            lines.append(
                f"{a.metric_id:<20} {a.optimizer:<15} {a.alignment_score:>8.4f}"
                f" {a.cost_tokens:>8}{marker}"
            )

    return "\n".join(lines)


def rank_optimizers(
    dashboard: ComparisonDashboard,
) -> list[tuple[str, float]]:
    """Rank optimizers by average alignment score across all metrics.

    Returns list of (optimizer, avg_score) sorted descending by avg_score.
    """
    scores: dict[str, list[float]] = {}
    for entry in dashboard.entries:
        for a in entry.attempts:
            scores.setdefault(a.optimizer, []).append(a.alignment_score)

    ranking = []
    for optimizer, vals in scores.items():
        avg = sum(vals) / len(vals)
        ranking.append((optimizer, avg))

    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking
