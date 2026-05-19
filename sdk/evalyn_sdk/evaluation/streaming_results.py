"""Streaming partial results for in-progress evaluations.

Start analyzing results before the full batch completes. A StreamingBuffer
accumulates PartialResult objects and computes live statistics as results
arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PartialResult:
    """A single result that arrived from the evaluation stream."""

    item_id: str
    metric_id: str
    score: float
    passed: bool
    timestamp_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "metric_id": self.metric_id,
            "score": self.score,
            "passed": self.passed,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class StreamingStats:
    """Live statistics computed from buffered partial results."""

    total_expected: int
    completed: int
    in_progress: int
    completion_pct: float
    avg_score: float
    pass_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_expected": self.total_expected,
            "completed": self.completed,
            "in_progress": self.in_progress,
            "completion_pct": round(self.completion_pct, 4),
            "avg_score": round(self.avg_score, 4),
            "pass_rate": round(self.pass_rate, 4),
        }

    def format_text(self) -> str:
        lines = [
            "Streaming Stats",
            f"  Completed:  {self.completed}/{self.total_expected}",
            f"  In progress: {self.in_progress}",
            f"  Completion:  {self.completion_pct:.1%}",
            f"  Avg score:   {self.avg_score:.4f}",
            f"  Pass rate:   {self.pass_rate:.1%}",
        ]
        return "\n".join(lines)


@dataclass
class StreamingBuffer:
    """Accumulates partial results and computes live statistics."""

    results: list[PartialResult] = field(default_factory=list)
    expected_count: int = 0

    def add_result(self, result: PartialResult) -> None:
        """Add a result to the buffer."""
        self.results.append(result)

    def get_stats(self) -> StreamingStats:
        """Compute stats from buffered results."""
        completed = len(self.results)
        in_progress = max(0, self.expected_count - completed)
        completion_pct = (
            completed / self.expected_count if self.expected_count > 0 else 0.0
        )
        avg_score = (
            sum(r.score for r in self.results) / completed if completed > 0 else 0.0
        )
        pass_rate = (
            sum(1 for r in self.results if r.passed) / completed
            if completed > 0
            else 0.0
        )
        return StreamingStats(
            total_expected=self.expected_count,
            completed=completed,
            in_progress=in_progress,
            completion_pct=completion_pct,
            avg_score=avg_score,
            pass_rate=pass_rate,
        )

    def get_results_by_metric(self, metric_id: str) -> list[PartialResult]:
        """Filter buffered results by metric id."""
        return [r for r in self.results if r.metric_id == metric_id]

    def get_results_by_item(self, item_id: str) -> list[PartialResult]:
        """Filter buffered results by item id."""
        return [r for r in self.results if r.item_id == item_id]

    def is_complete(self) -> bool:
        """Return True when all expected results have been received."""
        return len(self.results) >= self.expected_count and self.expected_count > 0

    def get_early_insights(self) -> dict[str, Any]:
        """Compute preliminary insights from results received so far."""
        completed = len(self.results)
        if completed == 0:
            return {
                "avg_score": 0.0,
                "pass_rate": 0.0,
                "failing_items": [],
                "completed": 0,
            }

        avg_score = sum(r.score for r in self.results) / completed
        pass_rate = sum(1 for r in self.results if r.passed) / completed

        # Collect unique item ids that have at least one failing result
        failing_items = sorted({r.item_id for r in self.results if not r.passed})

        return {
            "avg_score": round(avg_score, 4),
            "pass_rate": round(pass_rate, 4),
            "failing_items": failing_items,
            "completed": completed,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "expected_count": self.expected_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamingBuffer:
        results = [
            PartialResult(
                item_id=r["item_id"],
                metric_id=r["metric_id"],
                score=r["score"],
                passed=r["passed"],
                timestamp_ms=r.get("timestamp_ms", 0.0),
            )
            for r in data.get("results", [])
        ]
        return cls(
            results=results,
            expected_count=data.get("expected_count", 0),
        )
