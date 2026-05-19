"""Calibration batch processing.

Calibrate multiple metrics in one command by planning execution order,
recording per-metric results, and building an aggregate report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class BatchMetricConfig:
    """Configuration for a single metric in a batch calibration run."""

    metric_id: str
    optimizer: str = "default"
    priority: int = 0  # higher = first

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "optimizer": self.optimizer,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BatchMetricConfig:
        return cls(
            metric_id=data["metric_id"],
            optimizer=data.get("optimizer", "default"),
            priority=data.get("priority", 0),
        )


@dataclass
class BatchCalibrationResult:
    """Result of calibrating a single metric within a batch."""

    metric_id: str
    success: bool
    alignment_before: float = 0.0
    alignment_after: float = 0.0
    improvement: float = 0.0
    error_message: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "success": self.success,
            "alignment_before": self.alignment_before,
            "alignment_after": self.alignment_after,
            "improvement": self.improvement,
            "error_message": self.error_message,
        }


@dataclass
class BatchCalibrationReport:
    """Aggregate report for a batch calibration run."""

    results: List[BatchCalibrationResult] = field(default_factory=list)
    total_metrics: int = 0
    successful: int = 0
    failed: int = 0
    total_improvement: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "total_metrics": self.total_metrics,
            "successful": self.successful,
            "failed": self.failed,
            "total_improvement": self.total_improvement,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BatchCalibrationReport:
        results = []
        for r in data.get("results", []):
            results.append(
                BatchCalibrationResult(
                    metric_id=r["metric_id"],
                    success=r["success"],
                    alignment_before=r.get("alignment_before", 0.0),
                    alignment_after=r.get("alignment_after", 0.0),
                    improvement=r.get("improvement", 0.0),
                    error_message=r.get("error_message", ""),
                )
            )
        return cls(
            results=results,
            total_metrics=data.get("total_metrics", 0),
            successful=data.get("successful", 0),
            failed=data.get("failed", 0),
            total_improvement=data.get("total_improvement", 0.0),
        )

    def format_text(self) -> str:
        lines = [f"Batch Calibration Report: {self.total_metrics} metrics"]
        lines.append(f"  Successful: {self.successful}  Failed: {self.failed}")
        lines.append(f"  Total improvement: {self.total_improvement:.4f}")
        for r in self.results:
            status = "ok" if r.success else f"FAILED - {r.error_message}"
            lines.append(
                f"  {r.metric_id}: {r.alignment_before:.4f} -> {r.alignment_after:.4f}"
                f" ({r.improvement:+.4f}) [{status}]"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_batch_plan(metrics: List[BatchMetricConfig]) -> List[BatchMetricConfig]:
    """Sort metrics by priority descending and return the execution order."""
    return sorted(metrics, key=lambda m: m.priority, reverse=True)


def record_batch_result(
    metric_id: str,
    success: bool,
    before: float = 0.0,
    after: float = 0.0,
    error: str = "",
) -> BatchCalibrationResult:
    """Create a result for a single metric calibration.

    improvement is after - before when success, 0 otherwise.
    """
    improvement = (after - before) if success else 0.0
    return BatchCalibrationResult(
        metric_id=metric_id,
        success=success,
        alignment_before=before,
        alignment_after=after,
        improvement=improvement,
        error_message=error,
    )


def build_batch_report(
    results: List[BatchCalibrationResult],
) -> BatchCalibrationReport:
    """Aggregate individual results into a batch report."""
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    total_improvement = sum(r.improvement for r in results)
    return BatchCalibrationReport(
        results=results,
        total_metrics=len(results),
        successful=successful,
        failed=failed,
        total_improvement=total_improvement,
    )


def estimate_batch_time(
    metrics: List[BatchMetricConfig],
    avg_minutes_per_metric: float = 5.0,
) -> float:
    """Estimate total minutes for a batch calibration run."""
    return len(metrics) * avg_minutes_per_metric
