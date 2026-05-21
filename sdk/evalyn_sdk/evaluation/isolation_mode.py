"""Evaluation isolation mode.

Run each metric in isolation to prevent crashes from affecting other metrics.
Uses try/except to capture failures and record timing information.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class IsolationConfig:
    """Configuration for isolated metric execution."""

    enabled: bool = True
    timeout_seconds: float = 30.0
    capture_stderr: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
            "capture_stderr": self.capture_stderr,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IsolationConfig:
        return cls(
            enabled=data.get("enabled", True),
            timeout_seconds=data.get("timeout_seconds", 30.0),
            capture_stderr=data.get("capture_stderr", True),
        )


@dataclass
class IsolatedResult:
    """Result from running a single metric in isolation."""

    metric_id: str
    success: bool
    score: float | None = None
    error: str = ""
    stderr_output: str = ""
    duration_ms: float = 0.0
    timed_out: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "success": self.success,
            "score": self.score,
            "error": self.error,
            "stderr_output": self.stderr_output,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
        }


@dataclass
class IsolationReport:
    """Aggregated report from running all metrics in isolation."""

    results: list[IsolatedResult] = field(default_factory=list)
    total_metrics: int = 0
    succeeded: int = 0
    failed: int = 0
    timed_out: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "total_metrics": self.total_metrics,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "timed_out": self.timed_out,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IsolationReport:
        results = []
        for r in data.get("results", []):
            results.append(
                IsolatedResult(
                    metric_id=r["metric_id"],
                    success=r["success"],
                    score=r.get("score"),
                    error=r.get("error", ""),
                    stderr_output=r.get("stderr_output", ""),
                    duration_ms=r.get("duration_ms", 0.0),
                    timed_out=r.get("timed_out", False),
                )
            )
        return cls(
            results=results,
            total_metrics=data.get("total_metrics", 0),
            succeeded=data.get("succeeded", 0),
            failed=data.get("failed", 0),
            timed_out=data.get("timed_out", 0),
        )

    def format_text(self) -> str:
        """Format the report as human-readable text."""
        lines = [
            f"Isolation Report: {self.total_metrics} metrics",
            f"  Succeeded: {self.succeeded}",
            f"  Failed: {self.failed}",
            f"  Timed out: {self.timed_out}",
        ]
        if self.results:
            lines.append("")
            for r in self.results:
                status = "OK" if r.success else "FAIL"
                if r.timed_out:
                    status = "TIMEOUT"
                score_str = f" score={r.score}" if r.score is not None else ""
                lines.append(f"  [{status}] {r.metric_id}{score_str} ({r.duration_ms:.1f}ms)")
                if r.error:
                    lines.append(f"         error: {r.error}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def run_isolated(
    metric_id: str,
    evaluate_fn: Callable[[], dict[str, Any]],
    config: IsolationConfig,
) -> IsolatedResult:
    """Run evaluate_fn in isolation, capturing errors and timing."""
    start = time.monotonic()
    try:
        result = evaluate_fn()
        duration_ms = (time.monotonic() - start) * 1000
        score = result.get("score") if isinstance(result, dict) else None
        return IsolatedResult(
            metric_id=metric_id,
            success=True,
            score=score,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        duration_ms = (time.monotonic() - start) * 1000
        error_msg = str(exc)
        stderr_output = ""
        if config.capture_stderr:
            stderr_output = traceback.format_exc()
        return IsolatedResult(
            metric_id=metric_id,
            success=False,
            error=error_msg,
            stderr_output=stderr_output,
            duration_ms=duration_ms,
        )


def run_all_isolated(
    metrics: list[tuple[str, Callable[[], dict[str, Any]]]],
    config: IsolationConfig | None = None,
) -> IsolationReport:
    """Run all metrics in isolation and aggregate results."""
    if config is None:
        config = IsolationConfig()

    results: list[IsolatedResult] = []
    for metric_id, evaluate_fn in metrics:
        result = run_isolated(metric_id, evaluate_fn, config)
        results.append(result)

    succeeded = sum(1 for r in results if r.success)
    timed_out_count = sum(1 for r in results if r.timed_out)
    failed = len(results) - succeeded

    return IsolationReport(
        results=results,
        total_metrics=len(results),
        succeeded=succeeded,
        failed=failed,
        timed_out=timed_out_count,
    )


def should_use_isolation(
    metric_count: int,
    has_untrusted_metrics: bool = False,
) -> bool:
    """Recommend isolation if metric_count > 5 or has_untrusted_metrics."""
    if has_untrusted_metrics:
        return True
    return metric_count > 5
