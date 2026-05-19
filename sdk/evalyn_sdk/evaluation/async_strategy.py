"""Async evaluation strategy: sequential, parallel, and auto-select execution.

Uses ThreadPoolExecutor for parallel execution to keep things simple and
testable without requiring actual asyncio event loops.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AsyncConfig:
    """Configuration for async evaluation strategy."""

    max_concurrency: int = 10
    timeout_seconds: float = 30.0
    strategy: str = "async"  # "sequential", "parallel", "async"

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_concurrency": self.max_concurrency,
            "timeout_seconds": self.timeout_seconds,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AsyncConfig:
        return cls(
            max_concurrency=data.get("max_concurrency", 10),
            timeout_seconds=data.get("timeout_seconds", 30.0),
            strategy=data.get("strategy", "async"),
        )


@dataclass
class AsyncResult:
    """Result for a single task execution."""

    item_id: str
    metric_id: str
    success: bool
    result: dict[str, Any] | None = None
    error: str = ""
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "metric_id": self.metric_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class AsyncReport:
    """Aggregate report for a batch of async tasks."""

    results: list[AsyncResult] = field(default_factory=list)
    total_items: int = 0
    succeeded: int = 0
    failed: int = 0
    total_duration_ms: float = 0.0
    avg_concurrency: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "total_items": self.total_items,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "avg_concurrency": round(self.avg_concurrency, 2),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AsyncReport:
        results = [
            AsyncResult(
                item_id=r["item_id"],
                metric_id=r["metric_id"],
                success=r["success"],
                result=r.get("result"),
                error=r.get("error", ""),
                duration_ms=r.get("duration_ms", 0.0),
            )
            for r in data.get("results", [])
        ]
        return cls(
            results=results,
            total_items=data.get("total_items", 0),
            succeeded=data.get("succeeded", 0),
            failed=data.get("failed", 0),
            total_duration_ms=data.get("total_duration_ms", 0.0),
            avg_concurrency=data.get("avg_concurrency", 0.0),
        )

    def format_text(self) -> str:
        lines = [
            "Async Strategy Report",
            f"  Total items:       {self.total_items}",
            f"  Succeeded:         {self.succeeded}",
            f"  Failed:            {self.failed}",
            f"  Total duration:    {self.total_duration_ms:.1f}ms",
            f"  Avg concurrency:   {self.avg_concurrency:.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _run_single_task(
    item_id: str,
    metric_id: str,
    fn: Callable,
) -> AsyncResult:
    """Execute a single task and capture timing and errors."""
    start = time.monotonic()
    try:
        result = fn()
        duration = (time.monotonic() - start) * 1000
        return AsyncResult(
            item_id=item_id,
            metric_id=metric_id,
            success=True,
            result=result if isinstance(result, dict) else {"value": result},
            duration_ms=duration,
        )
    except Exception as exc:
        duration = (time.monotonic() - start) * 1000
        return AsyncResult(
            item_id=item_id,
            metric_id=metric_id,
            success=False,
            error=str(exc),
            duration_ms=duration,
        )


def create_async_config(
    max_concurrency: int = 10,
    timeout: float = 30.0,
) -> AsyncConfig:
    """Factory for AsyncConfig."""
    return AsyncConfig(max_concurrency=max_concurrency, timeout_seconds=timeout)


def run_sequential(
    tasks: list[tuple[str, str, Callable]],
    config: AsyncConfig,
) -> AsyncReport:
    """Run tasks one by one.

    Each task is a tuple of (item_id, metric_id, callable).
    """
    start = time.monotonic()
    results: list[AsyncResult] = []
    for item_id, metric_id, fn in tasks:
        results.append(_run_single_task(item_id, metric_id, fn))
    total_duration = (time.monotonic() - start) * 1000
    return build_async_report(results, total_duration_override=total_duration, concurrency=1.0)


def run_parallel(
    tasks: list[tuple[str, str, Callable]],
    config: AsyncConfig,
) -> AsyncReport:
    """Run tasks using ThreadPoolExecutor with max_concurrency workers."""
    if not tasks:
        return build_async_report([])
    start = time.monotonic()
    results: list[AsyncResult] = []
    workers = min(config.max_concurrency, len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_single_task, item_id, metric_id, fn): (item_id, metric_id)
            for item_id, metric_id, fn in tasks
        }
        for future in as_completed(futures):
            results.append(future.result())
    total_duration = (time.monotonic() - start) * 1000
    # Estimate average concurrency: sum of individual durations / wall clock time
    sum_durations = sum(r.duration_ms for r in results)
    avg_conc = sum_durations / total_duration if total_duration > 0 else 1.0
    return build_async_report(results, total_duration_override=total_duration, concurrency=avg_conc)


def select_strategy(item_count: int, config: AsyncConfig) -> str:
    """Auto-select execution strategy based on item count.

    Returns "sequential" for fewer than 5 items, "parallel" otherwise.
    """
    if item_count < 5:
        return "sequential"
    return "parallel"


def build_async_report(
    results: list[AsyncResult],
    total_duration_override: float | None = None,
    concurrency: float | None = None,
) -> AsyncReport:
    """Aggregate AsyncResults into an AsyncReport."""
    succeeded = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    total_dur = total_duration_override if total_duration_override is not None else sum(
        r.duration_ms for r in results
    )
    avg_conc = concurrency if concurrency is not None else (1.0 if not results else 1.0)
    return AsyncReport(
        results=results,
        total_items=len(results),
        succeeded=succeeded,
        failed=failed,
        total_duration_ms=total_dur,
        avg_concurrency=avg_conc,
    )
