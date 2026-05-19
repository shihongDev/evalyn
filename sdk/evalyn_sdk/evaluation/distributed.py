"""Distributed evaluation: fan out metric evaluation via task queue."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class TaskMessage:
    """A single evaluation task to be processed by a worker."""

    task_id: str
    item_id: str
    metric_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "item_id": self.item_id,
            "metric_id": self.metric_id,
            "payload": self.payload,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskMessage:
        return cls(
            task_id=data.get("task_id", ""),
            item_id=data.get("item_id", ""),
            metric_id=data.get("metric_id", ""),
            payload=data.get("payload", {}),
            priority=data.get("priority", 0),
        )


@dataclass
class WorkerStatus:
    """Current status of a distributed worker."""

    worker_id: str
    status: str = "idle"
    tasks_completed: int = 0
    current_task: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "status": self.status,
            "tasks_completed": self.tasks_completed,
            "current_task": self.current_task,
        }


@dataclass
class DistributedConfig:
    """Configuration for distributed evaluation."""

    num_workers: int = 4
    queue_type: str = "memory"
    retry_failed: bool = True
    timeout_seconds: float = 60.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "queue_type": self.queue_type,
            "retry_failed": self.retry_failed,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DistributedConfig:
        return cls(
            num_workers=data.get("num_workers", 4),
            queue_type=data.get("queue_type", "memory"),
            retry_failed=data.get("retry_failed", True),
            timeout_seconds=data.get("timeout_seconds", 60.0),
        )


@dataclass
class DistributedReport:
    """Summary report for a distributed evaluation run."""

    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    avg_task_duration_ms: float = 0.0
    workers_used: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "completed": self.completed,
            "failed": self.failed,
            "avg_task_duration_ms": self.avg_task_duration_ms,
            "workers_used": self.workers_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DistributedReport:
        return cls(
            total_tasks=data.get("total_tasks", 0),
            completed=data.get("completed", 0),
            failed=data.get("failed", 0),
            avg_task_duration_ms=data.get("avg_task_duration_ms", 0.0),
            workers_used=data.get("workers_used", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Distributed Evaluation Report")
        lines.append("-" * 40)
        lines.append(f"  Total tasks: {self.total_tasks}")
        lines.append(f"  Completed: {self.completed}")
        lines.append(f"  Failed: {self.failed}")
        lines.append(f"  Avg task duration: {self.avg_task_duration_ms:.1f} ms")
        lines.append(f"  Workers used: {self.workers_used}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TaskQueue
# ---------------------------------------------------------------------------


class TaskQueue:
    """In-memory task queue for local simulation."""

    def __init__(self) -> None:
        self._tasks: list[TaskMessage] = []

    def enqueue(self, task: TaskMessage) -> None:
        self._tasks.append(task)

    def dequeue(self) -> TaskMessage | None:
        if not self._tasks:
            return None
        return self._tasks.pop(0)

    def size(self) -> int:
        return len(self._tasks)

    def is_empty(self) -> bool:
        return len(self._tasks) == 0

    def peek(self) -> TaskMessage | None:
        if not self._tasks:
            return None
        return self._tasks[0]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_task_messages(
    items: list[str], metrics: list[str]
) -> list[TaskMessage]:
    """Create one task per item-metric pair."""
    tasks: list[TaskMessage] = []
    for item_id in items:
        for metric_id in metrics:
            tasks.append(
                TaskMessage(
                    task_id=str(uuid.uuid4()),
                    item_id=item_id,
                    metric_id=metric_id,
                )
            )
    return tasks


def partition_tasks(
    tasks: list[TaskMessage], num_workers: int
) -> list[list[TaskMessage]]:
    """Split tasks into worker partitions using round-robin."""
    if num_workers <= 0:
        return []
    partitions: list[list[TaskMessage]] = [[] for _ in range(num_workers)]
    for i, task in enumerate(tasks):
        partitions[i % num_workers].append(task)
    return partitions


def simulate_distributed_run(
    tasks: list[TaskMessage],
    num_workers: int,
    task_fn: Callable[[TaskMessage], Any],
) -> DistributedReport:
    """Simulate running tasks across workers using ThreadPoolExecutor."""
    if not tasks:
        return DistributedReport(workers_used=min(num_workers, 0))

    effective_workers = min(num_workers, len(tasks))
    completed = 0
    failed = 0
    durations: list[float] = []

    def _run_task(task: TaskMessage) -> bool:
        start = time.monotonic()
        try:
            task_fn(task)
            elapsed_ms = (time.monotonic() - start) * 1000
            durations.append(elapsed_ms)
            return True
        except Exception:
            elapsed_ms = (time.monotonic() - start) * 1000
            durations.append(elapsed_ms)
            return False

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        results = list(executor.map(_run_task, tasks))

    for success in results:
        if success:
            completed += 1
        else:
            failed += 1

    avg_duration = sum(durations) / len(durations) if durations else 0.0

    return DistributedReport(
        total_tasks=len(tasks),
        completed=completed,
        failed=failed,
        avg_task_duration_ms=avg_duration,
        workers_used=effective_workers,
    )


def estimate_distributed_time(
    total_tasks: int,
    num_workers: int,
    avg_task_ms: float = 100.0,
) -> float:
    """Estimate wall clock time in milliseconds for distributed execution."""
    if num_workers <= 0 or total_tasks <= 0:
        return 0.0
    import math

    tasks_per_worker = math.ceil(total_tasks / num_workers)
    return tasks_per_worker * avg_task_ms
