"""
CLI progress dashboard: unified progress view for concurrent operations.

Pure Python, no external dependencies. Provides data models, state
management, and text rendering for multi-task progress tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TaskProgress:
    """Progress state for a single task."""

    task_id: str
    name: str
    progress: float  # 0.0 to 1.0
    status: str  # "pending", "running", "complete", "failed"
    message: str = ""
    items_done: int = 0
    items_total: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "progress": self.progress,
            "status": self.status,
            "message": self.message,
            "items_done": self.items_done,
            "items_total": self.items_total,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskProgress:
        return cls(
            task_id=data["task_id"],
            name=data["name"],
            progress=data.get("progress", 0.0),
            status=data.get("status", "pending"),
            message=data.get("message", ""),
            items_done=data.get("items_done", 0),
            items_total=data.get("items_total", 0),
        )


@dataclass
class DashboardConfig:
    """Display configuration for the progress dashboard."""

    width: int = 60
    show_elapsed: bool = True
    show_rate: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "show_elapsed": self.show_elapsed,
            "show_rate": self.show_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DashboardConfig:
        return cls(
            width=data.get("width", 60),
            show_elapsed=data.get("show_elapsed", True),
            show_rate=data.get("show_rate", True),
        )


@dataclass
class DashboardState:
    """Snapshot of the full dashboard state."""

    tasks: list[TaskProgress] = field(default_factory=list)
    start_time: float = 0.0
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tasks": [t.as_dict() for t in self.tasks],
            "start_time": self.start_time,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DashboardState:
        tasks = [TaskProgress.from_dict(t) for t in data.get("tasks", [])]
        return cls(
            tasks=tasks,
            start_time=data.get("start_time", 0.0),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
        )


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def create_dashboard(task_names: list[str]) -> DashboardState:
    """Initialize a dashboard with pending tasks.

    Each task gets a task_id derived from its index ("task_0", "task_1", ...).
    """
    tasks = [
        TaskProgress(
            task_id=f"task_{i}",
            name=name,
            progress=0.0,
            status="pending",
        )
        for i, name in enumerate(task_names)
    ]
    return DashboardState(
        tasks=tasks,
        start_time=time.time(),
        elapsed_seconds=0.0,
    )


def update_task(
    state: DashboardState,
    task_id: str,
    progress: float,
    status: str = "running",
    message: str = "",
    items_done: int = 0,
) -> DashboardState:
    """Return a new DashboardState with the specified task updated.

    Does not mutate the original state.
    """
    new_tasks: list[TaskProgress] = []
    for task in state.tasks:
        if task.task_id == task_id:
            new_tasks.append(
                TaskProgress(
                    task_id=task.task_id,
                    name=task.name,
                    progress=progress,
                    status=status,
                    message=message,
                    items_done=items_done,
                    items_total=task.items_total,
                )
            )
        else:
            new_tasks.append(
                TaskProgress(
                    task_id=task.task_id,
                    name=task.name,
                    progress=task.progress,
                    status=task.status,
                    message=task.message,
                    items_done=task.items_done,
                    items_total=task.items_total,
                )
            )
    elapsed = time.time() - state.start_time
    return DashboardState(
        tasks=new_tasks,
        start_time=state.start_time,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
    "pending": "..",
    "running": ">>",
    "complete": "OK",
    "failed": "!!",
}


def render_progress_bar(progress: float, width: int = 30) -> str:
    """Render a progress bar string.

    Example: "[=====>              ] 50%"
    """
    progress = max(0.0, min(1.0, progress))
    filled = int(progress * width)
    pct = int(progress * 100)
    if filled > 0 and filled < width:
        bar = "=" * (filled - 1) + ">" + " " * (width - filled)
    elif filled == width:
        bar = "=" * width
    else:
        bar = " " * width
    return f"[{bar}] {pct:>3d}%"


def render_task_line(task: TaskProgress, width: int = 60) -> str:
    """Render a single task progress line.

    Example: "  [OK] TaskName [========] 100% (50/50 items)"
    """
    icon = _STATUS_ICONS.get(task.status, "??")
    bar_width = max(8, width - 40)
    bar = render_progress_bar(task.progress, bar_width)
    line = f"  [{icon}] {task.name:<12s} {bar}"
    if task.items_total > 0:
        line += f" ({task.items_done}/{task.items_total} items)"
    if task.message:
        line += f" {task.message}"
    return line


def render_dashboard(
    state: DashboardState, config: DashboardConfig | None = None
) -> str:
    """Render the full dashboard as a multi-line string.

    Includes header, per-task progress bars, and summary line.
    """
    if config is None:
        config = DashboardConfig()

    lines: list[str] = []
    overall = get_overall_progress(state)
    overall_pct = int(overall * 100)

    # Header
    lines.append(f"Progress: {overall_pct}%")
    lines.append("-" * config.width)

    # Per-task lines
    for task in state.tasks:
        lines.append(render_task_line(task, config.width))

    # Summary
    lines.append("-" * config.width)

    done_count = sum(
        1 for t in state.tasks if t.status in ("complete", "failed")
    )
    total_count = len(state.tasks)
    summary_parts: list[str] = [f"{done_count}/{total_count} tasks done"]

    if config.show_elapsed:
        elapsed = state.elapsed_seconds
        if elapsed >= 60:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            summary_parts.append(f"elapsed {mins}m{secs}s")
        else:
            summary_parts.append(f"elapsed {elapsed:.1f}s")

    lines.append("  ".join(summary_parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def get_overall_progress(state: DashboardState) -> float:
    """Return the mean progress across all tasks (0.0 to 1.0)."""
    if not state.tasks:
        return 0.0
    return sum(t.progress for t in state.tasks) / len(state.tasks)


def is_complete(state: DashboardState) -> bool:
    """Return True if all tasks are complete or failed."""
    if not state.tasks:
        return True
    return all(t.status in ("complete", "failed") for t in state.tasks)
