"""Calibration convergence visualization.

Plot alignment score vs optimization step to diagnose calibration
behavior - convergence, oscillation, plateau detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConvergencePoint:
    """A single point in the convergence trace."""

    step: int
    alignment_score: float
    prompt_length: int = 0
    tokens_used: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "alignment_score": self.alignment_score,
            "prompt_length": self.prompt_length,
            "tokens_used": self.tokens_used,
        }


@dataclass
class ConvergenceTrace:
    """Full convergence trace for a metric optimization run."""

    metric_id: str
    optimizer: str
    points: list[ConvergencePoint] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "optimizer": self.optimizer,
            "points": [p.as_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConvergenceTrace:
        points = [
            ConvergencePoint(
                step=p["step"],
                alignment_score=p["alignment_score"],
                prompt_length=p.get("prompt_length", 0),
                tokens_used=p.get("tokens_used", 0),
            )
            for p in data.get("points", [])
        ]
        return cls(
            metric_id=data["metric_id"],
            optimizer=data["optimizer"],
            points=points,
        )

    @property
    def best_score(self) -> float:
        """Max alignment score across points."""
        if not self.points:
            return 0.0
        return max(p.alignment_score for p in self.points)

    @property
    def best_step(self) -> int:
        """Step index of the best score."""
        if not self.points:
            return 0
        return max(self.points, key=lambda p: p.alignment_score).step

    @property
    def converged(self) -> bool:
        """True if last 3 points have < 0.01 score variance."""
        if len(self.points) < 3:
            return False
        last_3 = [p.alignment_score for p in self.points[-3:]]
        mean = sum(last_3) / len(last_3)
        variance = sum((s - mean) ** 2 for s in last_3) / len(last_3)
        return variance < 0.01

    @property
    def improvement(self) -> float:
        """Last score minus first score."""
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].alignment_score - self.points[0].alignment_score


class ConvergenceTracker:
    """Records convergence during optimization."""

    def __init__(self, metric_id: str, optimizer: str) -> None:
        self.metric_id = metric_id
        self.optimizer = optimizer
        self._points: list[ConvergencePoint] = []
        self._step: int = 0

    def record(
        self,
        alignment_score: float,
        prompt_length: int = 0,
        tokens_used: int = 0,
    ) -> None:
        """Add a convergence point with auto-incremented step."""
        self._points.append(
            ConvergencePoint(
                step=self._step,
                alignment_score=alignment_score,
                prompt_length=prompt_length,
                tokens_used=tokens_used,
            )
        )
        self._step += 1

    def get_trace(self) -> ConvergenceTrace:
        """Return the recorded trace."""
        return ConvergenceTrace(
            metric_id=self.metric_id,
            optimizer=self.optimizer,
            points=list(self._points),
        )

    def reset(self) -> None:
        """Clear all recorded points."""
        self._points.clear()
        self._step = 0


def render_ascii_convergence(trace: ConvergenceTrace, width: int = 60, height: int = 15) -> str:
    """Render an ASCII plot of alignment score vs step.

    X axis = steps, Y axis = alignment score. Uses '#' for points.
    """
    if not trace.points:
        return "(no data)"

    scores = [p.alignment_score for p in trace.points]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    # Build the grid
    grid = [[" "] * width for _ in range(height)]

    for point in trace.points:
        # Map step to x coordinate
        if len(trace.points) == 1:
            x = width // 2
        else:
            x = int(
                point.step
                / (trace.points[-1].step if trace.points[-1].step > 0 else 1)
                * (width - 1)
            )
        x = min(max(x, 0), width - 1)

        # Map score to y coordinate (inverted: row 0 = top = max score)
        if score_range == 0:
            y = height // 2
        else:
            y = int((1 - (point.alignment_score - min_score) / score_range) * (height - 1))
        y = min(max(y, 0), height - 1)

        grid[y][x] = "#"

    # Assemble lines with Y axis labels
    lines = []
    lines.append(f"  Convergence: {trace.metric_id} ({trace.optimizer})")
    lines.append(f"  Score range: {min_score:.4f} - {max_score:.4f}")
    lines.append("")
    for row_idx, row in enumerate(grid):
        if row_idx == 0:
            label = f"{max_score:.2f}"
        elif row_idx == height - 1:
            label = f"{min_score:.2f}"
        else:
            label = "     "
        lines.append(f"  {label:>5} |{''.join(row)}|")
    lines.append(f"        +{'-' * width}+")
    step_0 = str(trace.points[0].step)
    step_n = str(trace.points[-1].step)
    lines.append(f"         {step_0}{' ' * (width - len(step_0) - len(step_n))}{step_n}")
    lines.append(f"         {'steps':^{width}}")
    return "\n".join(lines)


def diagnose_convergence(trace: ConvergenceTrace) -> dict[str, Any]:
    """Analyze convergence behavior.

    Returns a dict with:
        converged: bool
        plateau_at_step: Optional[int] - step where plateau began
        oscillating: bool
        improving: bool
        recommendation: str
    """
    if not trace.points:
        return {
            "converged": False,
            "plateau_at_step": None,
            "oscillating": False,
            "improving": False,
            "recommendation": "No data recorded - run optimization first",
        }

    if len(trace.points) == 1:
        return {
            "converged": False,
            "plateau_at_step": None,
            "oscillating": False,
            "improving": False,
            "recommendation": "Only one data point - need more iterations",
        }

    scores = [p.alignment_score for p in trace.points]
    converged = trace.converged
    improving = scores[-1] > scores[0]

    # Detect oscillation: count direction changes
    direction_changes = 0
    for i in range(2, len(scores)):
        prev_dir = scores[i - 1] - scores[i - 2]
        curr_dir = scores[i] - scores[i - 1]
        if prev_dir * curr_dir < 0:
            direction_changes += 1
    oscillating = direction_changes >= max(len(scores) // 2, 2)

    # Detect plateau: find first step where remaining scores vary < 0.01
    plateau_at_step: int | None = None
    for i in range(len(scores)):
        remaining = scores[i:]
        if len(remaining) < 3:
            break
        mean = sum(remaining) / len(remaining)
        var = sum((s - mean) ** 2 for s in remaining) / len(remaining)
        if var < 0.01:
            plateau_at_step = trace.points[i].step
            break

    # Build recommendation
    if converged and improving:
        recommendation = "Optimization converged with improvement - good result"
    elif converged and not improving:
        recommendation = "Converged but no improvement - try a different optimizer"
    elif oscillating:
        recommendation = "Scores are oscillating - reduce learning rate or step size"
    elif improving:
        recommendation = "Still improving - consider running more iterations"
    else:
        recommendation = "Not converging - try a different optimization strategy"

    return {
        "converged": converged,
        "plateau_at_step": plateau_at_step,
        "oscillating": oscillating,
        "improving": improving,
        "recommendation": recommendation,
    }
