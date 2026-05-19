"""Calibration alignment curve.

Plot alignment vs annotation count to find diminishing returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AlignmentPoint:
    """A single point on the alignment curve."""

    num_annotations: int
    alignment_score: float
    marginal_improvement: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_annotations": self.num_annotations,
            "alignment_score": self.alignment_score,
            "marginal_improvement": self.marginal_improvement,
        }


@dataclass
class AlignmentCurve:
    """Alignment score vs annotation count with diminishing returns detection."""

    points: list[AlignmentPoint] = field(default_factory=list)
    diminishing_returns_at: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "points": [p.as_dict() for p in self.points],
            "diminishing_returns_at": self.diminishing_returns_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentCurve:
        points = [
            AlignmentPoint(
                num_annotations=p["num_annotations"],
                alignment_score=p["alignment_score"],
                marginal_improvement=p.get("marginal_improvement", 0.0),
            )
            for p in data.get("points", [])
        ]
        return cls(
            points=points,
            diminishing_returns_at=data.get("diminishing_returns_at"),
        )

    @property
    def max_alignment(self) -> float:
        """Highest alignment score across all points."""
        if not self.points:
            return 0.0
        return max(p.alignment_score for p in self.points)

    @property
    def total_improvement(self) -> float:
        """Last score minus first score."""
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].alignment_score - self.points[0].alignment_score

    def format_text(self) -> str:
        lines = [
            f"Alignment Curve: {len(self.points)} points",
            f"  Max alignment: {self.max_alignment:.4f}",
            f"  Total improvement: {self.total_improvement:.4f}",
        ]
        if self.diminishing_returns_at is not None:
            lines.append(
                f"  Diminishing returns at: {self.diminishing_returns_at} annotations"
            )
        else:
            lines.append("  Diminishing returns: not reached")
        return "\n".join(lines)


def build_alignment_curve(
    annotation_scores: list[tuple[int, float]],
) -> AlignmentCurve:
    """Build curve from (count, score) pairs.

    Computes marginal improvement between consecutive points.
    Detects diminishing returns: first point where marginal_improvement < 0.01.
    """
    if not annotation_scores:
        return AlignmentCurve()

    sorted_pairs = sorted(annotation_scores, key=lambda x: x[0])
    points: list[AlignmentPoint] = []
    diminishing_at: int | None = None

    for i, (count, score) in enumerate(sorted_pairs):
        if i == 0:
            marginal = 0.0
        else:
            marginal = score - sorted_pairs[i - 1][1]

        points.append(
            AlignmentPoint(
                num_annotations=count,
                alignment_score=score,
                marginal_improvement=marginal,
            )
        )

        if diminishing_at is None and i > 0 and marginal < 0.01:
            diminishing_at = count

    return AlignmentCurve(points=points, diminishing_returns_at=diminishing_at)


def render_ascii_curve(
    curve: AlignmentCurve, width: int = 60, height: int = 15
) -> str:
    """Render ASCII plot of alignment vs annotation count.

    X axis = annotation count, Y axis = alignment score. Uses '#' for points.
    """
    if not curve.points:
        return "(no data)"

    scores = [p.alignment_score for p in curve.points]
    counts = [p.num_annotations for p in curve.points]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    min_count = min(counts)
    max_count = max(counts)
    count_range = max_count - min_count

    # Build the grid
    grid = [[" "] * width for _ in range(height)]

    for point in curve.points:
        # Map count to x
        if count_range == 0:
            x = width // 2
        else:
            x = int((point.num_annotations - min_count) / count_range * (width - 1))
        x = min(max(x, 0), width - 1)

        # Map score to y (inverted: row 0 = top = max score)
        if score_range == 0:
            y = height // 2
        else:
            y = int(
                (1 - (point.alignment_score - min_score) / score_range) * (height - 1)
            )
        y = min(max(y, 0), height - 1)

        grid[y][x] = "#"

    # Assemble lines
    lines = []
    lines.append(f"  Alignment Curve ({len(curve.points)} points)")
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
    s0 = str(min_count)
    sn = str(max_count)
    lines.append(f"         {s0}{' ' * (width - len(s0) - len(sn))}{sn}")
    lines.append(f"         {'annotations':^{width}}")
    return "\n".join(lines)


def estimate_optimal_annotations(curve: AlignmentCurve) -> int:
    """Return annotation count at diminishing returns, or max if not reached."""
    if curve.diminishing_returns_at is not None:
        return curve.diminishing_returns_at
    if not curve.points:
        return 0
    return max(p.num_annotations for p in curve.points)


def compute_roi(
    curve: AlignmentCurve, cost_per_annotation: float = 1.0
) -> dict[str, Any]:
    """Compute return on investment for annotations.

    Returns dict with total_cost, total_improvement, cost_per_point,
    optimal_annotations, optimal_cost.
    """
    if not curve.points:
        return {
            "total_cost": 0.0,
            "total_improvement": 0.0,
            "cost_per_point": 0.0,
            "optimal_annotations": 0,
            "optimal_cost": 0.0,
        }

    max_count = max(p.num_annotations for p in curve.points)
    total_cost = max_count * cost_per_annotation
    total_improvement = curve.total_improvement
    cost_per_point = total_cost / total_improvement if total_improvement != 0 else 0.0
    optimal = estimate_optimal_annotations(curve)
    optimal_cost = optimal * cost_per_annotation

    return {
        "total_cost": total_cost,
        "total_improvement": total_improvement,
        "cost_per_point": cost_per_point,
        "optimal_annotations": optimal,
        "optimal_cost": optimal_cost,
    }
