"""Multi-objective calibration.

Optimize jointly for accuracy and cost using Pareto front analysis
and weighted scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveWeights:
    """Weights for multi-objective optimization."""

    accuracy_weight: float = 0.7
    cost_weight: float = 0.3

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy_weight": self.accuracy_weight,
            "cost_weight": self.cost_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectiveWeights:
        return cls(
            accuracy_weight=data.get("accuracy_weight", 0.7),
            cost_weight=data.get("cost_weight", 0.3),
        )


@dataclass
class ParetoPoint:
    """A single point in the accuracy-cost tradeoff space."""

    prompt: str
    accuracy: float
    cost_tokens: int
    combined_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "accuracy": self.accuracy,
            "cost_tokens": self.cost_tokens,
            "combined_score": self.combined_score,
        }


@dataclass
class MultiObjectiveResult:
    """Result of multi-objective optimization."""

    pareto_front: list[ParetoPoint]
    best_balanced: ParetoPoint | None = None
    best_accuracy: ParetoPoint | None = None
    cheapest: ParetoPoint | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "pareto_front": [p.as_dict() for p in self.pareto_front],
            "best_balanced": self.best_balanced.as_dict()
            if self.best_balanced
            else None,
            "best_accuracy": self.best_accuracy.as_dict()
            if self.best_accuracy
            else None,
            "cheapest": self.cheapest.as_dict() if self.cheapest else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiObjectiveResult:
        pareto_front = [
            ParetoPoint(
                prompt=p["prompt"],
                accuracy=p["accuracy"],
                cost_tokens=p["cost_tokens"],
                combined_score=p["combined_score"],
            )
            for p in data.get("pareto_front", [])
        ]
        best_balanced = None
        if data.get("best_balanced"):
            b = data["best_balanced"]
            best_balanced = ParetoPoint(
                prompt=b["prompt"],
                accuracy=b["accuracy"],
                cost_tokens=b["cost_tokens"],
                combined_score=b["combined_score"],
            )
        best_accuracy = None
        if data.get("best_accuracy"):
            b = data["best_accuracy"]
            best_accuracy = ParetoPoint(
                prompt=b["prompt"],
                accuracy=b["accuracy"],
                cost_tokens=b["cost_tokens"],
                combined_score=b["combined_score"],
            )
        cheapest = None
        if data.get("cheapest"):
            b = data["cheapest"]
            cheapest = ParetoPoint(
                prompt=b["prompt"],
                accuracy=b["accuracy"],
                cost_tokens=b["cost_tokens"],
                combined_score=b["combined_score"],
            )
        return cls(
            pareto_front=pareto_front,
            best_balanced=best_balanced,
            best_accuracy=best_accuracy,
            cheapest=cheapest,
        )

    def format_text(self) -> str:
        lines = ["Multi-Objective Optimization Result", ""]
        lines.append(f"Pareto front: {len(self.pareto_front)} points")
        for p in self.pareto_front:
            lines.append(
                f"  {p.prompt}: accuracy={p.accuracy:.4f}, "
                f"cost={p.cost_tokens}, combined={p.combined_score:.4f}"
            )
        lines.append("")
        if self.best_balanced:
            lines.append(f"Best balanced: {self.best_balanced.prompt}")
        if self.best_accuracy:
            lines.append(f"Best accuracy: {self.best_accuracy.prompt}")
        if self.cheapest:
            lines.append(f"Cheapest:      {self.cheapest.prompt}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_combined_score(
    accuracy: float,
    cost_tokens: int,
    weights: ObjectiveWeights,
    max_tokens: int = 10000,
) -> float:
    """Weighted combination of accuracy and cost efficiency.

    Formula: accuracy_weight * accuracy + cost_weight * (1 - cost_tokens/max_tokens)
    Result is clamped to [0, 1].
    """
    cost_ratio = cost_tokens / max_tokens if max_tokens > 0 else 1.0
    raw = weights.accuracy_weight * accuracy + weights.cost_weight * (1.0 - cost_ratio)
    return max(0.0, min(1.0, raw))


def find_pareto_front(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Return non-dominated points sorted by accuracy descending.

    A point is dominated if another point has both higher accuracy and
    lower cost_tokens.
    """
    if not points:
        return []
    front: list[ParetoPoint] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q.accuracy >= p.accuracy and q.cost_tokens <= p.cost_tokens:
                if q.accuracy > p.accuracy or q.cost_tokens < p.cost_tokens:
                    dominated = True
                    break
        if not dominated:
            front.append(p)
    front.sort(key=lambda p: p.accuracy, reverse=True)
    return front


def select_best_balanced(
    points: list[ParetoPoint],
    weights: ObjectiveWeights,
    max_tokens: int = 10000,
) -> ParetoPoint | None:
    """From a list of points, pick the one with highest combined_score."""
    if not points:
        return None
    # Recompute combined scores with the given weights
    best: ParetoPoint | None = None
    best_score = -1.0
    for p in points:
        score = compute_combined_score(p.accuracy, p.cost_tokens, weights, max_tokens)
        if score > best_score:
            best_score = score
            best = p
    return best


def optimize_multi_objective(
    candidates: list[tuple[str, float, int]],
    weights: ObjectiveWeights | None = None,
) -> MultiObjectiveResult:
    """Full multi-objective analysis.

    candidates: list of (prompt, accuracy, cost_tokens) tuples.
    """
    if weights is None:
        weights = ObjectiveWeights()

    if not candidates:
        return MultiObjectiveResult(pareto_front=[])

    # Find max tokens for normalization
    max_tokens = max(c[2] for c in candidates)
    if max_tokens == 0:
        max_tokens = 1

    # Build ParetoPoints with combined scores
    points: list[ParetoPoint] = []
    for prompt, accuracy, cost_tokens in candidates:
        score = compute_combined_score(accuracy, cost_tokens, weights, max_tokens)
        points.append(
            ParetoPoint(
                prompt=prompt,
                accuracy=accuracy,
                cost_tokens=cost_tokens,
                combined_score=score,
            )
        )

    front = find_pareto_front(points)
    best_balanced = select_best_balanced(front, weights, max_tokens)
    best_accuracy = max(front, key=lambda p: p.accuracy) if front else None
    cheapest = min(front, key=lambda p: p.cost_tokens) if front else None

    return MultiObjectiveResult(
        pareto_front=front,
        best_balanced=best_balanced,
        best_accuracy=best_accuracy,
        cheapest=cheapest,
    )
