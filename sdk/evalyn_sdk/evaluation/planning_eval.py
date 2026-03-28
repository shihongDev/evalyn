"""Planning evaluation.

Evaluate plan quality: completeness, ordering, efficiency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PlanStep:
    """A single step in a plan."""

    index: int
    description: str
    dependencies: List[int] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "description": self.description,
            "dependencies": list(self.dependencies),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanStep:
        return cls(
            index=data["index"],
            description=data["description"],
            dependencies=list(data.get("dependencies", [])),
        )


@dataclass
class PlanEvalResult:
    """Result of evaluating a plan."""

    completeness: float
    ordering_correctness: float
    efficiency: float
    replanning_quality: float
    overall: float
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "completeness": self.completeness,
            "ordering_correctness": self.ordering_correctness,
            "efficiency": self.efficiency,
            "replanning_quality": self.replanning_quality,
            "overall": self.overall,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanEvalResult:
        return cls(
            completeness=data["completeness"],
            ordering_correctness=data["ordering_correctness"],
            efficiency=data["efficiency"],
            replanning_quality=data["replanning_quality"],
            overall=data["overall"],
            details=dict(data.get("details", {})),
        )

    def format_text(self) -> str:
        lines = [
            "Plan Evaluation Result",
            f"  completeness:        {self.completeness:.3f}",
            f"  ordering_correctness: {self.ordering_correctness:.3f}",
            f"  efficiency:          {self.efficiency:.3f}",
            f"  replanning_quality:  {self.replanning_quality:.3f}",
            f"  overall:             {self.overall:.3f}",
        ]
        return "\n".join(lines)


def evaluate_completeness(
    plan_steps: List[PlanStep], required_topics: List[str]
) -> float:
    """Fraction of required_topics mentioned in plan step descriptions.

    Uses case-insensitive substring match.
    """
    if not required_topics:
        return 1.0
    covered = 0
    for topic in required_topics:
        topic_lower = topic.lower()
        for step in plan_steps:
            if topic_lower in step.description.lower():
                covered += 1
                break
    return covered / len(required_topics)


def evaluate_ordering(plan_steps: List[PlanStep]) -> float:
    """Check dependency ordering.

    For each step, its dependencies must have lower indices.
    Score = fraction of steps with correctly ordered dependencies.
    """
    if not plan_steps:
        return 1.0
    correct = 0
    for step in plan_steps:
        if all(dep < step.index for dep in step.dependencies):
            correct += 1
    return correct / len(plan_steps)


def evaluate_efficiency(plan_steps: List[PlanStep], max_steps: int = 20) -> float:
    """Penalize overly long plans.

    Score = 1.0 if len <= max_steps/2, linear decrease to 0.0 at 2*max_steps.
    """
    n = len(plan_steps)
    half = max_steps / 2
    double = 2 * max_steps
    if n <= half:
        return 1.0
    if n >= double:
        return 0.0
    return 1.0 - (n - half) / (double - half)


def evaluate_replanning(
    original_plan: List[PlanStep], revised_plan: List[PlanStep]
) -> float:
    """Measure how well a revised plan preserves completed steps.

    Score = fraction of original steps still present in revised plan
    (by description substring match).
    """
    if not original_plan:
        return 1.0
    preserved = 0
    for orig in original_plan:
        orig_lower = orig.description.lower()
        for rev in revised_plan:
            if orig_lower in rev.description.lower() or rev.description.lower() in orig_lower:
                preserved += 1
                break
    return preserved / len(original_plan)


def evaluate_plan(
    plan_steps: List[PlanStep],
    required_topics: List[str] = [],
    max_steps: int = 20,
) -> PlanEvalResult:
    """Full plan evaluation.

    Overall = average of completeness, ordering, efficiency.
    """
    completeness = evaluate_completeness(plan_steps, required_topics)
    ordering = evaluate_ordering(plan_steps)
    efficiency = evaluate_efficiency(plan_steps, max_steps)
    overall = (completeness + ordering + efficiency) / 3.0
    return PlanEvalResult(
        completeness=completeness,
        ordering_correctness=ordering,
        efficiency=efficiency,
        replanning_quality=0.0,
        overall=overall,
        details={
            "step_count": len(plan_steps),
            "max_steps": max_steps,
            "required_topics": list(required_topics),
        },
    )
