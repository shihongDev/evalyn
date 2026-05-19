"""Metric sensitivity analysis: measure score stability across perturbations.

Evaluates how stable a metric's scores are when small changes are made
to the input. High sensitivity = metric score changes a lot with minor
input changes. Low sensitivity = stable, robust metric.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SensitivityResult:
    """Sensitivity analysis result for a single metric."""

    metric_id: str
    mean_score: float
    std_dev: float
    coefficient_of_variation: float  # std_dev / mean (higher = more sensitive)
    scores: List[float] = field(default_factory=list)
    classification: str = ""  # "stable", "moderate", "volatile"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "mean_score": round(self.mean_score, 4),
            "std_dev": round(self.std_dev, 4),
            "coefficient_of_variation": round(self.coefficient_of_variation, 4),
            "classification": self.classification,
            "num_samples": len(self.scores),
        }

    def format_text(self) -> str:
        return (
            f"{self.metric_id}: {self.classification} "
            f"(CV={self.coefficient_of_variation:.3f}, "
            f"mean={self.mean_score:.3f}, std={self.std_dev:.3f})"
        )


@dataclass
class SensitivityReport:
    """Sensitivity analysis report across all metrics."""

    results: List[SensitivityResult] = field(default_factory=list)

    @property
    def stable_metrics(self) -> List[str]:
        return [r.metric_id for r in self.results if r.classification == "stable"]

    @property
    def volatile_metrics(self) -> List[str]:
        return [r.metric_id for r in self.results if r.classification == "volatile"]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metrics": [r.as_dict() for r in self.results],
            "stable_count": len(self.stable_metrics),
            "volatile_count": len(self.volatile_metrics),
        }

    def format_text(self) -> str:
        lines = ["Metric Sensitivity Analysis", "=" * 40]
        for r in sorted(self.results, key=lambda x: x.coefficient_of_variation, reverse=True):
            lines.append(f"  {r.format_text()}")
        if self.volatile_metrics:
            lines.append("")
            lines.append(f"  Volatile metrics ({len(self.volatile_metrics)}): "
                        f"{', '.join(self.volatile_metrics)}")
            lines.append("  Consider: increase samples or switch judge model for volatile metrics")
        return "\n".join(lines)


def analyze_sensitivity(
    scores_by_metric: Dict[str, List[float]],
    stable_threshold: float = 0.05,
    volatile_threshold: float = 0.15,
) -> SensitivityReport:
    """Analyze metric score sensitivity from multiple evaluation runs.

    Args:
        scores_by_metric: Dict of metric_id -> list of scores across runs.
            Multiple runs of the same data produce multiple scores per metric.
        stable_threshold: CV below this = "stable" (default 0.05 = 5%).
        volatile_threshold: CV above this = "volatile" (default 0.15 = 15%).

    Returns:
        SensitivityReport with per-metric classification.
    """
    results = []
    for metric_id, scores in sorted(scores_by_metric.items()):
        if len(scores) < 2:
            results.append(SensitivityResult(
                metric_id=metric_id,
                mean_score=scores[0] if scores else 0.0,
                std_dev=0.0,
                coefficient_of_variation=0.0,
                scores=list(scores),
                classification="insufficient_data",
            ))
            continue

        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(variance)
        cv = std / abs(mean) if mean != 0 else (0.0 if std == 0 else float("inf"))

        if cv <= stable_threshold:
            classification = "stable"
        elif cv >= volatile_threshold:
            classification = "volatile"
        else:
            classification = "moderate"

        results.append(SensitivityResult(
            metric_id=metric_id,
            mean_score=mean,
            std_dev=std,
            coefficient_of_variation=cv,
            scores=list(scores),
            classification=classification,
        ))

    return SensitivityReport(results=results)


def perturb_text(text: str, perturbation_type: str = "typo", seed: int = 42) -> str:
    """Apply a small perturbation to text for sensitivity testing.

    Args:
        text: Original text.
        perturbation_type: Type of perturbation:
            "typo": swap two adjacent characters
            "case": randomize case of a few characters
            "whitespace": add/remove spaces
        seed: Random seed.

    Returns:
        Perturbed text (guaranteed different from original if len > 1).
    """
    if len(text) <= 1:
        return text

    rng = random.Random(seed)

    if perturbation_type == "typo":
        chars = list(text)
        idx = rng.randint(0, len(chars) - 2)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    elif perturbation_type == "case":
        chars = list(text)
        num_changes = max(1, len(chars) // 10)
        for _ in range(num_changes):
            idx = rng.randint(0, len(chars) - 1)
            chars[idx] = chars[idx].swapcase()
        return "".join(chars)

    elif perturbation_type == "whitespace":
        # Add a space at a random position
        idx = rng.randint(1, len(text) - 1)
        return text[:idx] + " " + text[idx:]

    return text
