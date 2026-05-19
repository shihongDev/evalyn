"""Judge debiasing: detect and correct systematic biases in LLM judges.

Identifies position bias, length bias, and verbosity bias in evaluation
scores and applies statistical corrections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BiasMetrics:
    """Measured bias levels for a judge."""

    position_bias: float = 0.0
    length_bias: float = 0.0
    verbosity_bias: float = 0.0

    def has_significant_bias(self, threshold: float = 0.1) -> bool:
        """True if any bias metric exceeds the threshold."""
        return (
            abs(self.position_bias) > threshold
            or abs(self.length_bias) > threshold
            or abs(self.verbosity_bias) > threshold
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "position_bias": round(self.position_bias, 4),
            "length_bias": round(self.length_bias, 4),
            "verbosity_bias": round(self.verbosity_bias, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BiasMetrics:
        return cls(
            position_bias=data.get("position_bias", 0.0),
            length_bias=data.get("length_bias", 0.0),
            verbosity_bias=data.get("verbosity_bias", 0.0),
        )


@dataclass
class DebiasedScore:
    """A score before and after bias correction."""

    original_score: float
    debiased_score: float
    correction: float
    method: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_score": round(self.original_score, 4),
            "debiased_score": round(self.debiased_score, 4),
            "correction": round(self.correction, 4),
            "method": self.method,
        }


@dataclass
class DebiasReport:
    """Full bias analysis report."""

    metrics: BiasMetrics = field(default_factory=BiasMetrics)
    scores: list[DebiasedScore] = field(default_factory=list)
    method: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.as_dict(),
            "scores": [s.as_dict() for s in self.scores],
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DebiasReport:
        metrics = BiasMetrics.from_dict(data.get("metrics", {}))
        scores = [
            DebiasedScore(
                original_score=s["original_score"],
                debiased_score=s["debiased_score"],
                correction=s["correction"],
                method=s["method"],
            )
            for s in data.get("scores", [])
        ]
        return cls(
            metrics=metrics,
            scores=scores,
            method=data.get("method", ""),
        )

    def format_text(self) -> str:
        lines = [
            f"Judge Debiasing Report (method: {self.method})",
            "",
            "Bias Metrics:",
            f"  Position bias:  {self.metrics.position_bias:+.4f}",
            f"  Length bias:    {self.metrics.length_bias:+.4f}",
            f"  Verbosity bias: {self.metrics.verbosity_bias:+.4f}",
            f"  Significant:    {self.metrics.has_significant_bias()}",
            "",
            f"Debiased Scores: {len(self.scores)} items",
        ]
        return "\n".join(lines)


def detect_position_bias(
    scores_first: list[float], scores_second: list[float]
) -> float:
    """Compute position bias as mean score difference (first - second).

    Positive value means the judge favors items in the first position.
    """
    if not scores_first or not scores_second:
        return 0.0
    n = min(len(scores_first), len(scores_second))
    if n == 0:
        return 0.0
    total = sum(scores_first[i] - scores_second[i] for i in range(n))
    return total / n


def detect_length_bias(lengths: list[int], scores: list[float]) -> float:
    """Pearson correlation between output length and score.

    Returns 0.0 for insufficient data (fewer than 2 pairs).
    """
    if len(lengths) < 2 or len(scores) < 2:
        return 0.0
    n = min(len(lengths), len(scores))
    xs = [float(lengths[i]) for i in range(n)]
    ys = [scores[i] for i in range(n)]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    var_x = sum((xs[i] - mean_x) ** 2 for i in range(n))
    var_y = sum((ys[i] - mean_y) ** 2 for i in range(n))

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-12:
        return 0.0
    return cov / denom


def correct_length_bias(
    scores: list[float], lengths: list[int]
) -> list[DebiasedScore]:
    """Apply linear regression correction to remove length bias.

    Subtracts the predicted score component from length, keeps
    residual + mean score.
    """
    if not scores or not lengths:
        return []

    n = min(len(scores), len(lengths))
    xs = [float(lengths[i]) for i in range(n)]
    ys = [scores[i] for i in range(n)]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    var_x = sum((x - mean_x) ** 2 for x in xs)

    if var_x < 1e-12:
        # No variance in lengths - no correction possible
        return [
            DebiasedScore(
                original_score=ys[i],
                debiased_score=ys[i],
                correction=0.0,
                method="length_regression",
            )
            for i in range(n)
        ]

    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    slope = cov / var_x

    results = []
    for i in range(n):
        predicted_component = slope * (xs[i] - mean_x)
        debiased = ys[i] - predicted_component
        correction = -predicted_component
        results.append(
            DebiasedScore(
                original_score=ys[i],
                debiased_score=debiased,
                correction=correction,
                method="length_regression",
            )
        )
    return results


def correct_position_bias(
    scores_first: list[float], scores_second: list[float]
) -> list[float]:
    """Average first-position and second-position scores for each item."""
    if not scores_first or not scores_second:
        return []
    n = min(len(scores_first), len(scores_second))
    return [(scores_first[i] + scores_second[i]) / 2.0 for i in range(n)]


def analyze_judge_bias(
    scores: list[float],
    lengths: list[int],
    positions: list[int] | None = None,
) -> DebiasReport:
    """Full bias analysis: detect and correct biases.

    Args:
        scores: Raw judge scores.
        lengths: Output lengths corresponding to each score.
        positions: Optional position indicators (0 = first, 1 = second).
            If provided, used to split scores for position bias detection.
    """
    if not scores or not lengths:
        return DebiasReport(method="combined")

    length_bias = detect_length_bias(lengths, scores)

    position_bias = 0.0
    if positions is not None and len(positions) == len(scores):
        first_scores = [
            scores[i] for i in range(len(scores)) if positions[i] == 0
        ]
        second_scores = [
            scores[i] for i in range(len(scores)) if positions[i] == 1
        ]
        position_bias = detect_position_bias(first_scores, second_scores)

    # Verbosity bias approximated as length bias (they are closely related)
    verbosity_bias = length_bias

    metrics = BiasMetrics(
        position_bias=position_bias,
        length_bias=length_bias,
        verbosity_bias=verbosity_bias,
    )

    debiased_scores = correct_length_bias(scores, lengths)

    return DebiasReport(
        metrics=metrics,
        scores=debiased_scores,
        method="combined",
    )
