"""Hybrid confidence scoring.

Combines multiple confidence methods into a single robust score
via weighted ensemble. Falls back gracefully when methods are
unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MethodScore:
    """Score from a single confidence method."""

    method: str
    score: float
    available: bool = True
    weight: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "score": round(self.score, 6),
            "available": self.available,
            "weight": round(self.weight, 4),
        }


@dataclass
class HybridScore:
    """Combined confidence score from multiple methods."""

    combined_score: float = 0.0
    method_scores: list[MethodScore] = field(default_factory=list)
    methods_used: int = 0
    methods_unavailable: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "combined_score": round(self.combined_score, 6),
            "methods_used": self.methods_used,
            "methods_unavailable": list(self.methods_unavailable),
            "method_scores": [m.as_dict() for m in self.method_scores],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HybridScore:
        method_scores = [
            MethodScore(
                method=m["method"],
                score=m["score"],
                available=m.get("available", True),
                weight=m.get("weight", 1.0),
            )
            for m in data.get("method_scores", [])
        ]
        return cls(
            combined_score=data.get("combined_score", 0.0),
            methods_used=data.get("methods_used", 0),
            methods_unavailable=data.get("methods_unavailable", []),
            method_scores=method_scores,
        )

    def format_text(self) -> str:
        parts = [f"Hybrid confidence: {self.combined_score:.4f}"]
        for ms in self.method_scores:
            status = "used" if ms.available else "unavailable"
            parts.append(f"  {ms.method}: {ms.score:.4f} (w={ms.weight:.2f}, {status})")
        if self.methods_unavailable:
            parts.append(f"  Fallbacks: {', '.join(self.methods_unavailable)}")
        return "\n".join(parts)


# Default weights calibrated for common providers
DEFAULT_WEIGHTS: dict[str, float] = {
    "logprobs": 0.35,
    "consistency": 0.30,
    "verbalized": 0.20,
    "deepconf": 0.15,
}


def compute_hybrid_score(
    scores: dict[str, float | None],
    weights: dict[str, float] | None = None,
) -> HybridScore:
    """Combine confidence scores from multiple methods.

    Falls back gracefully when methods are unavailable (None score).
    Remaining weights are re-normalized to sum to 1.0.

    Args:
        scores: Dict of method_name -> confidence score (None if unavailable).
        weights: Custom weights per method. Defaults to DEFAULT_WEIGHTS.

    Returns:
        HybridScore with combined score and per-method breakdown.
    """
    w = weights or DEFAULT_WEIGHTS

    method_scores = []
    available_sum = 0.0
    weighted_sum = 0.0
    unavailable = []

    for method, score in sorted(scores.items()):
        method_weight = w.get(method, 1.0 / max(len(scores), 1))
        if score is None:
            method_scores.append(
                MethodScore(
                    method=method,
                    score=0.0,
                    available=False,
                    weight=method_weight,
                )
            )
            unavailable.append(method)
        else:
            method_scores.append(
                MethodScore(
                    method=method,
                    score=score,
                    available=True,
                    weight=method_weight,
                )
            )
            available_sum += method_weight
            weighted_sum += method_weight * score

    # Re-normalize by available weight
    if available_sum > 0:
        combined = weighted_sum / available_sum
    else:
        combined = 0.0

    return HybridScore(
        combined_score=combined,
        method_scores=method_scores,
        methods_used=len(scores) - len(unavailable),
        methods_unavailable=unavailable,
    )


def compute_with_bayesian_update(
    scores: dict[str, float | None],
    prior: float = 0.5,
    method_reliability: dict[str, float] | None = None,
) -> HybridScore:
    """Combine confidence using Bayesian updating.

    Uses each method's score as evidence to update a prior belief
    about correctness. Method reliability (0-1) controls how much
    each method shifts the posterior.

    Args:
        scores: Dict of method_name -> confidence score (None if unavailable).
        prior: Prior probability of correctness (0-1).
        method_reliability: How reliable each method is (0-1). Defaults to 0.8.

    Returns:
        HybridScore with Bayesian combined score.
    """
    reliability = method_reliability or {}
    default_rel = 0.8

    method_scores = []
    unavailable = []
    posterior = prior

    for method, score in sorted(scores.items()):
        rel = reliability.get(method, default_rel)

        if score is None:
            method_scores.append(
                MethodScore(
                    method=method,
                    score=0.0,
                    available=False,
                    weight=rel,
                )
            )
            unavailable.append(method)
            continue

        method_scores.append(
            MethodScore(
                method=method,
                score=score,
                available=True,
                weight=rel,
            )
        )

        # Bayesian update: P(correct|evidence) using likelihood ratio
        # P(score|correct) ~ rel * score + (1-rel) * 0.5
        # P(score|incorrect) ~ rel * (1-score) + (1-rel) * 0.5
        p_evidence_given_correct = rel * score + (1 - rel) * 0.5
        p_evidence_given_incorrect = rel * (1 - score) + (1 - rel) * 0.5

        numerator = p_evidence_given_correct * posterior
        denominator = p_evidence_given_correct * posterior + p_evidence_given_incorrect * (
            1 - posterior
        )

        if denominator > 0:
            posterior = numerator / denominator

    return HybridScore(
        combined_score=posterior,
        method_scores=method_scores,
        methods_used=len(scores) - len(unavailable),
        methods_unavailable=unavailable,
    )
