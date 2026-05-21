"""Cross-provider calibration.

Calibrate for consistency when switching judge providers by computing
correlations and score deltas across provider pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ProviderScore:
    """A single score from a provider for an item."""

    provider: str
    item_id: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "item_id": self.item_id,
            "score": self.score,
        }


@dataclass
class ConsistencyResult:
    """Pairwise consistency between two providers."""

    provider_a: str
    provider_b: str
    correlation: float
    mean_delta: float
    max_delta: float
    num_items: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider_a": self.provider_a,
            "provider_b": self.provider_b,
            "correlation": self.correlation,
            "mean_delta": self.mean_delta,
            "max_delta": self.max_delta,
            "num_items": self.num_items,
        }

    def format_text(self) -> str:
        lines = [
            f"{self.provider_a} vs {self.provider_b}:",
            f"  correlation: {self.correlation:.4f}",
            f"  mean delta:  {self.mean_delta:.4f}",
            f"  max delta:   {self.max_delta:.4f}",
            f"  items:       {self.num_items}",
        ]
        return "\n".join(lines)


@dataclass
class CrossProviderReport:
    """Full cross-provider comparison report."""

    comparisons: list[ConsistencyResult]
    most_consistent_pair: tuple[str, str] | None = None
    least_consistent_pair: tuple[str, str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "comparisons": [c.as_dict() for c in self.comparisons],
            "most_consistent_pair": list(self.most_consistent_pair)
            if self.most_consistent_pair
            else None,
            "least_consistent_pair": list(self.least_consistent_pair)
            if self.least_consistent_pair
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossProviderReport:
        comparisons = [
            ConsistencyResult(
                provider_a=c["provider_a"],
                provider_b=c["provider_b"],
                correlation=c["correlation"],
                mean_delta=c["mean_delta"],
                max_delta=c["max_delta"],
                num_items=c["num_items"],
            )
            for c in data.get("comparisons", [])
        ]
        most = data.get("most_consistent_pair")
        least = data.get("least_consistent_pair")
        return cls(
            comparisons=comparisons,
            most_consistent_pair=tuple(most) if most else None,
            least_consistent_pair=tuple(least) if least else None,
        )

    def format_text(self) -> str:
        lines = ["Cross-Provider Consistency Report", ""]
        for c in self.comparisons:
            lines.append(c.format_text())
            lines.append("")
        if self.most_consistent_pair:
            lines.append(
                f"Most consistent:  {self.most_consistent_pair[0]} / {self.most_consistent_pair[1]}"
            )
        if self.least_consistent_pair:
            lines.append(
                f"Least consistent: {self.least_consistent_pair[0]} / {self.least_consistent_pair[1]}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def compute_provider_consistency(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    provider_a: str,
    provider_b: str,
) -> ConsistencyResult:
    """Compute Pearson correlation and mean/max score delta between two providers."""
    common = sorted(set(scores_a) & set(scores_b))
    if not common:
        return ConsistencyResult(
            provider_a=provider_a,
            provider_b=provider_b,
            correlation=0.0,
            mean_delta=0.0,
            max_delta=0.0,
            num_items=0,
        )
    xs = [scores_a[k] for k in common]
    ys = [scores_b[k] for k in common]
    deltas = [abs(x - y) for x, y in zip(xs, ys)]
    return ConsistencyResult(
        provider_a=provider_a,
        provider_b=provider_b,
        correlation=_pearson(xs, ys),
        mean_delta=sum(deltas) / len(deltas),
        max_delta=max(deltas),
        num_items=len(common),
    )


def compare_all_providers(
    provider_scores: dict[str, dict[str, float]],
) -> CrossProviderReport:
    """Compare all provider pairs and identify most/least consistent."""
    providers = sorted(provider_scores)
    comparisons: list[ConsistencyResult] = []
    for i, pa in enumerate(providers):
        for pb in providers[i + 1 :]:
            result = compute_provider_consistency(provider_scores[pa], provider_scores[pb], pa, pb)
            comparisons.append(result)

    most_consistent: tuple[str, str] | None = None
    least_consistent: tuple[str, str] | None = None

    if comparisons:
        valid = [c for c in comparisons if c.num_items > 0]
        if valid:
            best = max(valid, key=lambda c: c.correlation)
            worst = min(valid, key=lambda c: c.correlation)
            most_consistent = (best.provider_a, best.provider_b)
            least_consistent = (worst.provider_a, worst.provider_b)

    return CrossProviderReport(
        comparisons=comparisons,
        most_consistent_pair=most_consistent,
        least_consistent_pair=least_consistent,
    )


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Z-score normalize a provider's scores to zero mean, unit variance."""
    if not scores:
        return {}
    vals = list(scores.values())
    n = len(vals)
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0:
        return dict.fromkeys(scores, 0.0)
    return {k: (v - mean) / std for k, v in scores.items()}


def find_divergent_items(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    threshold: float = 0.3,
) -> list[str]:
    """Items where abs(score_a - score_b) > threshold."""
    common = sorted(set(scores_a) & set(scores_b))
    return [k for k in common if abs(scores_a[k] - scores_b[k]) > threshold]
