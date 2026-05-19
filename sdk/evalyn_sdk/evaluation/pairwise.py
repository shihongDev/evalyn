"""Pairwise comparison (A vs B) with Elo rating system.

Compare two model outputs head-to-head and aggregate results into
Elo ratings for ranking multiple models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PairwiseResult:
    """Result of comparing two outputs on a single item."""

    item_id: str
    model_a: str
    model_b: str
    winner: str  # "a", "b", or "tie"
    confidence: float = 0.0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "winner": self.winner,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PairwiseResult:
        return cls(
            item_id=data["item_id"],
            model_a=data["model_a"],
            model_b=data["model_b"],
            winner=data["winner"],
            confidence=data.get("confidence", 0.0),
            reason=data.get("reason", ""),
        )


@dataclass
class EloRating:
    """Elo rating for a single model."""

    model: str
    rating: float = 1500.0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    games: int = 0

    @property
    def win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.wins / self.games

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "rating": round(self.rating, 2),
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "games": self.games,
            "win_rate": round(self.win_rate, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EloRating:
        return cls(
            model=data["model"],
            rating=data.get("rating", 1500.0),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            ties=data.get("ties", 0),
            games=data.get("games", 0),
        )


@dataclass
class PairwiseReport:
    """Full pairwise comparison report with Elo ratings."""

    results: list[PairwiseResult] = field(default_factory=list)
    ratings: dict[str, EloRating] = field(default_factory=dict)
    total_comparisons: int = 0

    @property
    def win_loss_matrix(self) -> dict[str, dict[str, int]]:
        """Build model_a -> model_b -> win count matrix."""
        matrix: dict[str, dict[str, int]] = {}
        for r in self.results:
            if r.model_a not in matrix:
                matrix[r.model_a] = {}
            if r.model_b not in matrix:
                matrix[r.model_b] = {}
            if r.model_b not in matrix[r.model_a]:
                matrix[r.model_a][r.model_b] = 0
            if r.model_a not in matrix[r.model_b]:
                matrix[r.model_b][r.model_a] = 0
            if r.winner == "a":
                matrix[r.model_a][r.model_b] += 1
            elif r.winner == "b":
                matrix[r.model_b][r.model_a] += 1
        return matrix

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "ratings": {k: v.as_dict() for k, v in self.ratings.items()},
            "total_comparisons": self.total_comparisons,
            "win_loss_matrix": self.win_loss_matrix,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PairwiseReport:
        results = [PairwiseResult.from_dict(r) for r in data.get("results", [])]
        ratings = {
            k: EloRating.from_dict(v) for k, v in data.get("ratings", {}).items()
        }
        return cls(
            results=results,
            ratings=ratings,
            total_comparisons=data.get("total_comparisons", 0),
        )

    def format_text(self) -> str:
        lines = [
            f"Pairwise Report: {self.total_comparisons} comparisons",
            "",
            "Elo Ratings:",
        ]
        sorted_ratings = sorted(
            self.ratings.values(), key=lambda r: r.rating, reverse=True
        )
        for r in sorted_ratings:
            lines.append(
                f"  {r.model:20s}  rating={r.rating:.0f}  "
                f"W={r.wins} L={r.losses} T={r.ties}  "
                f"win_rate={r.win_rate:.1%}"
            )
        return "\n".join(lines)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo expected score: 1 / (1 + 10^((rb - ra) / 400))."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def compare_pair(
    item_id: str,
    output_a: str,
    output_b: str,
    model_a: str = "A",
    model_b: str = "B",
    criterion: str = "quality",
) -> PairwiseResult:
    """Compare two outputs using a simple length heuristic.

    Longer, more detailed output wins. Confidence is based on the
    length difference ratio. This is a local heuristic - no LLM judge.
    """
    len_a = len(output_a.strip())
    len_b = len(output_b.strip())
    max_len = max(len_a, len_b)

    if max_len == 0:
        return PairwiseResult(
            item_id=item_id,
            model_a=model_a,
            model_b=model_b,
            winner="tie",
            confidence=1.0,
            reason="both outputs empty",
        )

    diff_ratio = abs(len_a - len_b) / max_len

    if diff_ratio < 0.05:
        winner = "tie"
        confidence = 1.0 - diff_ratio
        reason = f"similar length ({criterion})"
    elif len_a > len_b:
        winner = "a"
        confidence = min(diff_ratio, 1.0)
        reason = f"longer/more detailed output ({criterion})"
    else:
        winner = "b"
        confidence = min(diff_ratio, 1.0)
        reason = f"longer/more detailed output ({criterion})"

    return PairwiseResult(
        item_id=item_id,
        model_a=model_a,
        model_b=model_b,
        winner=winner,
        confidence=confidence,
        reason=reason,
    )


def compute_elo_ratings(
    results: list[PairwiseResult],
    k_factor: float = 32.0,
    initial_rating: float = 1500.0,
) -> dict[str, EloRating]:
    """Compute Elo ratings from pairwise comparison results.

    K-factor controls how much each game shifts ratings.
    Higher K means more volatile ratings.
    """
    ratings: dict[str, EloRating] = {}

    for r in results:
        if r.model_a not in ratings:
            ratings[r.model_a] = EloRating(model=r.model_a, rating=initial_rating)
        if r.model_b not in ratings:
            ratings[r.model_b] = EloRating(model=r.model_b, rating=initial_rating)

        ra = ratings[r.model_a]
        rb = ratings[r.model_b]

        ea = expected_score(ra.rating, rb.rating)
        eb = 1.0 - ea

        if r.winner == "a":
            sa, sb = 1.0, 0.0
            ra.wins += 1
            rb.losses += 1
        elif r.winner == "b":
            sa, sb = 0.0, 1.0
            ra.losses += 1
            rb.wins += 1
        else:
            sa, sb = 0.5, 0.5
            ra.ties += 1
            rb.ties += 1

        ra.rating += k_factor * (sa - ea)
        rb.rating += k_factor * (sb - eb)
        ra.games += 1
        rb.games += 1

    return ratings


def build_pairwise_report(results: list[PairwiseResult]) -> PairwiseReport:
    """Build a full pairwise report with Elo ratings."""
    ratings = compute_elo_ratings(results)
    return PairwiseReport(
        results=results,
        ratings=ratings,
        total_comparisons=len(results),
    )
