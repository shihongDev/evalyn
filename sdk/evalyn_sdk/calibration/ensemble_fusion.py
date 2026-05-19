"""Calibration ensemble fusion via tournament selection.

Run multiple optimizers and fuse outputs by tournament, best-of,
or weighted merge strategies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class OptimizerCandidate:
    """A candidate prompt produced by an optimizer."""

    optimizer_name: str
    prompt: str
    alignment_score: float
    cost_tokens: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "optimizer_name": self.optimizer_name,
            "prompt": self.prompt,
            "alignment_score": self.alignment_score,
            "cost_tokens": self.cost_tokens,
        }


@dataclass
class TournamentResult:
    """Result of an ensemble fusion pass."""

    winner: OptimizerCandidate
    participants: List[OptimizerCandidate]
    rounds: int
    method: str = "tournament"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner.as_dict(),
            "participants": [p.as_dict() for p in self.participants],
            "rounds": self.rounds,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TournamentResult:
        winner = OptimizerCandidate(
            optimizer_name=data["winner"]["optimizer_name"],
            prompt=data["winner"]["prompt"],
            alignment_score=data["winner"]["alignment_score"],
            cost_tokens=data["winner"].get("cost_tokens", 0),
        )
        participants = [
            OptimizerCandidate(
                optimizer_name=p["optimizer_name"],
                prompt=p["prompt"],
                alignment_score=p["alignment_score"],
                cost_tokens=p.get("cost_tokens", 0),
            )
            for p in data.get("participants", [])
        ]
        return cls(
            winner=winner,
            participants=participants,
            rounds=data.get("rounds", 0),
            method=data.get("method", "tournament"),
        )

    def format_text(self) -> str:
        lines = [
            f"Ensemble Fusion Result ({self.method})",
            f"  Winner: {self.winner.optimizer_name}",
            f"  Score:  {self.winner.alignment_score:.4f}",
            f"  Rounds: {self.rounds}",
            f"  Participants: {len(self.participants)}",
        ]
        for p in self.participants:
            lines.append(f"    - {p.optimizer_name}: {p.alignment_score:.4f}")
        return "\n".join(lines)


@dataclass
class FusionConfig:
    """Configuration for ensemble fusion."""

    method: str = "tournament"  # "tournament" / "best_of" / "weighted_merge"
    tournament_rounds: int = 3

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "tournament_rounds": self.tournament_rounds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FusionConfig:
        return cls(
            method=data.get("method", "tournament"),
            tournament_rounds=data.get("tournament_rounds", 3),
        )


# ---------------------------------------------------------------------------
# Fusion Functions
# ---------------------------------------------------------------------------


def tournament_select(
    candidates: List[OptimizerCandidate],
    rounds: int = 3,
    seed: Optional[int] = None,
) -> TournamentResult:
    """Random tournament: pair candidates each round, advance the better one.

    Repeats until one winner remains. If the pool has an odd number,
    the last unpaired candidate advances automatically.
    """
    if not candidates:
        raise ValueError("Cannot run tournament with zero candidates")

    if len(candidates) == 1:
        return TournamentResult(
            winner=candidates[0],
            participants=list(candidates),
            rounds=0,
            method="tournament",
        )

    rng = random.Random(seed)
    pool = list(candidates)
    rounds_played = 0

    for _ in range(rounds):
        if len(pool) <= 1:
            break
        rng.shuffle(pool)
        next_pool: List[OptimizerCandidate] = []
        i = 0
        while i < len(pool):
            if i + 1 < len(pool):
                a, b = pool[i], pool[i + 1]
                next_pool.append(a if a.alignment_score >= b.alignment_score else b)
                i += 2
            else:
                next_pool.append(pool[i])
                i += 1
        pool = next_pool
        rounds_played += 1

    # If multiple remain after all rounds, pick the best
    winner = max(pool, key=lambda c: c.alignment_score)
    return TournamentResult(
        winner=winner,
        participants=list(candidates),
        rounds=rounds_played,
        method="tournament",
    )


def best_of_select(
    candidates: List[OptimizerCandidate],
) -> TournamentResult:
    """Pick the candidate with the highest alignment score."""
    if not candidates:
        raise ValueError("Cannot select from zero candidates")

    winner = max(candidates, key=lambda c: c.alignment_score)
    return TournamentResult(
        winner=winner,
        participants=list(candidates),
        rounds=0,
        method="best_of",
    )


def weighted_merge_prompts(
    candidates: List[OptimizerCandidate],
) -> str:
    """Create a merged prompt from candidates.

    Simple strategy: return the prompt from the best candidate.
    """
    if not candidates:
        return ""
    best = max(candidates, key=lambda c: c.alignment_score)
    return best.prompt


def run_ensemble_fusion(
    candidates: List[OptimizerCandidate],
    config: Optional[FusionConfig] = None,
) -> TournamentResult:
    """Route to the appropriate fusion method based on config."""
    if config is None:
        config = FusionConfig()

    if config.method == "tournament":
        return tournament_select(candidates, rounds=config.tournament_rounds)
    elif config.method == "best_of":
        return best_of_select(candidates)
    elif config.method == "weighted_merge":
        merged = weighted_merge_prompts(candidates)
        # Build a result using the best candidate as winner
        if not candidates:
            raise ValueError("Cannot fuse zero candidates")
        best = max(candidates, key=lambda c: c.alignment_score)
        winner = OptimizerCandidate(
            optimizer_name=best.optimizer_name,
            prompt=merged,
            alignment_score=best.alignment_score,
            cost_tokens=best.cost_tokens,
        )
        return TournamentResult(
            winner=winner,
            participants=list(candidates),
            rounds=0,
            method="weighted_merge",
        )
    else:
        raise ValueError(f"Unknown fusion method: {config.method}")
