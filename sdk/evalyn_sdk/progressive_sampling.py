"""Progressive sampling: start small, expand if metrics are inconclusive.

Iteratively grows the sample until the confidence interval is narrow
enough or the budget is exhausted. Pure Python, no external dependencies.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProgressiveConfig:
    """Configuration for progressive sampling."""

    initial_size: int = 20
    expansion_factor: float = 2.0
    max_size: int = 500
    ci_width_threshold: float = 0.1
    max_rounds: int = 5
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "initial_size": self.initial_size,
            "expansion_factor": self.expansion_factor,
            "max_size": self.max_size,
            "ci_width_threshold": self.ci_width_threshold,
            "max_rounds": self.max_rounds,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgressiveConfig:
        return cls(
            initial_size=data.get("initial_size", 20),
            expansion_factor=data.get("expansion_factor", 2.0),
            max_size=data.get("max_size", 500),
            ci_width_threshold=data.get("ci_width_threshold", 0.1),
            max_rounds=data.get("max_rounds", 5),
            seed=data.get("seed"),
        )


@dataclass
class ProgressiveRound:
    """One round of progressive sampling."""

    round_number: int
    sample_size: int
    ci_width: float
    sufficient: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "sample_size": self.sample_size,
            "ci_width": self.ci_width,
            "sufficient": self.sufficient,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgressiveRound:
        return cls(
            round_number=data["round_number"],
            sample_size=data["sample_size"],
            ci_width=data["ci_width"],
            sufficient=data["sufficient"],
        )


@dataclass
class ProgressiveResult:
    """Result of progressive sampling."""

    final_sample_ids: list[str] = field(default_factory=list)
    rounds: list[ProgressiveRound] = field(default_factory=list)
    total_rounds: int = 0
    final_ci_width: float = 0.0
    converged: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "final_sample_ids": self.final_sample_ids,
            "rounds": [r.as_dict() for r in self.rounds],
            "total_rounds": self.total_rounds,
            "final_ci_width": self.final_ci_width,
            "converged": self.converged,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgressiveResult:
        return cls(
            final_sample_ids=data.get("final_sample_ids", []),
            rounds=[ProgressiveRound.from_dict(r) for r in data.get("rounds", [])],
            total_rounds=data.get("total_rounds", 0),
            final_ci_width=data.get("final_ci_width", 0.0),
            converged=data.get("converged", False),
        )


def compute_ci_width(scores: list[float], confidence: float = 0.95) -> float:
    """Width of confidence interval: z * std / sqrt(n).

    Uses z=1.96 for 95% confidence. Returns 0.0 for fewer than 2 scores.
    """
    n = len(scores)
    if n < 2:
        return 0.0
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    std = math.sqrt(variance)
    z = 1.96  # 95% confidence
    return z * std / math.sqrt(n)


def is_sufficient(ci_width: float, threshold: float) -> bool:
    """True if the confidence interval is narrow enough."""
    return ci_width <= threshold


def expand_sample(
    current_ids: list[str],
    all_ids: list[str],
    expansion_factor: float,
    max_size: int,
    seed: int | None = None,
) -> list[str]:
    """Expand the current sample by adding more items from the remaining pool.

    The new target size is min(len(current) * expansion_factor, max_size).
    New items are drawn randomly from the pool excluding current_ids.
    Returns the full expanded list (current + new).
    """
    current_set = set(current_ids)
    remaining = [x for x in all_ids if x not in current_set]

    new_target = min(int(len(current_ids) * expansion_factor), max_size)
    to_add = new_target - len(current_ids)
    if to_add <= 0 or not remaining:
        return list(current_ids)

    rng = random.Random(seed)
    to_add = min(to_add, len(remaining))
    new_items = rng.sample(remaining, to_add)
    return list(current_ids) + new_items


def run_progressive_sampling(
    all_ids: list[str],
    evaluate_fn: Callable[[list[str]], list[float]],
    config: ProgressiveConfig | None = None,
) -> ProgressiveResult:
    """Full progressive sampling loop.

    1. Draw initial sample
    2. Evaluate, compute CI
    3. If CI too wide, expand and repeat
    4. Stop when CI narrow enough, max rounds reached, or pool exhausted
    """
    if config is None:
        config = ProgressiveConfig()

    if not all_ids:
        return ProgressiveResult(
            final_sample_ids=[],
            rounds=[],
            total_rounds=0,
            final_ci_width=0.0,
            converged=True,
        )

    rng = random.Random(config.seed)

    # Initial sample
    initial_size = min(config.initial_size, len(all_ids))
    current_ids = rng.sample(all_ids, initial_size)

    rounds: list[ProgressiveRound] = []
    converged = False

    for round_num in range(1, config.max_rounds + 1):
        scores = evaluate_fn(current_ids)
        ci = compute_ci_width(scores)
        sufficient = is_sufficient(ci, config.ci_width_threshold)

        rounds.append(
            ProgressiveRound(
                round_number=round_num,
                sample_size=len(current_ids),
                ci_width=ci,
                sufficient=sufficient,
            )
        )

        if sufficient:
            converged = True
            break

        # Check if we can expand
        if len(current_ids) >= config.max_size:
            break
        if len(current_ids) >= len(all_ids):
            break

        # Expand - use a deterministic seed derived from round
        expand_seed = (config.seed + round_num) if config.seed is not None else None
        current_ids = expand_sample(
            current_ids,
            all_ids,
            config.expansion_factor,
            config.max_size,
            seed=expand_seed,
        )

    final_ci = rounds[-1].ci_width if rounds else 0.0

    return ProgressiveResult(
        final_sample_ids=current_ids,
        rounds=rounds,
        total_rounds=len(rounds),
        final_ci_width=final_ci,
        converged=converged,
    )


def format_progressive_report(result: ProgressiveResult) -> str:
    """Format a human-readable progressive sampling report."""
    lines: list[str] = []
    lines.append("Progressive Sampling Report")
    lines.append("=" * 40)
    lines.append(f"Total rounds: {result.total_rounds}")
    lines.append(f"Final sample size: {len(result.final_sample_ids)}")
    lines.append(f"Final CI width: {result.final_ci_width:.4f}")
    lines.append(f"Converged: {result.converged}")
    lines.append("")
    lines.append("Rounds:")
    lines.append(f"  {'Round':>5}  {'Size':>6}  {'CI Width':>10}  {'Sufficient':>10}")
    lines.append(f"  {'-' * 5}  {'-' * 6}  {'-' * 10}  {'-' * 10}")
    for r in result.rounds:
        suf = "yes" if r.sufficient else "no"
        lines.append(f"  {r.round_number:>5}  {r.sample_size:>6}  {r.ci_width:>10.4f}  {suf:>10}")
    lines.append("")
    lines.append("Final sample IDs:")
    for item_id in result.final_sample_ids:
        lines.append(f"  - {item_id}")
    return "\n".join(lines)
