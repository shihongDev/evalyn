"""
Simulation evaluation loop: generate, evaluate, and iterate on simulated data.

Orchestrates a generate-evaluate loop where each round feeds failure patterns
from the previous round back into the generator. Pure Python - no external
deps, no LLM calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class LoopConfig:
    """Configuration for the evaluation loop."""

    max_rounds: int = 3
    items_per_round: int = 10
    convergence_threshold: float = 0.02
    target_pass_rate: float = 0.9

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "items_per_round": self.items_per_round,
            "convergence_threshold": self.convergence_threshold,
            "target_pass_rate": self.target_pass_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopConfig:
        return cls(
            max_rounds=data.get("max_rounds", 3),
            items_per_round=data.get("items_per_round", 10),
            convergence_threshold=data.get("convergence_threshold", 0.02),
            target_pass_rate=data.get("target_pass_rate", 0.9),
        )


@dataclass
class RoundResult:
    """Result of a single evaluation round."""

    round_number: int
    items_generated: int
    items_evaluated: int
    pass_rate: float
    failure_patterns: list[str] = field(default_factory=list)
    improved_from_previous: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "items_generated": self.items_generated,
            "items_evaluated": self.items_evaluated,
            "pass_rate": self.pass_rate,
            "failure_patterns": self.failure_patterns,
            "improved_from_previous": self.improved_from_previous,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoundResult:
        return cls(
            round_number=data["round_number"],
            items_generated=data["items_generated"],
            items_evaluated=data["items_evaluated"],
            pass_rate=data["pass_rate"],
            failure_patterns=data.get("failure_patterns", []),
            improved_from_previous=data.get("improved_from_previous", False),
        )


@dataclass
class LoopResult:
    """Aggregate result of the full evaluation loop."""

    rounds: list[RoundResult] = field(default_factory=list)
    final_pass_rate: float = 0.0
    converged: bool = False
    total_items: int = 0
    total_rounds: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rounds": [r.as_dict() for r in self.rounds],
            "final_pass_rate": self.final_pass_rate,
            "converged": self.converged,
            "total_items": self.total_items,
            "total_rounds": self.total_rounds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopResult:
        return cls(
            rounds=[RoundResult.from_dict(r) for r in data.get("rounds", [])],
            final_pass_rate=data.get("final_pass_rate", 0.0),
            converged=data.get("converged", False),
            total_items=data.get("total_items", 0),
            total_rounds=data.get("total_rounds", 0),
        )


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def check_convergence(rounds: list[RoundResult], threshold: float = 0.02) -> bool:
    """True if last 2 rounds have pass rate within threshold of each other."""
    if len(rounds) < 2:
        return False
    return abs(rounds[-1].pass_rate - rounds[-2].pass_rate) <= threshold


def extract_round_failure_patterns(
    results: list[dict[str, Any]],
    score_field: str = "score",
    threshold: float = 0.5,
) -> list[str]:
    """Extract failure descriptions from low-scoring items.

    Uses the "input" field truncated to 50 chars as the pattern identifier.
    """
    patterns: list[str] = []
    for item in results:
        score = item.get(score_field, 0.0)
        if score < threshold:
            raw_input = str(item.get("input", ""))
            truncated = raw_input[:50]
            patterns.append(truncated)
    return patterns


def should_stop(rounds: list[RoundResult], config: LoopConfig) -> bool:
    """Check whether the loop should stop.

    Stops if any of:
    - converged (last 2 rounds within convergence_threshold)
    - max rounds reached
    - target pass rate achieved
    """
    if len(rounds) >= config.max_rounds:
        return True
    if rounds and rounds[-1].pass_rate >= config.target_pass_rate:
        return True
    if check_convergence(rounds, config.convergence_threshold):
        return True
    return False


def run_eval_loop(
    generate_fn: Callable[[int, list[str]], list[dict[str, Any]]],
    evaluate_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    config: LoopConfig | None = None,
) -> LoopResult:
    """Run the generate-evaluate loop.

    Args:
        generate_fn: Callable(items_per_round, failure_patterns) -> list of dicts.
        evaluate_fn: Callable(items) -> list of dicts with at least a "score" field.
        config: Loop configuration. Uses defaults if None.

    Returns:
        LoopResult with per-round breakdown.
    """
    if config is None:
        config = LoopConfig()

    rounds: list[RoundResult] = []
    failure_patterns: list[str] = []

    for round_num in range(1, config.max_rounds + 1):
        items = generate_fn(config.items_per_round, failure_patterns)
        evaluated = evaluate_fn(items)

        items_generated = len(items)
        items_evaluated = len(evaluated)

        passing = sum(1 for e in evaluated if e.get("score", 0.0) >= 0.5)
        pass_rate = passing / items_evaluated if items_evaluated > 0 else 0.0

        failure_patterns = extract_round_failure_patterns(evaluated)

        improved = False
        if rounds:
            improved = pass_rate > rounds[-1].pass_rate

        rr = RoundResult(
            round_number=round_num,
            items_generated=items_generated,
            items_evaluated=items_evaluated,
            pass_rate=pass_rate,
            failure_patterns=failure_patterns,
            improved_from_previous=improved,
        )
        rounds.append(rr)

        if should_stop(rounds, config):
            break

    total_items = sum(r.items_evaluated for r in rounds)
    final_pass_rate = rounds[-1].pass_rate if rounds else 0.0
    converged = check_convergence(rounds, config.convergence_threshold)

    return LoopResult(
        rounds=rounds,
        final_pass_rate=final_pass_rate,
        converged=converged,
        total_items=total_items,
        total_rounds=len(rounds),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_loop_report(result: LoopResult) -> str:
    """Human-readable report with per-round breakdown."""
    lines: list[str] = []
    lines.append("Evaluation Loop Report")
    lines.append("=" * 40)
    lines.append(f"Total rounds: {result.total_rounds}")
    lines.append(f"Total items: {result.total_items}")
    lines.append(f"Final pass rate: {result.final_pass_rate:.2%}")
    lines.append(f"Converged: {result.converged}")
    lines.append("")
    lines.append("Per-round breakdown:")
    lines.append("-" * 40)
    for r in result.rounds:
        improved_marker = " [improved]" if r.improved_from_previous else ""
        lines.append(
            f"  Round {r.round_number}: "
            f"pass_rate={r.pass_rate:.2%}, "
            f"generated={r.items_generated}, "
            f"evaluated={r.items_evaluated}, "
            f"failures={len(r.failure_patterns)}"
            f"{improved_marker}"
        )
    return "\n".join(lines)


def compute_improvement_rate(rounds: list[RoundResult]) -> float:
    """Compute (last pass_rate - first pass_rate) / number of rounds."""
    if len(rounds) < 2:
        return 0.0
    return (rounds[-1].pass_rate - rounds[0].pass_rate) / len(rounds)
