"""Prompt length regularization.

Penalize prompt length during calibration to keep judge prompts concise.
Supports linear, quadratic, and threshold penalty methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class LengthPenalty:
    """Result of computing a length penalty for a single prompt."""

    original_length: int
    penalized_score: float
    penalty: float
    method: str = "linear"

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_length": self.original_length,
            "penalized_score": self.penalized_score,
            "penalty": self.penalty,
            "method": self.method,
        }


@dataclass
class RegularizationConfig:
    """Configuration for length regularization."""

    method: str = "linear"  # "linear", "quadratic", "threshold"
    penalty_weight: float = 0.1
    max_length: int = 2000
    threshold_length: int = 1000

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "penalty_weight": self.penalty_weight,
            "max_length": self.max_length,
            "threshold_length": self.threshold_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegularizationConfig:
        return cls(
            method=data.get("method", "linear"),
            penalty_weight=data.get("penalty_weight", 0.1),
            max_length=data.get("max_length", 2000),
            threshold_length=data.get("threshold_length", 1000),
        )


@dataclass
class RegularizationReport:
    """Summary report across multiple prompts."""

    prompts_evaluated: int
    avg_length: float
    avg_penalty: float
    prompts_over_threshold: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompts_evaluated": self.prompts_evaluated,
            "avg_length": self.avg_length,
            "avg_penalty": self.avg_penalty,
            "prompts_over_threshold": self.prompts_over_threshold,
        }

    def format_text(self) -> str:
        lines = [
            f"Regularization Report ({self.prompts_evaluated} prompts)",
            f"  Avg length:  {self.avg_length:.1f}",
            f"  Avg penalty: {self.avg_penalty:.4f}",
            f"  Over threshold: {self.prompts_over_threshold}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_length_penalty(text: str, config: RegularizationConfig) -> LengthPenalty:
    """Compute length penalty for a single prompt.

    Methods:
    - "linear": penalty = weight * len(text) / max_length, clamped to [0, weight]
    - "quadratic": penalty = weight * (len(text) / max_length) ** 2
    - "threshold": penalty = weight if len(text) > threshold_length, else 0
    """
    length = len(text)
    weight = config.penalty_weight
    max_len = config.max_length

    if config.method == "quadratic":
        if max_len == 0:
            ratio = 0.0
        else:
            ratio = length / max_len
        penalty = weight * ratio * ratio
    elif config.method == "threshold":
        penalty = weight if length > config.threshold_length else 0.0
    else:
        # linear (default)
        if max_len == 0:
            ratio = 0.0
        else:
            ratio = length / max_len
        penalty = weight * ratio
        penalty = max(0.0, min(weight, penalty))

    return LengthPenalty(
        original_length=length,
        penalized_score=0.0,  # placeholder - caller applies to score
        penalty=penalty,
        method=config.method,
    )


def apply_regularization(accuracy: float, text: str, config: RegularizationConfig) -> float:
    """Apply length regularization to an accuracy score.

    Returns accuracy - penalty, clamped to [0, 1].
    """
    result = compute_length_penalty(text, config)
    penalized = accuracy - result.penalty
    return max(0.0, min(1.0, penalized))


def evaluate_prompt_lengths(
    prompts: list[str], config: RegularizationConfig
) -> RegularizationReport:
    """Analyze lengths and penalties across multiple prompts."""
    if not prompts:
        return RegularizationReport(
            prompts_evaluated=0,
            avg_length=0.0,
            avg_penalty=0.0,
            prompts_over_threshold=0,
        )

    total_length = 0
    total_penalty = 0.0
    over_threshold = 0

    for prompt in prompts:
        length = len(prompt)
        total_length += length
        result = compute_length_penalty(prompt, config)
        total_penalty += result.penalty
        if length > config.threshold_length:
            over_threshold += 1

    count = len(prompts)
    return RegularizationReport(
        prompts_evaluated=count,
        avg_length=total_length / count,
        avg_penalty=total_penalty / count,
        prompts_over_threshold=over_threshold,
    )


def suggest_compression(text: str, max_length: int = 1000) -> dict[str, Any]:
    """Analyze a prompt for compression opportunities.

    Returns a dict with current_length, over_budget, excess_chars, suggestion.
    """
    current_length = len(text)
    over_budget = current_length > max_length
    excess_chars = max(0, current_length - max_length)

    if not over_budget:
        suggestion = "Prompt is within budget."
    else:
        suggestion = (
            f"Prompt exceeds budget by {excess_chars} chars. "
            "Consider removing redundant instructions or shortening examples."
        )

    return {
        "current_length": current_length,
        "over_budget": over_budget,
        "excess_chars": excess_chars,
        "suggestion": suggestion,
    }
