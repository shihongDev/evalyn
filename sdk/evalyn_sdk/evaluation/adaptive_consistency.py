"""Adaptive consistency sampling: stop early when judge agreement is clear.

Runs a sample function multiple times per item, stopping as soon as the
agreement rate among collected decisions exceeds a configurable threshold.
This saves budget when judges agree quickly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive consistency sampling."""

    max_samples: int = 5
    min_samples: int = 3
    agreement_threshold: float = 1.0  # fraction needed to stop early

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_samples": self.max_samples,
            "min_samples": self.min_samples,
            "agreement_threshold": self.agreement_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdaptiveConfig:
        return cls(
            max_samples=data.get("max_samples", 5),
            min_samples=data.get("min_samples", 3),
            agreement_threshold=data.get("agreement_threshold", 1.0),
        )


@dataclass
class AdaptiveSampleResult:
    """Result for a single item after adaptive sampling."""

    item_id: str
    samples_taken: int
    samples_max: int
    early_stopped: bool
    agreement_rate: float
    final_decision: bool  # passed or not
    savings_pct: float  # percentage of samples saved

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "samples_taken": self.samples_taken,
            "samples_max": self.samples_max,
            "early_stopped": self.early_stopped,
            "agreement_rate": round(self.agreement_rate, 4),
            "final_decision": self.final_decision,
            "savings_pct": round(self.savings_pct, 2),
        }


@dataclass
class AdaptiveReport:
    """Aggregate report across all adaptively-sampled items."""

    results: List[AdaptiveSampleResult] = field(default_factory=list)
    total_samples_taken: int = 0
    total_samples_possible: int = 0
    overall_savings_pct: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "total_samples_taken": self.total_samples_taken,
            "total_samples_possible": self.total_samples_possible,
            "overall_savings_pct": round(self.overall_savings_pct, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdaptiveReport:
        results = [
            AdaptiveSampleResult(**r) for r in data.get("results", [])
        ]
        return cls(
            results=results,
            total_samples_taken=data.get("total_samples_taken", 0),
            total_samples_possible=data.get("total_samples_possible", 0),
            overall_savings_pct=data.get("overall_savings_pct", 0.0),
        )

    def format_text(self) -> str:
        lines = [
            "Adaptive Consistency Report",
            f"  Items:             {len(self.results)}",
            f"  Samples taken:     {self.total_samples_taken}",
            f"  Samples possible:  {self.total_samples_possible}",
            f"  Overall savings:   {self.overall_savings_pct:.1f}%",
        ]
        early = sum(1 for r in self.results if r.early_stopped)
        lines.append(f"  Early stopped:     {early}/{len(self.results)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_agreement_rate(decisions: List[bool]) -> float:
    """Fraction of the majority class among decisions.

    Example: [True, True, False] -> 2/3 = 0.667.
    Returns 0.0 for an empty list.
    """
    if not decisions:
        return 0.0
    true_count = sum(decisions)
    majority = max(true_count, len(decisions) - true_count)
    return majority / len(decisions)


def should_stop_early(decisions: List[bool], config: AdaptiveConfig) -> bool:
    """Check whether enough agreement exists to stop sampling.

    Returns True when len(decisions) >= min_samples and the agreement
    rate meets or exceeds the threshold.
    """
    if len(decisions) < config.min_samples:
        return False
    return compute_agreement_rate(decisions) >= config.agreement_threshold


def run_adaptive_sampling(
    item_id: str,
    sample_fn: Callable[[], bool],
    config: AdaptiveConfig,
) -> AdaptiveSampleResult:
    """Run sample_fn up to max_samples times, stopping early on agreement.

    Args:
        item_id: Identifier for the item being sampled.
        sample_fn: Callable that returns a boolean decision each call.
        config: Adaptive sampling configuration.

    Returns:
        AdaptiveSampleResult with savings info.
    """
    decisions: List[bool] = []
    early_stopped = False

    for _ in range(config.max_samples):
        decisions.append(sample_fn())
        if should_stop_early(decisions, config):
            early_stopped = len(decisions) < config.max_samples
            break

    agreement = compute_agreement_rate(decisions)
    true_count = sum(decisions)
    final_decision = true_count > len(decisions) / 2
    saved = config.max_samples - len(decisions)
    savings_pct = (saved / config.max_samples * 100) if config.max_samples > 0 else 0.0

    return AdaptiveSampleResult(
        item_id=item_id,
        samples_taken=len(decisions),
        samples_max=config.max_samples,
        early_stopped=early_stopped,
        agreement_rate=agreement,
        final_decision=final_decision,
        savings_pct=savings_pct,
    )


def compute_adaptive_report(
    results: List[AdaptiveSampleResult],
) -> AdaptiveReport:
    """Aggregate multiple adaptive sample results into a report."""
    total_taken = sum(r.samples_taken for r in results)
    total_possible = sum(r.samples_max for r in results)
    savings = (
        (total_possible - total_taken) / total_possible * 100
        if total_possible > 0
        else 0.0
    )
    return AdaptiveReport(
        results=results,
        total_samples_taken=total_taken,
        total_samples_possible=total_possible,
        overall_savings_pct=savings,
    )
