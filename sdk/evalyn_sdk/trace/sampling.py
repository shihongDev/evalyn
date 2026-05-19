from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..models import Span


@dataclass
class SamplingConfig:
    """Configuration for trace sampling."""

    rate: float  # 0.0 to 1.0
    always_capture_errors: bool = True
    always_capture_slow: bool = True
    slow_threshold_ms: float = 5000.0
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rate": self.rate,
            "always_capture_errors": self.always_capture_errors,
            "always_capture_slow": self.always_capture_slow,
            "slow_threshold_ms": self.slow_threshold_ms,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplingConfig:
        return cls(
            rate=data["rate"],
            always_capture_errors=data.get("always_capture_errors", True),
            always_capture_slow=data.get("always_capture_slow", True),
            slow_threshold_ms=data.get("slow_threshold_ms", 5000.0),
            seed=data.get("seed"),
        )


@dataclass
class SamplingDecision:
    """Result of a sampling decision."""

    should_capture: bool
    reason: str  # "sampled", "rate_excluded", "error_priority", "slow_priority"

    def as_dict(self) -> dict[str, Any]:
        return {
            "should_capture": self.should_capture,
            "reason": self.reason,
        }


def should_sample(
    config: SamplingConfig, span: Span | None = None
) -> SamplingDecision:
    """Determine if a trace should be captured.

    Priority order:
    1. Error spans (if always_capture_errors is True)
    2. Slow spans (if always_capture_slow is True)
    3. Rate-based sampling
    """
    # Check error priority
    if span is not None and config.always_capture_errors:
        if span.status == "error":
            return SamplingDecision(should_capture=True, reason="error_priority")

    # Check slow priority
    if span is not None and config.always_capture_slow:
        duration = span.duration_ms
        if duration is not None and duration >= config.slow_threshold_ms:
            return SamplingDecision(should_capture=True, reason="slow_priority")

    # Rate-based sampling (stateless - use create_sampler for seeded determinism)
    if random.random() < config.rate:
        return SamplingDecision(should_capture=True, reason="sampled")

    return SamplingDecision(should_capture=False, reason="rate_excluded")


def create_sampler(
    config: SamplingConfig,
) -> Callable[[Span | None], SamplingDecision]:
    """Return a callable that applies sampling decisions.

    Preserves RNG state across calls for consistent behavior.
    """
    rng = random.Random(config.seed)

    def sampler(span: Span | None = None) -> SamplingDecision:
        # Check error priority
        if span is not None and config.always_capture_errors:
            if span.status == "error":
                return SamplingDecision(
                    should_capture=True, reason="error_priority"
                )

        # Check slow priority
        if span is not None and config.always_capture_slow:
            duration = span.duration_ms
            if duration is not None and duration >= config.slow_threshold_ms:
                return SamplingDecision(
                    should_capture=True, reason="slow_priority"
                )

        # Rate-based sampling (uses persistent RNG)
        if rng.random() < config.rate:
            return SamplingDecision(should_capture=True, reason="sampled")

        return SamplingDecision(should_capture=False, reason="rate_excluded")

    return sampler
