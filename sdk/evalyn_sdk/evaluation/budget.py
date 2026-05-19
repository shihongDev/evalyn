"""Evaluation budget control: stop if cost or tokens exceed limits.

Provides a BudgetTracker that monitors cumulative token usage and cost
during evaluation, and signals when limits are exceeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BudgetConfig:
    """Configuration for evaluation budget limits.

    Set to None to disable a limit. When a limit is exceeded,
    the tracker signals to stop evaluation gracefully.
    """

    max_cost_usd: float | None = None
    max_tokens: int | None = None
    warning_threshold: float = 0.8  # warn at 80% of limit

    def is_configured(self) -> bool:
        return self.max_cost_usd is not None or self.max_tokens is not None


class BudgetTracker:
    """Tracks cumulative cost and tokens during evaluation.

    Usage:
        tracker = BudgetTracker(BudgetConfig(max_cost_usd=1.0))
        for item in items:
            if tracker.should_stop:
                break
            result = evaluate(item)
            tracker.record(cost=result.cost, tokens=result.tokens)
    """

    def __init__(self, config: BudgetConfig | None = None):
        self.config = config or BudgetConfig()
        self.total_cost_usd: float = 0.0
        self.total_tokens: int = 0
        self.items_evaluated: int = 0
        self._warning_issued: bool = False

    @property
    def should_stop(self) -> bool:
        """Check if budget limits have been exceeded."""
        if not self.config.is_configured():
            return False

        if self.config.max_cost_usd is not None:
            if self.total_cost_usd >= self.config.max_cost_usd:
                return True

        if self.config.max_tokens is not None:
            if self.total_tokens >= self.config.max_tokens:
                return True

        return False

    @property
    def should_warn(self) -> bool:
        """Check if approaching budget limits (at warning threshold)."""
        if not self.config.is_configured():
            return False

        threshold = self.config.warning_threshold

        if self.config.max_cost_usd is not None:
            if self.total_cost_usd >= self.config.max_cost_usd * threshold:
                return True

        if self.config.max_tokens is not None:
            if self.total_tokens >= self.config.max_tokens * threshold:
                return True

        return False

    @property
    def cost_remaining(self) -> float | None:
        """Remaining cost budget, or None if no cost limit."""
        if self.config.max_cost_usd is None:
            return None
        return max(0.0, self.config.max_cost_usd - self.total_cost_usd)

    @property
    def tokens_remaining(self) -> int | None:
        """Remaining token budget, or None if no token limit."""
        if self.config.max_tokens is None:
            return None
        return max(0, self.config.max_tokens - self.total_tokens)

    @property
    def cost_utilization(self) -> float | None:
        """Fraction of cost budget used (0.0-1.0), or None if no limit."""
        if self.config.max_cost_usd is None or self.config.max_cost_usd == 0:
            return None
        return min(1.0, self.total_cost_usd / self.config.max_cost_usd)

    @property
    def token_utilization(self) -> float | None:
        """Fraction of token budget used (0.0-1.0), or None if no limit."""
        if self.config.max_tokens is None or self.config.max_tokens == 0:
            return None
        return min(1.0, self.total_tokens / self.config.max_tokens)

    def record(
        self,
        cost_usd: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record cost and tokens from an evaluation.

        Args:
            cost_usd: Cost of this evaluation in USD.
            input_tokens: Input tokens used.
            output_tokens: Output tokens used.
        """
        self.total_cost_usd += cost_usd
        self.total_tokens += input_tokens + output_tokens
        self.items_evaluated += 1

    def summary(self) -> dict:
        """Return a summary of budget usage."""
        d = {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "items_evaluated": self.items_evaluated,
            "budget_exceeded": self.should_stop,
        }
        if self.config.max_cost_usd is not None:
            d["max_cost_usd"] = self.config.max_cost_usd
            d["cost_utilization"] = round(self.cost_utilization or 0, 4)
        if self.config.max_tokens is not None:
            d["max_tokens"] = self.config.max_tokens
            d["token_utilization"] = round(self.token_utilization or 0, 4)
        return d
