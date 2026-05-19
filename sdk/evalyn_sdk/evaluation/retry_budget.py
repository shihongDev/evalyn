"""Evaluation retry budget: bound total retries across all items.

Prevents retry storms from consuming excessive tokens when provider
errors are widespread. Tracks retries per-metric and per-item.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetryBudgetConfig:
    """Configuration for retry budget limits."""

    max_retries_total: Optional[int] = None  # None = unlimited
    max_retries_per_item: int = 3
    max_retries_per_metric: Optional[int] = None  # None = unlimited

    def is_configured(self) -> bool:
        return self.max_retries_total is not None


class RetryBudget:
    """Tracks retry usage and enforces budget limits.

    Usage:
        budget = RetryBudget(RetryBudgetConfig(max_retries_total=50))

        if budget.can_retry("item-1", "helpfulness"):
            budget.record_retry("item-1", "helpfulness")
            # ... retry the evaluation
    """

    def __init__(self, config: Optional[RetryBudgetConfig] = None):
        self.config = config or RetryBudgetConfig()
        self._total_retries: int = 0
        self._per_item: Dict[str, int] = {}
        self._per_metric: Dict[str, int] = {}

    @property
    def total_retries(self) -> int:
        return self._total_retries

    @property
    def budget_exhausted(self) -> bool:
        """Check if the total retry budget is exhausted."""
        if self.config.max_retries_total is None:
            return False
        return self._total_retries >= self.config.max_retries_total

    @property
    def retries_remaining(self) -> Optional[int]:
        """Remaining retry budget, or None if unlimited."""
        if self.config.max_retries_total is None:
            return None
        return max(0, self.config.max_retries_total - self._total_retries)

    def can_retry(self, item_id: str, metric_id: str) -> bool:
        """Check if a retry is allowed for this item-metric pair.

        Args:
            item_id: The item being evaluated.
            metric_id: The metric that failed.

        Returns:
            True if retry is within budget.
        """
        # Check total budget
        if self.config.max_retries_total is not None:
            if self._total_retries >= self.config.max_retries_total:
                return False

        # Check per-item limit
        item_retries = self._per_item.get(item_id, 0)
        if item_retries >= self.config.max_retries_per_item:
            return False

        # Check per-metric limit
        if self.config.max_retries_per_metric is not None:
            metric_retries = self._per_metric.get(metric_id, 0)
            if metric_retries >= self.config.max_retries_per_metric:
                return False

        return True

    def record_retry(self, item_id: str, metric_id: str) -> None:
        """Record a retry attempt.

        Args:
            item_id: The item being retried.
            metric_id: The metric being retried.
        """
        self._total_retries += 1
        self._per_item[item_id] = self._per_item.get(item_id, 0) + 1
        self._per_metric[metric_id] = self._per_metric.get(metric_id, 0) + 1

    def summary(self) -> Dict[str, Any]:
        """Return retry budget usage summary."""
        d: Dict[str, Any] = {
            "total_retries": self._total_retries,
            "budget_exhausted": self.budget_exhausted,
            "items_retried": len(self._per_item),
            "metrics_retried": len(self._per_metric),
        }
        if self.config.max_retries_total is not None:
            d["max_retries_total"] = self.config.max_retries_total
            d["retries_remaining"] = self.retries_remaining
        # Top retry consumers
        if self._per_item:
            top_items = sorted(self._per_item.items(), key=lambda x: x[1], reverse=True)[:5]
            d["top_retried_items"] = [{"item_id": k, "retries": v} for k, v in top_items]
        if self._per_metric:
            top_metrics = sorted(self._per_metric.items(), key=lambda x: x[1], reverse=True)[:5]
            d["top_retried_metrics"] = [{"metric_id": k, "retries": v} for k, v in top_metrics]
        return d
