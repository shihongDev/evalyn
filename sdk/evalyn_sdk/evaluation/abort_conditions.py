"""Evaluation abort conditions: compound rules for stopping evaluation.

Define complex abort rules like "abort if any safety metric < 50% on
any item" or "abort if overall pass rate drops below 30%".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AbortCondition:
    """A single abort condition."""

    name: str
    description: str
    check_fn: Callable  # (context: Dict) -> bool, returns True to abort

    def check(self, context: Dict[str, Any]) -> bool:
        """Evaluate this condition against current evaluation state.

        Args:
            context: Dict with keys like "pass_rate", "metric_pass_rates",
                "error_count", "items_evaluated", "cost_usd".

        Returns:
            True if evaluation should abort.
        """
        try:
            return self.check_fn(context)
        except Exception:
            return False  # condition errors don't trigger abort


@dataclass
class AbortCheckResult:
    """Result of checking all abort conditions."""

    should_abort: bool
    triggered_conditions: List[str] = field(default_factory=list)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "should_abort": self.should_abort,
            "triggered_conditions": list(self.triggered_conditions),
        }

    def format_text(self) -> str:
        if not self.should_abort:
            return "Abort check: CONTINUE (no conditions triggered)"
        lines = ["Abort check: ABORT"]
        for cond in self.triggered_conditions:
            lines.append(f"  - {cond}")
        return "\n".join(lines)


class AbortController:
    """Manages and evaluates abort conditions during evaluation.

    Usage:
        controller = AbortController()
        controller.add_pass_rate_condition(min_rate=0.3)
        controller.add_safety_condition(safety_metrics=["toxicity_safety"])

        # During evaluation:
        result = controller.check({
            "pass_rate": 0.2,
            "metric_pass_rates": {"toxicity_safety": 0.4},
        })
        if result.should_abort:
            break
    """

    def __init__(self):
        self._conditions: List[AbortCondition] = []

    @property
    def condition_count(self) -> int:
        return len(self._conditions)

    def add_condition(self, condition: AbortCondition) -> None:
        """Add a custom abort condition."""
        self._conditions.append(condition)

    def add_pass_rate_condition(self, min_rate: float = 0.2) -> None:
        """Abort if overall pass rate drops below threshold."""
        self._conditions.append(AbortCondition(
            name=f"min_pass_rate_{min_rate}",
            description=f"Abort if pass rate < {min_rate:.0%}",
            check_fn=lambda ctx: ctx.get("pass_rate", 1.0) < min_rate,
        ))

    def add_safety_condition(
        self,
        safety_metrics: List[str],
        min_rate: float = 0.5,
    ) -> None:
        """Abort if any safety metric drops below threshold."""
        def check(ctx):
            rates = ctx.get("metric_pass_rates", {})
            for mid in safety_metrics:
                rate = rates.get(mid)
                if rate is not None and rate < min_rate:
                    return True
            return False

        self._conditions.append(AbortCondition(
            name=f"safety_gate_{min_rate}",
            description=f"Abort if any safety metric < {min_rate:.0%}",
            check_fn=check,
        ))

    def add_error_rate_condition(self, max_error_rate: float = 0.2) -> None:
        """Abort if error rate exceeds threshold."""
        def check(ctx):
            errors = ctx.get("error_count", 0)
            total = ctx.get("items_evaluated", 0)
            if total == 0:
                return False
            return errors / total > max_error_rate

        self._conditions.append(AbortCondition(
            name=f"max_error_rate_{max_error_rate}",
            description=f"Abort if error rate > {max_error_rate:.0%}",
            check_fn=check,
        ))

    def add_cost_condition(self, max_cost_usd: float) -> None:
        """Abort if cumulative cost exceeds limit."""
        self._conditions.append(AbortCondition(
            name=f"max_cost_{max_cost_usd}",
            description=f"Abort if cost > ${max_cost_usd}",
            check_fn=lambda ctx: ctx.get("cost_usd", 0) > max_cost_usd,
        ))

    def check(self, context: Dict[str, Any]) -> AbortCheckResult:
        """Evaluate all conditions against current state.

        Args:
            context: Current evaluation state dict.

        Returns:
            AbortCheckResult with triggered conditions.
        """
        triggered = []
        for cond in self._conditions:
            if cond.check(context):
                triggered.append(cond.description)

        return AbortCheckResult(
            should_abort=len(triggered) > 0,
            triggered_conditions=triggered,
            context_snapshot=dict(context),
        )
