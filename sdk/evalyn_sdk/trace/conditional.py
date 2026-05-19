from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import Callable


@dataclass
class TracingCondition:
    """A condition that decides whether tracing should occur."""

    name: str
    condition_type: str  # "sample_rate", "predicate", "environment"
    config: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "condition_type": self.condition_type,
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TracingCondition:
        return cls(
            name=data["name"],
            condition_type=data["condition_type"],
            config=data.get("config", {}),
        )


@dataclass
class TracingDecision:
    """Result of evaluating a tracing condition."""

    should_trace: bool
    reason: str
    condition_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "should_trace": self.should_trace,
            "reason": self.reason,
            "condition_name": self.condition_name,
        }


def sample_rate_condition(
    rate: float, seed: int | None = None
) -> TracingCondition:
    """Create a rate-based sampling condition."""
    config: dict[str, Any] = {"rate": rate}
    if seed is not None:
        config["seed"] = seed
    return TracingCondition(
        name=f"sample_{rate}",
        condition_type="sample_rate",
        config=config,
    )


def environment_condition(allowed_envs: list[str]) -> TracingCondition:
    """Only trace when EVALYN_ENV is in the allowed list."""
    return TracingCondition(
        name="env_filter",
        condition_type="environment",
        config={"allowed_envs": list(allowed_envs)},
    )


def predicate_condition(
    name: str, predicate: Callable[..., bool]
) -> tuple[TracingCondition, Callable[..., bool]]:
    """Create a predicate-based condition.

    Returns the condition dataclass and the wrapped predicate callable.
    The predicate is stored outside the dataclass since callables are not
    serializable, but is returned so the caller can pass it in context
    when evaluating.
    """
    condition = TracingCondition(
        name=name,
        condition_type="predicate",
        config={},
    )
    return condition, predicate


def evaluate_condition(
    condition: TracingCondition,
    context: dict[str, Any] | None = None,
) -> TracingDecision:
    """Evaluate a single tracing condition."""
    ctx = context or {}

    if condition.condition_type == "sample_rate":
        rate = condition.config.get("rate", 1.0)
        seed = condition.config.get("seed")
        rng = random.Random(seed)
        if rng.random() < rate:
            return TracingDecision(
                should_trace=True,
                reason="sampled",
                condition_name=condition.name,
            )
        return TracingDecision(
            should_trace=False,
            reason="rate_excluded",
            condition_name=condition.name,
        )

    if condition.condition_type == "environment":
        allowed = condition.config.get("allowed_envs", [])
        current = os.environ.get("EVALYN_ENV", "development")
        if current in allowed:
            return TracingDecision(
                should_trace=True,
                reason=f"environment '{current}' allowed",
                condition_name=condition.name,
            )
        return TracingDecision(
            should_trace=False,
            reason=f"environment '{current}' not in allowed list",
            condition_name=condition.name,
        )

    if condition.condition_type == "predicate":
        predicate = ctx.get("predicate")
        if predicate is not None and callable(predicate):
            predicate_args = ctx.get("predicate_args", ())
            predicate_kwargs = ctx.get("predicate_kwargs", {})
            result = predicate(*predicate_args, **predicate_kwargs)
            if result:
                return TracingDecision(
                    should_trace=True,
                    reason="predicate passed",
                    condition_name=condition.name,
                )
            return TracingDecision(
                should_trace=False,
                reason="predicate failed",
                condition_name=condition.name,
            )
        return TracingDecision(
            should_trace=False,
            reason="no predicate provided in context",
            condition_name=condition.name,
        )

    return TracingDecision(
        should_trace=False,
        reason=f"unknown condition type: {condition.condition_type}",
        condition_name=condition.name,
    )


def evaluate_conditions(
    conditions: list[TracingCondition],
    mode: str = "all",
    context: dict[str, Any] | None = None,
) -> TracingDecision:
    """Evaluate multiple conditions.

    mode="all": all conditions must pass (AND logic).
    mode="any": at least one condition must pass (OR logic).
    """
    if not conditions:
        return TracingDecision(
            should_trace=True,
            reason="no conditions to evaluate",
        )

    decisions = [evaluate_condition(c, context) for c in conditions]

    if mode == "all":
        for d in decisions:
            if not d.should_trace:
                return TracingDecision(
                    should_trace=False,
                    reason=f"condition '{d.condition_name}' failed: {d.reason}",
                    condition_name=d.condition_name,
                )
        return TracingDecision(
            should_trace=True,
            reason="all conditions passed",
            condition_name=decisions[-1].condition_name,
        )

    if mode == "any":
        for d in decisions:
            if d.should_trace:
                return TracingDecision(
                    should_trace=True,
                    reason=f"condition '{d.condition_name}' passed: {d.reason}",
                    condition_name=d.condition_name,
                )
        return TracingDecision(
            should_trace=False,
            reason="no conditions passed",
            condition_name=decisions[-1].condition_name,
        )

    return TracingDecision(
        should_trace=False,
        reason=f"unknown mode: {mode}",
    )
