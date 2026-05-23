"""Production guards for evaluation runs: cost budgets and quality gates.

This module provides two production-hardening primitives used by the
`evalyn run-eval` CLI:

1. BudgetTracker: cumulative LLM-judge cost tracker with warning and
   hard-stop thresholds (default warning at 80% of the budget). Wraps the
   existing `calculate_cost` helper so pricing logic stays in one place.

2. ThresholdExpression: parses CI/CD-style quality gate expressions like
   `helpfulness_accuracy<0.8` and evaluates them against the mean score of
   the corresponding metric after a run completes.

Both are kept deliberately small and free of CLI/runner imports so they can
be unit-tested in isolation and reused by other CLI commands in the future.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Cost budget tracking
# ---------------------------------------------------------------------------


class BudgetExceededError(RuntimeError):
    """Raised when cumulative LLM-judge cost exceeds the configured budget.

    The runner catches this and treats it like a clean interrupt: partial
    results are persisted and the CLI exits with code 2. Strategies attach
    in-memory partial results via `partial_results` so the runner can build
    a final EvalRun even when checkpointing is disabled.
    """

    def __init__(
        self,
        current_cost: float,
        max_cost: float,
        partial_results: list | None = None,
    ):
        self.current_cost = current_cost
        self.max_cost = max_cost
        self.partial_results = partial_results or []
        super().__init__(
            f"Cost budget exceeded: ${current_cost:.4f} >= ${max_cost:.4f}"
        )


@dataclass
class BudgetStatus:
    """Snapshot of budget state after recording a result."""

    cumulative_cost: float
    fraction_used: float
    warning_triggered: bool
    exceeded: bool


class BudgetTracker:
    """Track cumulative LLM-judge cost against a USD budget.

    The tracker is intentionally permissive about input shape: it accepts
    raw `(model, input_tokens, output_tokens)` tuples or MetricResult-like
    objects via `record_result`. Each call returns a BudgetStatus snapshot
    so callers can react (print warning, halt, etc.).

    A None max_cost disables tracking entirely - all `record_*` calls
    become no-ops that still return a status with `exceeded=False`.
    """

    def __init__(self, max_cost: float | None, warning_fraction: float = 0.8):
        if max_cost is not None and max_cost <= 0:
            raise ValueError(f"max_cost must be positive, got {max_cost}")
        if not 0 < warning_fraction <= 1:
            raise ValueError(
                f"warning_fraction must be in (0, 1], got {warning_fraction}"
            )
        self.max_cost = max_cost
        self.warning_fraction = warning_fraction
        self.cumulative_cost = 0.0
        self._warning_emitted = False

    @property
    def enabled(self) -> bool:
        return self.max_cost is not None

    def _status(self, warning_triggered: bool) -> BudgetStatus:
        if self.max_cost is None:
            return BudgetStatus(
                cumulative_cost=self.cumulative_cost,
                fraction_used=0.0,
                warning_triggered=False,
                exceeded=False,
            )
        fraction = self.cumulative_cost / self.max_cost
        return BudgetStatus(
            cumulative_cost=self.cumulative_cost,
            fraction_used=fraction,
            warning_triggered=warning_triggered,
            exceeded=self.cumulative_cost >= self.max_cost,
        )

    def record_cost(self, cost: float) -> BudgetStatus:
        """Add raw cost in USD and return current status.

        Crossing the warning threshold returns warning_triggered=True
        exactly once across the lifetime of the tracker.
        """
        if not self.enabled:
            return self._status(False)
        if cost < 0:
            cost = 0.0
        self.cumulative_cost += cost
        crossed_warning = False
        if (
            not self._warning_emitted
            and self.cumulative_cost >= self.max_cost * self.warning_fraction
        ):
            self._warning_emitted = True
            crossed_warning = True
        return self._status(crossed_warning)

    def record_result(self, result: Any) -> BudgetStatus:
        """Record cost derived from a MetricResult-like object.

        The object must expose `model`, `input_tokens`, `output_tokens`.
        Results without a model (objective metrics) contribute zero cost.
        """
        model = getattr(result, "model", None)
        if not model:
            return self._status(False)
        input_tok = getattr(result, "input_tokens", 0) or 0
        output_tok = getattr(result, "output_tokens", 0) or 0
        # Local import keeps this module dependency-free at import time.
        from ..trace.instrumentation.providers._shared import calculate_cost

        cost = calculate_cost(model, input_tok, output_tok)
        return self.record_cost(cost)


# ---------------------------------------------------------------------------
# Quality gate / threshold expressions
# ---------------------------------------------------------------------------


# Order longer operators first so the regex prefers `<=`/`>=` over `<`/`>`.
_VALID_OPS = ("<=", ">=", "==", "!=", "<", ">")
_THRESHOLD_RE = re.compile(
    r"""
    ^
    \s*
    (?P<metric>[A-Za-z_][A-Za-z0-9_.\-]*)   # metric id
    \s*
    (?P<op><=|>=|==|!=|<|>)                 # comparison
    \s*
    (?P<value>-?\d+(\.\d+)?)                # float / int
    \s*
    $
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class ThresholdExpression:
    """A parsed `--fail-on` expression like `helpfulness_accuracy<0.8`.

    The expression states the *failure condition* (CI/CD convention):
    `metric<0.8` means "fail the gate when the metric's mean score is
    less than 0.8". `is_breached` returns True exactly when the inequality
    holds.
    """

    metric_id: str
    op: str
    threshold: float
    raw: str

    def evaluate(self, score: float) -> bool:
        """Return True when the failure condition holds (gate is breached)."""
        if self.op == "<":
            return score < self.threshold
        if self.op == "<=":
            return score <= self.threshold
        if self.op == ">":
            return score > self.threshold
        if self.op == ">=":
            return score >= self.threshold
        if self.op == "==":
            return score == self.threshold
        if self.op == "!=":
            return score != self.threshold
        raise ValueError(f"Unknown operator: {self.op}")

    def is_breached(self, score: float) -> bool:
        """Return True when the gate fails for this score (alias of evaluate)."""
        return self.evaluate(score)


def parse_threshold_expression(expr: str) -> ThresholdExpression:
    """Parse a `--fail-on` expression. Raises ValueError on bad input."""
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("Threshold expression cannot be empty")
    match = _THRESHOLD_RE.match(expr)
    if not match:
        raise ValueError(
            f"Invalid threshold expression: {expr!r}. "
            f"Expected '<metric>{{{'|'.join(_VALID_OPS)}}}<number>' "
            f"(e.g. 'helpfulness_accuracy<0.8')"
        )
    return ThresholdExpression(
        metric_id=match.group("metric"),
        op=match.group("op"),
        threshold=float(match.group("value")),
        raw=expr.strip(),
    )


@dataclass(frozen=True)
class ThresholdBreach:
    """One metric that failed a threshold check."""

    expression: ThresholdExpression
    actual_score: float

    def format(self) -> str:
        return (
            f"FAIL: {self.expression.metric_id} mean={self.actual_score:.4f} "
            f"violates {self.expression.raw}"
        )


@dataclass(frozen=True)
class MissingMetricWarning:
    """A `--fail-on` referenced a metric not present in the run."""

    expression: ThresholdExpression

    def format(self) -> str:
        return (
            f"WARNING: --fail-on {self.expression.raw} skipped: "
            f"metric '{self.expression.metric_id}' not in run results"
        )


@dataclass
class ThresholdReport:
    """Aggregated result of evaluating all thresholds against a run."""

    breaches: list[ThresholdBreach]
    missing: list[MissingMetricWarning]

    @property
    def any_breached(self) -> bool:
        return bool(self.breaches)


def _extract_metric_means(run_or_summary: Any) -> dict[str, float]:
    """Extract `{metric_id: mean_score}` from an EvalRun or its summary dict.

    Accepts:
    - An EvalRun-like object with a `summary` dict
    - A raw summary dict shaped like `{"metrics": {id: {"avg_score": ...}}}`
    """
    summary = getattr(run_or_summary, "summary", run_or_summary)
    if not isinstance(summary, dict):
        return {}
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    means: dict[str, float] = {}
    for metric_id, stats in metrics.items():
        if not isinstance(stats, dict):
            continue
        avg = stats.get("avg_score")
        if isinstance(avg, (int, float)):
            means[metric_id] = float(avg)
    return means


def evaluate_thresholds(
    expressions: list[ThresholdExpression],
    run_or_summary: Any,
) -> ThresholdReport:
    """Evaluate all threshold expressions against a finished run.

    Metrics that are missing from the run (e.g. typo, or metric failed
    to load) yield a MissingMetricWarning rather than a breach.
    """
    means = _extract_metric_means(run_or_summary)
    breaches: list[ThresholdBreach] = []
    missing: list[MissingMetricWarning] = []
    for expr in expressions:
        if expr.metric_id not in means:
            missing.append(MissingMetricWarning(expression=expr))
            continue
        score = means[expr.metric_id]
        if expr.is_breached(score):
            breaches.append(ThresholdBreach(expression=expr, actual_score=score))
    return ThresholdReport(breaches=breaches, missing=missing)


__all__ = [
    "BudgetExceededError",
    "BudgetStatus",
    "BudgetTracker",
    "MissingMetricWarning",
    "ThresholdBreach",
    "ThresholdExpression",
    "ThresholdReport",
    "evaluate_thresholds",
    "parse_threshold_expression",
]
