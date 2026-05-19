"""Pytest plugin for evalyn: use evalyn metrics as test assertions.

Usage in test files:
    from evalyn_sdk.pytest_plugin import evalyn_assert

    def test_helpfulness(my_agent):
        result = my_agent("What is Python?")
        evalyn_assert(
            input={"query": "What is Python?"},
            output=result,
            metrics=["nonempty_output", "json_valid"],
            threshold=0.5,
        )

Or with the decorator:
    from evalyn_sdk.pytest_plugin import evalyn_test

    @evalyn_test(metrics=["nonempty_output"])
    def test_output(input, output):
        return {"query": "test"}, "non-empty response"
"""

from __future__ import annotations

from typing import Any

from .metrics.factory import build_objective_metric
from .models import DatasetItem, FunctionCall, MetricResult


class EvalynAssertionError(AssertionError):
    """Raised when an evalyn metric assertion fails."""

    def __init__(self, message: str, results: list[MetricResult]):
        super().__init__(message)
        self.results = results


def evalyn_assert(
    input: Any,
    output: Any,
    metrics: list[str],
    threshold: float = 0.5,
    item_id: str = "test-item",
) -> list[MetricResult]:
    """Assert that output passes specified evalyn metrics.

    Args:
        input: The input data (query, prompt, etc.).
        output: The output to evaluate.
        metrics: List of metric IDs to check.
        threshold: Minimum score to pass (for non-binary metrics).
        item_id: Item identifier for results.

    Returns:
        List of MetricResult objects.

    Raises:
        EvalynAssertionError: If any metric fails.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    call = FunctionCall(
        id="test-call",
        function_name="test",
        inputs=input if isinstance(input, dict) else {"input": input},
        output=output,
        error=None,
        started_at=now,
        ended_at=now,
        duration_ms=0,
        session_id=None,
    )
    item = DatasetItem(
        id=item_id,
        input=input if isinstance(input, dict) else {"input": input},
        output=output,
    )

    results = []
    failures = []

    for metric_id in metrics:
        try:
            metric = build_objective_metric(metric_id, {})
            if metric is None:
                failures.append(f"{metric_id}: metric not found")
                continue
            result = metric.evaluate(call, item)
            results.append(result)
            if result.passed is False:
                failures.append(
                    f"{metric_id}: FAIL (score={result.score})"
                )
            elif result.score is not None and result.score < threshold:
                failures.append(
                    f"{metric_id}: below threshold (score={result.score:.3f} < {threshold})"
                )
        except Exception as e:
            failures.append(f"{metric_id}: error ({e})")

    if failures:
        msg = "Evalyn assertion failed:\n" + "\n".join(f"  - {f}" for f in failures)
        raise EvalynAssertionError(msg, results)

    return results


def evalyn_check(
    input: Any,
    output: Any,
    metrics: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Check output against metrics without raising (returns results dict).

    Args:
        input: The input data.
        output: The output to evaluate.
        metrics: List of metric IDs.
        threshold: Minimum score threshold.

    Returns:
        Dict with "passed" (bool), "results" (list), "failures" (list).
    """
    try:
        results = evalyn_assert(input, output, metrics, threshold)
        return {"passed": True, "results": results, "failures": []}
    except EvalynAssertionError as e:
        return {"passed": False, "results": e.results, "failures": [str(e)]}
