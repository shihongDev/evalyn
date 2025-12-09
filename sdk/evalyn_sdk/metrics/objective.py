from __future__ import annotations

from typing import Any, Optional

from .registry import Metric
from ..models import DatasetItem, FunctionCall, MetricResult, MetricSpec


def latency_metric(metric_id: str = "latency_ms") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Latency (ms)",
        type="objective",
        description="Execution time in milliseconds.",
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        return MetricResult(
            metric_id=spec.id,
            item_id=item.id,
            call_id=call.id,
            score=call.duration_ms,
            passed=None,
            details={"duration_ms": call.duration_ms},
        )

    return Metric(spec, handler)


def exact_match_metric(
    metric_id: str = "exact_match",
    expected_field: str = "expected",
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Exact Match",
        type="objective",
        description="Checks if the output matches the expected value exactly.",
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        expected = getattr(item, expected_field, None)
        actual = call.output
        passed = actual == expected
        score = 1.0 if passed else 0.0
        return MetricResult(
            metric_id=spec.id,
            item_id=item.id,
            call_id=call.id,
            score=score,
            passed=passed,
            details={"expected": expected, "actual": actual},
        )

    return Metric(spec, handler)


def substring_metric(
    metric_id: str = "substring",
    needle: Optional[str] = None,
    expected_field: str = "expected_substring",
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Substring",
        type="objective",
        description="Checks whether the output contains a target substring.",
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        target = needle or item.metadata.get(expected_field) or ""
        output_text = call.output or ""
        passed = target in output_text
        score = 1.0 if passed else 0.0
        return MetricResult(
            metric_id=spec.id,
            item_id=item.id,
            call_id=call.id,
            score=score,
            passed=passed,
            details={"target": target, "output_excerpt": str(output_text)[:200]},
        )

    return Metric(spec, handler)


def cost_metric(metric_id: str = "cost") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Token/Cost",
        type="objective",
        description="Records cost metadata if present on the call.",
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        cost_info: Any = call.metadata.get("cost")
        return MetricResult(
            metric_id=spec.id,
            item_id=item.id,
            call_id=call.id,
            score=cost_info.get("total") if isinstance(cost_info, dict) else None,
            passed=None,
            details={"cost": cost_info},
        )

    return Metric(spec, handler)


def register_builtin_metrics(registry) -> None:
    registry.register(latency_metric())
    registry.register(exact_match_metric())
    registry.register(cost_metric())
