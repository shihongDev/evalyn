"""Metric dependency resolution.

Supports declaring that one metric depends on another (e.g., "only run
helpfulness if json_valid passes"). Provides topological sorting and
conditional execution.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class MetricDependency:
    """A dependency between two metrics.

    Means: only run `metric_id` if `depends_on` passes (or meets condition).
    """

    metric_id: str
    depends_on: str
    condition: str = "passed"  # "passed" or "score_above:0.5"

    def check(self, dependency_result) -> bool:
        """Check if the dependency condition is met.

        Args:
            dependency_result: MetricResult from the dependency metric.

        Returns:
            True if condition is met and the dependent metric should run.
        """
        if dependency_result is None:
            return False

        if self.condition == "passed":
            return dependency_result.passed is True

        if self.condition.startswith("score_above:"):
            try:
                threshold = float(self.condition.split(":")[1])
                return (dependency_result.score or 0.0) >= threshold
            except (ValueError, IndexError):
                return False

        return True  # unknown condition = run anyway


def topological_sort(
    metrics: list,
    dependencies: list[MetricDependency],
) -> tuple[list, dict[str, list[str]]]:
    """Sort metrics in dependency order.

    Metrics with no dependencies come first. Metrics that depend on others
    come after their dependencies.

    Args:
        metrics: List of Metric objects.
        dependencies: List of MetricDependency declarations.

    Returns:
        Tuple of (sorted metrics, dependency map: metric_id -> [depends_on_ids])

    Raises:
        ValueError: If a cycle is detected in dependencies.
    """
    metric_map = {m.spec.id: m for m in metrics}
    dep_map: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {m.spec.id: 0 for m in metrics}

    for dep in dependencies:
        if dep.metric_id in metric_map and dep.depends_on in metric_map:
            dep_map[dep.metric_id].append(dep.depends_on)
            in_degree[dep.metric_id] = in_degree.get(dep.metric_id, 0) + 1

    # Kahn's algorithm for topological sort
    queue = deque([mid for mid, deg in in_degree.items() if deg == 0])
    sorted_ids = []

    while queue:
        mid = queue.popleft()
        sorted_ids.append(mid)

        # For each metric that depends on mid, reduce in-degree
        for other_mid, deps in dep_map.items():
            if mid in deps:
                in_degree[other_mid] -= 1
                if in_degree[other_mid] == 0:
                    queue.append(other_mid)

    if len(sorted_ids) != len(metrics):
        # Cycle detected
        missing = set(m.spec.id for m in metrics) - set(sorted_ids)
        raise ValueError(f"Cycle detected in metric dependencies involving: {missing}")

    sorted_metrics = [metric_map[mid] for mid in sorted_ids if mid in metric_map]
    return sorted_metrics, dict(dep_map)


def resolve_skips(
    metric_id: str,
    dependencies: list[MetricDependency],
    results: dict[str, object],
) -> str | None:
    """Check if a metric should be skipped based on dependency results.

    Args:
        metric_id: The metric to check.
        dependencies: All dependency declarations.
        results: Dict of metric_id -> MetricResult for already-evaluated metrics.

    Returns:
        Skip reason string if metric should be skipped, None if it should run.
    """
    for dep in dependencies:
        if dep.metric_id == metric_id:
            dep_result = results.get(dep.depends_on)
            if dep_result is None:
                return f"dependency '{dep.depends_on}' has no result"
            if not dep.check(dep_result):
                return f"dependency '{dep.depends_on}' condition '{dep.condition}' not met"
    return None


@dataclass
class ConditionalMetricChain:
    """A chain where a follow-up metric runs only when a trigger metric fails.

    Example: if toxicity_safety fails, run toxicity_type_classifier to diagnose.

    Usage:
        chain = ConditionalMetricChain(
            trigger_metric_id="toxicity_safety",
            condition="failed",
            follow_up_metric=toxicity_type_classifier_metric,
        )
    """

    trigger_metric_id: str
    condition: str  # "failed", "passed", "score_below:0.5"
    follow_up_metric: object  # Metric instance

    def should_run(self, trigger_result) -> bool:
        """Check if the follow-up should run based on the trigger result."""
        if trigger_result is None:
            return False

        if self.condition == "failed":
            return trigger_result.passed is False
        if self.condition == "passed":
            return trigger_result.passed is True

        if self.condition.startswith("score_below:"):
            try:
                threshold = float(self.condition.split(":")[1])
                return (trigger_result.score or 0.0) < threshold
            except (ValueError, IndexError):
                return False

        return False


def evaluate_chains(
    chains: list[ConditionalMetricChain],
    results: dict[str, object],
    call: object,
    item: object,
) -> list:
    """Evaluate conditional metric chains based on existing results.

    Args:
        chains: List of ConditionalMetricChain definitions.
        results: Dict of metric_id -> MetricResult from primary evaluation.
        call: FunctionCall for evaluation.
        item: DatasetItem for evaluation.

    Returns:
        List of MetricResult from triggered follow-up metrics.
    """
    follow_up_results = []
    for chain in chains:
        trigger_result = results.get(chain.trigger_metric_id)
        if chain.should_run(trigger_result):
            try:
                result = chain.follow_up_metric.evaluate(call, item)
                follow_up_results.append(result)
            except Exception:
                pass  # follow-up failure is non-fatal
    return follow_up_results
