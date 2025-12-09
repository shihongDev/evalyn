from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

from ..models import DatasetItem, FunctionCall, MetricResult, MetricSpec


class Metric:
    """Runtime metric that binds a spec to an evaluation function."""

    def __init__(
        self,
        spec: MetricSpec,
        handler: Callable[[FunctionCall, DatasetItem], MetricResult],
    ):
        self.spec = spec
        self.handler = handler

    def evaluate(self, call: FunctionCall, item: DatasetItem) -> MetricResult:
        return self.handler(call, item)


class MetricRegistry:
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:
        self._metrics[metric.spec.id] = metric

    def get(self, metric_id: str) -> Optional[Metric]:
        return self._metrics.get(metric_id)

    def list(self) -> List[Metric]:
        return list(self._metrics.values())

    def apply_all(self, call: FunctionCall, item: DatasetItem) -> List[MetricResult]:
        return [metric.evaluate(call, item) for metric in self._metrics.values()]
