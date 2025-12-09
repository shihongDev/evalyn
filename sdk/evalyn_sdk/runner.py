from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, List, Optional
from uuid import uuid4

from .decorators import get_default_tracer
from .metrics.registry import Metric
from .models import DatasetItem, EvalRun, MetricResult, now_utc
from .storage.base import StorageBackend
from .tracing import EvalTracer


class EvalRunner:
    """
    Executes a dataset against a target function, applies metrics, and stores the run.
    If `instrument=True`, the runner will wrap the target function with the tracer automatically.
    """

    def __init__(
        self,
        target_fn: Callable,
        metrics: Iterable[Metric],
        tracer: Optional[EvalTracer] = None,
        storage: Optional[StorageBackend] = None,
        dataset_name: str = "dataset",
        instrument: bool = True,
    ):
        self.tracer = tracer or get_default_tracer()
        if storage:
            self.tracer.attach_storage(storage)
        self.dataset_name = dataset_name
        self.metrics: List[Metric] = list(metrics)
        already_wrapped = getattr(target_fn, "_evalyn_instrumented", False)
        self.target_fn = (
            target_fn
            if not instrument or already_wrapped
            else self.tracer.instrument(target_fn)
        )

    def run_dataset(self, dataset: Iterable[DatasetItem]) -> EvalRun:
        metric_results: List[MetricResult] = []
        failures: List[str] = []

        for item in dataset:
            try:
                self.target_fn(**item.inputs)
            except Exception:
                # The tracer already captured the error; continue so metrics can still record failure state.
                failures.append(item.id)

            call = self.tracer.last_call
            if call is None:
                raise RuntimeError(
                    "No trace was captured for the last call. Ensure the function is instrumented with @eval."
                )

            for metric in self.metrics:
                metric_results.append(metric.evaluate(call, item))

        summary = self._summarize(metric_results, failures)
        run = EvalRun(
            id=str(uuid4()),
            dataset_name=self.dataset_name,
            created_at=now_utc(),
            metric_results=metric_results,
            metrics=[m.spec for m in self.metrics],
            summary=summary,
        )

        if self.tracer.storage:
            self.tracer.storage.store_eval_run(run)

        return run

    @staticmethod
    def _summarize(results: List[MetricResult], failures: List[str]) -> dict:
        by_metric: defaultdict[str, List[MetricResult]] = defaultdict(list)
        for res in results:
            by_metric[res.metric_id].append(res)

        summary = {"metrics": {}, "failed_items": failures}
        for metric_id, metric_results in by_metric.items():
            scores = [r.score for r in metric_results if r.score is not None]
            passes = [r.passed for r in metric_results if r.passed is not None]
            summary["metrics"][metric_id] = {
                "count": len(metric_results),
                "avg_score": (sum(scores) / len(scores)) if scores else None,
                "pass_rate": (sum(1 for p in passes if p) / len(passes)) if passes else None,
            }
        return summary
