"""Declarative evaluation API: single-call evaluation pattern.

Matches the industry-standard pattern used by Braintrust, W&B Weave,
and DeepEval: specify data, task, and scorers in one call.

Usage:
    from evalyn_sdk.declarative import Eval

    result = Eval(
        name="helpfulness-test",
        data=[{"input": "What is Python?", "output": "A programming language"}],
        scorers=["output_nonempty", "json_valid"],
    ).run()
    print(result.pass_rate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .models import DatasetItem, MetricResult


@dataclass
class EvalConfig:
    """Configuration for a declarative evaluation."""

    name: str = ""
    provider: str = "gemini"
    max_workers: int = 1
    tags: Optional[List[str]] = None


@dataclass
class EvalOutput:
    """Result from a declarative evaluation."""

    name: str
    items_evaluated: int
    results: List[MetricResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Overall pass rate (items passing all metrics)."""
        if not self.results:
            return 0.0
        item_pass: Dict[str, bool] = {}
        for r in self.results:
            if r.item_id not in item_pass:
                item_pass[r.item_id] = True
            if r.passed is False:
                item_pass[r.item_id] = False
        if not item_pass:
            return 0.0
        return sum(1 for v in item_pass.values() if v) / len(item_pass)

    @property
    def metric_pass_rates(self) -> Dict[str, float]:
        counts: Dict[str, list] = {}
        for r in self.results:
            mid = r.metric_id
            if mid not in counts:
                counts[mid] = [0, 0]
            counts[mid][1] += 1
            if r.passed is True:
                counts[mid][0] += 1
        return {mid: p / t if t > 0 else 0.0 for mid, (p, t) in counts.items()}

    @property
    def failed_items(self) -> List[str]:
        item_pass: Dict[str, bool] = {}
        for r in self.results:
            if r.item_id not in item_pass:
                item_pass[r.item_id] = True
            if r.passed is False:
                item_pass[r.item_id] = False
        return [iid for iid, v in item_pass.items() if not v]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "items_evaluated": self.items_evaluated,
            "pass_rate": round(self.pass_rate, 4),
            "metric_pass_rates": {k: round(v, 4) for k, v in self.metric_pass_rates.items()},
            "failed_items": self.failed_items,
            "total_results": len(self.results),
        }


class Eval:
    """Declarative evaluation: data + scorers in one call.

    Usage:
        result = Eval(
            name="test",
            data=[{"input": "q", "output": "a"}],
            scorers=["output_nonempty"],
        ).run()
    """

    def __init__(
        self,
        name: str = "",
        data: Optional[List] = None,
        scorers: Optional[List[str]] = None,
        task: Optional[Callable] = None,
        config: Optional[EvalConfig] = None,
    ):
        self.name = name
        self._data = data or []
        self._scorers = scorers or []
        self._task = task
        self._config = config or EvalConfig(name=name)

    def run(self) -> EvalOutput:
        """Execute the evaluation and return results."""
        from datetime import datetime, timezone

        from .metrics.factory import build_objective_metric
        from .models import FunctionCall

        # Build items
        items = []
        for i, d in enumerate(self._data):
            if isinstance(d, DatasetItem):
                items.append(d)
            elif isinstance(d, dict):
                items.append(DatasetItem(
                    id=d.get("id", f"item-{i}"),
                    input=d.get("input", {}),
                    output=d.get("output"),
                ))
            else:
                items.append(DatasetItem(id=f"item-{i}", input={}, output=str(d)))

        # Apply task function if provided
        if self._task:
            for item in items:
                try:
                    result = self._task(item.input)
                    item.output = result
                except Exception:
                    pass

        # Build metrics
        metrics = []
        for scorer_id in self._scorers:
            try:
                m = build_objective_metric(scorer_id, {})
                if m:
                    metrics.append(m)
            except (KeyError, Exception):
                pass

        # Evaluate
        now = datetime.now(timezone.utc)
        all_results = []
        for item in items:
            call = FunctionCall(
                id=f"call-{item.id}",
                function_name=self.name or "eval",
                inputs=item.input if isinstance(item.input, dict) else {"input": item.input},
                output=item.output,
                error=None,
                started_at=now,
                ended_at=now,
                duration_ms=0,
                session_id=None,
            )
            for metric in metrics:
                try:
                    result = metric.evaluate(call, item)
                    all_results.append(result)
                except Exception:
                    pass

        return EvalOutput(
            name=self.name,
            items_evaluated=len(items),
            results=all_results,
        )
