from __future__ import annotations

from typing import Any

from .registry import Metric
from ..judges import LLMJudge
from ..models import DatasetItem, FunctionCall, MetricResult, MetricSpec


def subjective_metric(
    metric_id: str,
    judge: LLMJudge,
    description: str = "LLM judge subjective score",
    success_threshold: float = 0.5,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name=f"Subjective - {judge.name}",
        type="subjective",
        description=description,
        config={"success_threshold": success_threshold, "model": judge.model},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        judge_raw = judge.score(call, item)
        score = judge_raw.get("score")
        passed = score is not None and score >= success_threshold
        return MetricResult(
            metric_id=spec.id,
            item_id=item.id,
            call_id=call.id,
            score=score,
            passed=passed,
            details={"judge": judge.name, "reason": judge_raw.get("reason")},
            raw_judge=judge_raw,
        )

    return Metric(spec, handler)
