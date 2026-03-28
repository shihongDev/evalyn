"""Tests for conditional metric chains."""

import pytest
from datetime import datetime, timezone
from evalyn_sdk.models import Metric, MetricSpec, MetricResult, FunctionCall, DatasetItem
from evalyn_sdk.evaluation.dependencies import (
    ConditionalMetricChain,
    evaluate_chains,
)


def _result(metric_id: str, passed: bool = True, score: float = 1.0) -> MetricResult:
    return MetricResult(
        metric_id=metric_id, item_id="item-1", call_id="call-1",
        score=score, passed=passed, details={},
    )


def _metric(mid: str) -> Metric:
    spec = MetricSpec(id=mid, name=mid, type="objective")
    return Metric(spec, lambda c, i: MetricResult(
        metric_id=mid, item_id=i.id, call_id=c.id,
        score=0.5, passed=True, details={"from_chain": True},
    ))


def _call():
    now = datetime.now(timezone.utc)
    return FunctionCall(
        id="call-1", function_name="test", inputs={}, output="out",
        error=None, started_at=now, ended_at=now, duration_ms=10, session_id=None,
    )


def _item():
    return DatasetItem(id="item-1", input={}, output="out")


class TestConditionalMetricChain:
    def test_should_run_on_failure(self):
        chain = ConditionalMetricChain(
            trigger_metric_id="safety",
            condition="failed",
            follow_up_metric=_metric("safety_type"),
        )
        assert chain.should_run(_result("safety", passed=False)) is True
        assert chain.should_run(_result("safety", passed=True)) is False

    def test_should_run_on_pass(self):
        chain = ConditionalMetricChain(
            trigger_metric_id="basic",
            condition="passed",
            follow_up_metric=_metric("detailed"),
        )
        assert chain.should_run(_result("basic", passed=True)) is True
        assert chain.should_run(_result("basic", passed=False)) is False

    def test_should_run_score_below(self):
        chain = ConditionalMetricChain(
            trigger_metric_id="quality",
            condition="score_below:0.5",
            follow_up_metric=_metric("diagnosis"),
        )
        assert chain.should_run(_result("quality", score=0.3)) is True
        assert chain.should_run(_result("quality", score=0.8)) is False

    def test_should_run_none_result(self):
        chain = ConditionalMetricChain(
            trigger_metric_id="x",
            condition="failed",
            follow_up_metric=_metric("y"),
        )
        assert chain.should_run(None) is False

    def test_unknown_condition(self):
        chain = ConditionalMetricChain(
            trigger_metric_id="x",
            condition="unknown_cond",
            follow_up_metric=_metric("y"),
        )
        assert chain.should_run(_result("x")) is False


class TestEvaluateChains:
    def test_chain_triggered(self):
        chains = [
            ConditionalMetricChain(
                trigger_metric_id="safety",
                condition="failed",
                follow_up_metric=_metric("safety_type"),
            ),
        ]
        results = {"safety": _result("safety", passed=False)}
        follow_ups = evaluate_chains(chains, results, _call(), _item())
        assert len(follow_ups) == 1
        assert follow_ups[0].metric_id == "safety_type"

    def test_chain_not_triggered(self):
        chains = [
            ConditionalMetricChain(
                trigger_metric_id="safety",
                condition="failed",
                follow_up_metric=_metric("safety_type"),
            ),
        ]
        results = {"safety": _result("safety", passed=True)}
        follow_ups = evaluate_chains(chains, results, _call(), _item())
        assert len(follow_ups) == 0

    def test_multiple_chains(self):
        chains = [
            ConditionalMetricChain("safety", "failed", _metric("safety_type")),
            ConditionalMetricChain("quality", "failed", _metric("quality_diag")),
        ]
        results = {
            "safety": _result("safety", passed=False),
            "quality": _result("quality", passed=True),
        }
        follow_ups = evaluate_chains(chains, results, _call(), _item())
        assert len(follow_ups) == 1
        assert follow_ups[0].metric_id == "safety_type"

    def test_missing_trigger_no_crash(self):
        chains = [ConditionalMetricChain("missing", "failed", _metric("diag"))]
        follow_ups = evaluate_chains(chains, {}, _call(), _item())
        assert len(follow_ups) == 0

    def test_empty_chains(self):
        follow_ups = evaluate_chains([], {}, _call(), _item())
        assert follow_ups == []
