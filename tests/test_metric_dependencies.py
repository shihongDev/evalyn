"""Tests for metric dependency resolution."""

import pytest
from evalyn_sdk.models import Metric, MetricSpec, MetricResult
from evalyn_sdk.evaluation.dependencies import (
    MetricDependency,
    topological_sort,
    resolve_skips,
)


def _metric(mid: str) -> Metric:
    spec = MetricSpec(id=mid, name=mid, type="objective")
    return Metric(spec, lambda c, i: MetricResult(
        metric_id=mid, item_id=i.id, call_id=c.id,
        score=1.0, passed=True, details={},
    ))


def _result(metric_id: str, passed: bool = True, score: float = 1.0) -> MetricResult:
    return MetricResult(
        metric_id=metric_id, item_id="item-1", call_id="call-1",
        score=score, passed=passed, details={},
    )


class TestMetricDependency:
    def test_check_passed_true(self):
        dep = MetricDependency(metric_id="b", depends_on="a", condition="passed")
        assert dep.check(_result("a", passed=True)) is True

    def test_check_passed_false(self):
        dep = MetricDependency(metric_id="b", depends_on="a", condition="passed")
        assert dep.check(_result("a", passed=False)) is False

    def test_check_none_result(self):
        dep = MetricDependency(metric_id="b", depends_on="a")
        assert dep.check(None) is False

    def test_check_score_above_met(self):
        dep = MetricDependency(metric_id="b", depends_on="a", condition="score_above:0.7")
        assert dep.check(_result("a", score=0.8)) is True

    def test_check_score_above_not_met(self):
        dep = MetricDependency(metric_id="b", depends_on="a", condition="score_above:0.7")
        assert dep.check(_result("a", score=0.5)) is False

    def test_check_unknown_condition(self):
        dep = MetricDependency(metric_id="b", depends_on="a", condition="unknown")
        assert dep.check(_result("a")) is True  # unknown = run anyway


class TestTopologicalSort:
    def test_no_dependencies(self):
        metrics = [_metric("a"), _metric("b"), _metric("c")]
        sorted_m, dep_map = topological_sort(metrics, [])
        assert len(sorted_m) == 3
        assert dep_map == {}

    def test_simple_chain(self):
        metrics = [_metric("a"), _metric("b"), _metric("c")]
        deps = [
            MetricDependency(metric_id="b", depends_on="a"),
            MetricDependency(metric_id="c", depends_on="b"),
        ]
        sorted_m, dep_map = topological_sort(metrics, deps)
        ids = [m.spec.id for m in sorted_m]
        assert ids.index("a") < ids.index("b") < ids.index("c")

    def test_diamond_dependency(self):
        metrics = [_metric("a"), _metric("b"), _metric("c"), _metric("d")]
        deps = [
            MetricDependency(metric_id="b", depends_on="a"),
            MetricDependency(metric_id="c", depends_on="a"),
            MetricDependency(metric_id="d", depends_on="b"),
            MetricDependency(metric_id="d", depends_on="c"),
        ]
        sorted_m, dep_map = topological_sort(metrics, deps)
        ids = [m.spec.id for m in sorted_m]
        assert ids.index("a") < ids.index("b")
        assert ids.index("a") < ids.index("c")
        assert ids.index("b") < ids.index("d")
        assert ids.index("c") < ids.index("d")

    def test_cycle_raises(self):
        metrics = [_metric("a"), _metric("b")]
        deps = [
            MetricDependency(metric_id="a", depends_on="b"),
            MetricDependency(metric_id="b", depends_on="a"),
        ]
        with pytest.raises(ValueError, match="Cycle"):
            topological_sort(metrics, deps)

    def test_unknown_dependency_ignored(self):
        metrics = [_metric("a")]
        deps = [MetricDependency(metric_id="a", depends_on="nonexistent")]
        sorted_m, _ = topological_sort(metrics, deps)
        assert len(sorted_m) == 1

    def test_empty(self):
        sorted_m, _ = topological_sort([], [])
        assert sorted_m == []


class TestResolveSkips:
    def test_no_dependencies_runs(self):
        reason = resolve_skips("a", [], {})
        assert reason is None  # should run

    def test_dependency_met_runs(self):
        deps = [MetricDependency(metric_id="b", depends_on="a")]
        results = {"a": _result("a", passed=True)}
        reason = resolve_skips("b", deps, results)
        assert reason is None  # should run

    def test_dependency_not_met_skips(self):
        deps = [MetricDependency(metric_id="b", depends_on="a")]
        results = {"a": _result("a", passed=False)}
        reason = resolve_skips("b", deps, results)
        assert reason is not None
        assert "not met" in reason

    def test_dependency_missing_result_skips(self):
        deps = [MetricDependency(metric_id="b", depends_on="a")]
        results = {}
        reason = resolve_skips("b", deps, results)
        assert reason is not None
        assert "no result" in reason

    def test_unrelated_dependency_ignored(self):
        deps = [MetricDependency(metric_id="c", depends_on="a")]
        results = {"a": _result("a", passed=False)}
        reason = resolve_skips("b", deps, results)
        assert reason is None  # b has no dependencies

    def test_score_condition(self):
        deps = [MetricDependency(metric_id="b", depends_on="a", condition="score_above:0.8")]
        results = {"a": _result("a", score=0.9)}
        assert resolve_skips("b", deps, results) is None

    def test_score_condition_not_met(self):
        deps = [MetricDependency(metric_id="b", depends_on="a", condition="score_above:0.8")]
        results = {"a": _result("a", score=0.5)}
        assert resolve_skips("b", deps, results) is not None
