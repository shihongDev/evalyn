"""Tests for metric normalization and split-model routing."""

import pytest
from evalyn_sdk.models import Metric, MetricSpec, MetricResult
from evalyn_sdk.analysis.normalization import (
    normalize_z_score, normalize_min_max, normalize_run_scores,
)
from evalyn_sdk.evaluation.split_routing import (
    plan_split_routing, estimate_routing_savings, RoutingPlan, RoutingResult,
)


def _metric(mid, mtype="objective"):
    spec = MetricSpec(id=mid, name=mid, type=mtype)
    return Metric(spec, lambda c, i: MetricResult(
        metric_id=mid, item_id=i.id, call_id=c.id,
        score=1.0, passed=True, details={},
    ))


def _result(mid, score):
    return MetricResult(
        metric_id=mid, item_id="i1", call_id="c1",
        score=score, passed=True, details={},
    )


# =============================================================================
# Normalization tests
# =============================================================================

class TestZScoreNormalization:
    def test_basic(self):
        scores = {"a": [1.0, 2.0, 3.0, 4.0, 5.0]}
        result = normalize_z_score(scores)
        assert result["a"].method == "z_score"
        assert result["a"].original_mean == pytest.approx(3.0)
        # Middle value should have z-score near 0
        assert abs(result["a"].normalized_scores[2]) < 0.01

    def test_negative_and_positive(self):
        scores = {"a": [0.0, 10.0]}
        result = normalize_z_score(scores)
        ns = result["a"].normalized_scores
        assert ns[0] < 0  # below mean
        assert ns[1] > 0  # above mean

    def test_constant_scores(self):
        scores = {"a": [5.0, 5.0, 5.0]}
        result = normalize_z_score(scores)
        assert all(s == 0.0 for s in result["a"].normalized_scores)

    def test_single_score(self):
        scores = {"a": [0.8]}
        result = normalize_z_score(scores)
        assert result["a"].normalized_scores == [0.8]

    def test_empty(self):
        assert normalize_z_score({}) == {}

    def test_multiple_metrics(self):
        scores = {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}
        result = normalize_z_score(scores)
        assert len(result) == 2


class TestMinMaxNormalization:
    def test_basic(self):
        scores = {"a": [0.0, 5.0, 10.0]}
        result = normalize_min_max(scores)
        ns = result["a"].normalized_scores
        assert ns[0] == pytest.approx(0.0)
        assert ns[1] == pytest.approx(0.5)
        assert ns[2] == pytest.approx(1.0)

    def test_custom_range(self):
        scores = {"a": [0.0, 10.0]}
        result = normalize_min_max(scores, target_min=0.0, target_max=100.0)
        assert result["a"].normalized_scores[0] == pytest.approx(0.0)
        assert result["a"].normalized_scores[1] == pytest.approx(100.0)

    def test_constant_scores(self):
        scores = {"a": [5.0, 5.0, 5.0]}
        result = normalize_min_max(scores)
        # All same: map to midpoint
        assert all(s == pytest.approx(0.5) for s in result["a"].normalized_scores)

    def test_already_0_1(self):
        scores = {"a": [0.0, 0.5, 1.0]}
        result = normalize_min_max(scores)
        ns = result["a"].normalized_scores
        assert ns == pytest.approx([0.0, 0.5, 1.0])

    def test_empty_scores(self):
        scores = {"a": []}
        result = normalize_min_max(scores)
        assert result["a"].normalized_scores == []


class TestNormalizeRunScores:
    def test_from_results(self):
        results = [_result("a", 0.8), _result("a", 0.9), _result("b", 0.5)]
        normalized = normalize_run_scores(results, method="min_max")
        assert "a" in normalized
        assert "b" in normalized

    def test_z_score_method(self):
        results = [_result("a", 0.5), _result("a", 0.7), _result("a", 0.9)]
        normalized = normalize_run_scores(results, method="z_score")
        assert normalized["a"].method == "z_score"

    def test_skips_none_scores(self):
        results = [
            MetricResult(metric_id="a", item_id="i1", call_id="c1",
                        score=None, passed=None, details={}),
        ]
        normalized = normalize_run_scores(results)
        assert "a" not in normalized  # no valid scores

    def test_as_dict(self):
        results = [_result("a", 0.8), _result("a", 0.9)]
        normalized = normalize_run_scores(results)
        d = normalized["a"].as_dict()
        assert "metric_id" in d
        assert "method" in d
        assert "normalized_mean" in d


# =============================================================================
# Split routing tests
# =============================================================================

class TestPlanSplitRouting:
    def test_mixed_metrics(self):
        metrics = [
            _metric("json_valid", "objective"),
            _metric("helpfulness", "subjective"),
            _metric("latency", "objective"),
        ]
        plan = plan_split_routing(metrics)
        assert plan.local_count == 2
        assert plan.api_count == 1
        assert len(plan.local_metrics) == 2
        assert len(plan.api_metrics) == 1

    def test_all_objective(self):
        metrics = [_metric("a"), _metric("b")]
        plan = plan_split_routing(metrics)
        assert plan.local_count == 2
        assert plan.api_count == 0

    def test_all_subjective(self):
        metrics = [_metric("a", "subjective"), _metric("b", "subjective")]
        plan = plan_split_routing(metrics)
        assert plan.local_count == 0
        assert plan.api_count == 2

    def test_empty(self):
        plan = plan_split_routing([])
        assert plan.local_count == 0
        assert plan.api_count == 0

    def test_as_dict(self):
        plan = plan_split_routing([_metric("a"), _metric("b", "subjective")])
        d = plan.as_dict()
        assert d["local_count"] == 1
        assert d["api_count"] == 1
        assert d["total"] == 2

    def test_format_text(self):
        plan = plan_split_routing([_metric("a"), _metric("b", "subjective")])
        text = plan.format_text()
        assert "Local" in text
        assert "API" in text


class TestEstimateRoutingSavings:
    def test_all_objective_saves_everything(self):
        plan = RoutingPlan(
            local_metrics=[_metric("a"), _metric("b")],
            local_count=2, api_count=0,
        )
        savings = estimate_routing_savings(plan, item_count=100)
        assert savings["savings_pct"] == 1.0

    def test_all_subjective_saves_nothing(self):
        plan = RoutingPlan(
            api_metrics=[_metric("a", "subjective")],
            local_count=0, api_count=1,
        )
        savings = estimate_routing_savings(plan, item_count=100)
        assert savings["savings_pct"] == 0.0

    def test_mixed_savings(self):
        plan = RoutingPlan(
            local_metrics=[_metric("obj")],
            api_metrics=[_metric("subj", "subjective")],
            local_count=1, api_count=1,
        )
        savings = estimate_routing_savings(plan, item_count=100)
        assert savings["savings_pct"] == pytest.approx(0.5)
        assert savings["local_fraction"] == 0.5

    def test_empty_plan(self):
        plan = RoutingPlan()
        savings = estimate_routing_savings(plan, item_count=100)
        assert savings["savings_pct"] == 0.0


class TestRoutingResult:
    def test_all_results(self):
        r = RoutingResult(
            local_results=[_result("a", 1.0)],
            api_results=[_result("b", 0.8)],
        )
        assert r.total_results == 2
        assert len(r.all_results) == 2

    def test_savings_report(self):
        r = RoutingResult(api_cost_usd=0.05)
        report = r.savings_report(full_api_cost_estimate=0.10)
        assert report["saved_usd"] == pytest.approx(0.05)
        assert report["savings_pct"] == pytest.approx(0.5)
