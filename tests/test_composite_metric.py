"""Tests for CompositeMetric."""

import pytest
from datetime import datetime, timezone
from evalyn_sdk.models import (
    CompositeMetric,
    DatasetItem,
    FunctionCall,
    Metric,
    MetricResult,
    MetricSpec,
)


def _make_call(output="test output"):
    now = datetime.now(timezone.utc)
    return FunctionCall(
        id="call-1",
        function_name="test",
        inputs={"query": "test"},
        output=output,
        error=None,
        started_at=now,
        ended_at=now,
        duration_ms=10,
        session_id=None,
    )


def _make_item():
    return DatasetItem(id="item-1", input={"query": "test"}, output="test output")


def _make_metric(metric_id: str, score: float, passed: bool) -> Metric:
    """Create a simple metric that always returns the given score and passed."""
    spec = MetricSpec(id=metric_id, name=metric_id, type="objective")

    def handler(call, item):
        return MetricResult(
            metric_id=metric_id,
            item_id=item.id,
            call_id=call.id,
            score=score,
            passed=passed,
            details={},
        )

    return Metric(spec, handler)


class TestCompositeMetricCreation:
    """Test composite metric construction."""

    def test_basic_creation(self):
        m1 = _make_metric("a", 0.8, True)
        m2 = _make_metric("b", 0.6, True)
        composite = CompositeMetric(
            metric_id="composite",
            children=[(m1, 0.5), (m2, 0.5)],
        )
        assert composite.spec.id == "composite"
        assert len(composite.children) == 2

    def test_invalid_aggregation_raises(self):
        m1 = _make_metric("a", 0.8, True)
        with pytest.raises(ValueError, match="aggregation"):
            CompositeMetric(
                metric_id="bad",
                children=[(m1, 1.0)],
                aggregation="invalid",
            )

    def test_custom_description(self):
        m1 = _make_metric("a", 0.8, True)
        composite = CompositeMetric(
            metric_id="test",
            children=[(m1, 1.0)],
            description="My custom composite",
        )
        assert composite.spec.description == "My custom composite"

    def test_default_description(self):
        m1 = _make_metric("a", 0.8, True)
        composite = CompositeMetric(
            metric_id="test",
            children=[(m1, 1.0)],
        )
        assert "1 children" in composite.spec.description


class TestWeightedAverage:
    """Test weighted_average aggregation."""

    def test_equal_weights(self):
        m1 = _make_metric("a", 0.8, True)
        m2 = _make_metric("b", 0.6, True)
        composite = CompositeMetric(
            metric_id="avg",
            children=[(m1, 1.0), (m2, 1.0)],
            aggregation="weighted_average",
        )
        result = composite.evaluate(_make_call(), _make_item())
        assert result.score == pytest.approx(0.7)

    def test_unequal_weights(self):
        m1 = _make_metric("a", 1.0, True)
        m2 = _make_metric("b", 0.0, False)
        composite = CompositeMetric(
            metric_id="avg",
            children=[(m1, 0.7), (m2, 0.3)],
            aggregation="weighted_average",
        )
        result = composite.evaluate(_make_call(), _make_item())
        assert result.score == pytest.approx(0.7)

    def test_single_child(self):
        m1 = _make_metric("a", 0.9, True)
        composite = CompositeMetric(
            metric_id="single",
            children=[(m1, 1.0)],
            aggregation="weighted_average",
        )
        result = composite.evaluate(_make_call(), _make_item())
        assert result.score == pytest.approx(0.9)

    def test_zero_weights(self):
        m1 = _make_metric("a", 0.8, True)
        composite = CompositeMetric(
            metric_id="zero",
            children=[(m1, 0.0)],
            aggregation="weighted_average",
        )
        result = composite.evaluate(_make_call(), _make_item())
        assert result.score == 0.0


class TestMinAggregation:
    """Test min aggregation."""

    def test_min_returns_lowest(self):
        m1 = _make_metric("a", 0.9, True)
        m2 = _make_metric("b", 0.3, False)
        m3 = _make_metric("c", 0.7, True)
        composite = CompositeMetric(
            metric_id="min_test",
            children=[(m1, 1.0), (m2, 1.0), (m3, 1.0)],
            aggregation="min",
        )
        result = composite.evaluate(_make_call(), _make_item())
        assert result.score == pytest.approx(0.3)

    def test_min_all_same(self):
        m1 = _make_metric("a", 0.5, True)
        m2 = _make_metric("b", 0.5, True)
        composite = CompositeMetric(
            metric_id="min_same",
            children=[(m1, 1.0), (m2, 1.0)],
            aggregation="min",
        )
        result = composite.evaluate(_make_call(), _make_item())
        assert result.score == pytest.approx(0.5)


class TestMaxAggregation:
    """Test max aggregation."""

    def test_max_returns_highest(self):
        m1 = _make_metric("a", 0.3, False)
        m2 = _make_metric("b", 0.9, True)
        composite = CompositeMetric(
            metric_id="max_test",
            children=[(m1, 1.0), (m2, 1.0)],
            aggregation="max",
        )
        result = composite.evaluate(_make_call(), _make_item())
        assert result.score == pytest.approx(0.9)


class TestAllPassAggregation:
    """Test all_pass aggregation."""

    def test_all_pass_when_all_pass(self):
        m1 = _make_metric("a", 0.8, True)
        m2 = _make_metric("b", 0.9, True)
        composite = CompositeMetric(
            metric_id="all",
            children=[(m1, 1.0), (m2, 1.0)],
            aggregation="all_pass",
            threshold=0.5,
        )
        result = composite.evaluate(_make_call(), _make_item())
        assert result.score == 1.0
        assert result.passed is True

    def test_all_pass_fails_when_one_fails(self):
        m1 = _make_metric("a", 0.8, True)
        m2 = _make_metric("b", 0.2, False)
        composite = CompositeMetric(
            metric_id="all",
            children=[(m1, 1.0), (m2, 1.0)],
            aggregation="all_pass",
            threshold=0.5,
        )
        result = composite.evaluate(_make_call(), _make_item())
        assert result.score == 0.0
        assert result.passed is False


class TestThreshold:
    """Test pass/fail threshold."""

    def test_passes_above_threshold(self):
        m1 = _make_metric("a", 0.8, True)
        composite = CompositeMetric(
            metric_id="thresh",
            children=[(m1, 1.0)],
            threshold=0.7,
        )
        result = composite.evaluate(_make_call(), _make_item())
        assert result.passed is True

    def test_fails_below_threshold(self):
        m1 = _make_metric("a", 0.4, False)
        composite = CompositeMetric(
            metric_id="thresh",
            children=[(m1, 1.0)],
            threshold=0.7,
        )
        result = composite.evaluate(_make_call(), _make_item())
        assert result.passed is False

    def test_passes_at_exact_threshold(self):
        m1 = _make_metric("a", 0.7, True)
        composite = CompositeMetric(
            metric_id="thresh",
            children=[(m1, 1.0)],
            threshold=0.7,
        )
        result = composite.evaluate(_make_call(), _make_item())
        assert result.passed is True


class TestResultDetails:
    """Test that result details include child information."""

    def test_details_include_children(self):
        m1 = _make_metric("helpfulness", 0.8, True)
        m2 = _make_metric("safety", 0.9, True)
        composite = CompositeMetric(
            metric_id="quality",
            children=[(m1, 0.6), (m2, 0.4)],
        )
        result = composite.evaluate(_make_call(), _make_item())
        assert result.details["aggregation"] == "weighted_average"
        assert result.details["threshold"] == 0.5
        children = result.details["children"]
        assert len(children) == 2
        assert children[0]["metric_id"] == "helpfulness"
        assert children[0]["weight"] == 0.6
        assert children[1]["metric_id"] == "safety"
        assert children[1]["weight"] == 0.4

    def test_metric_id_in_result(self):
        m1 = _make_metric("a", 0.8, True)
        composite = CompositeMetric(metric_id="my_composite", children=[(m1, 1.0)])
        result = composite.evaluate(_make_call(), _make_item())
        assert result.metric_id == "my_composite"

    def test_item_id_in_result(self):
        m1 = _make_metric("a", 0.8, True)
        composite = CompositeMetric(metric_id="c", children=[(m1, 1.0)])
        item = DatasetItem(id="custom-item", input={}, output="")
        result = composite.evaluate(_make_call(), item)
        assert result.item_id == "custom-item"


class TestNoneScores:
    """Test handling of None scores from child metrics."""

    def test_none_score_treated_as_zero(self):
        """Child metrics with None scores should be treated as 0.0."""
        spec = MetricSpec(id="none_metric", name="none", type="objective")

        def handler(call, item):
            return MetricResult(
                metric_id="none_metric", item_id=item.id, call_id=call.id,
                score=None, passed=None, details={},
            )

        none_metric = Metric(spec, handler)
        composite = CompositeMetric(
            metric_id="with_none",
            children=[(none_metric, 1.0)],
            aggregation="weighted_average",
        )
        result = composite.evaluate(_make_call(), _make_item())
        assert result.score == 0.0
