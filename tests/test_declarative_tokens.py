"""Tests for declarative evaluation API and metric token count estimation."""

import pytest
from evalyn_sdk.declarative import Eval, EvalOutput, EvalConfig
from evalyn_sdk.models import DatasetItem, MetricSpec
from evalyn_sdk.metrics.token_count import (
    estimate_metric_tokens, estimate_all_tokens, MetricTokenEstimate,
)


# =============================================================================
# Declarative API tests
# =============================================================================

class TestEval:
    def test_basic_run(self):
        result = Eval(
            name="test",
            data=[{"input": "hello", "output": "world"}],
            scorers=["output_nonempty"],
        ).run()
        assert isinstance(result, EvalOutput)
        assert result.items_evaluated == 1
        assert result.pass_rate >= 0.0

    def test_multiple_items(self):
        data = [
            {"input": "q1", "output": "answer 1"},
            {"input": "q2", "output": "answer 2"},
            {"input": "q3", "output": "answer 3"},
        ]
        result = Eval(name="multi", data=data, scorers=["output_nonempty"]).run()
        assert result.items_evaluated == 3
        assert len(result.results) == 3

    def test_multiple_scorers(self):
        data = [{"input": "q", "output": '{"key": "value"}'}]
        result = Eval(data=data, scorers=["output_nonempty", "json_valid"]).run()
        assert len(result.results) == 2

    def test_with_name(self):
        result = Eval(name="experiment-v2", data=[{"output": "x"}], scorers=["output_nonempty"]).run()
        assert result.name == "experiment-v2"

    def test_empty_output_fails(self):
        result = Eval(
            data=[{"input": "q", "output": ""}],
            scorers=["output_nonempty"],
        ).run()
        assert result.pass_rate == 0.0
        assert len(result.failed_items) == 1

    def test_dataset_items(self):
        items = [DatasetItem(id="d1", input="q", output="a")]
        result = Eval(data=items, scorers=["output_nonempty"]).run()
        assert result.items_evaluated == 1

    def test_with_task_function(self):
        def my_task(input_data):
            return f"response to {input_data}"

        result = Eval(
            data=[{"input": "hello"}],
            task=my_task,
            scorers=["output_nonempty"],
        ).run()
        assert result.items_evaluated == 1
        assert result.pass_rate == 1.0

    def test_metric_pass_rates(self):
        result = Eval(
            data=[{"output": "content"}, {"output": ""}],
            scorers=["output_nonempty"],
        ).run()
        rates = result.metric_pass_rates
        assert "output_nonempty" in rates
        assert rates["output_nonempty"] == 0.5

    def test_as_dict(self):
        result = Eval(
            name="test",
            data=[{"output": "a"}],
            scorers=["output_nonempty"],
        ).run()
        d = result.as_dict()
        assert "name" in d
        assert "pass_rate" in d
        assert "metric_pass_rates" in d

    def test_empty_data(self):
        result = Eval(data=[], scorers=["output_nonempty"]).run()
        assert result.items_evaluated == 0
        assert result.pass_rate == 0.0

    def test_empty_scorers(self):
        result = Eval(data=[{"output": "a"}], scorers=[]).run()
        assert result.items_evaluated == 1
        assert len(result.results) == 0

    def test_unknown_scorer_skipped(self):
        result = Eval(
            data=[{"output": "a"}],
            scorers=["nonexistent_metric_xyz"],
        ).run()
        assert len(result.results) == 0


# =============================================================================
# Token count estimation tests
# =============================================================================

class TestEstimateMetricTokens:
    def test_objective_is_free(self):
        spec = MetricSpec(id="json_valid", name="json", type="objective")
        est = estimate_metric_tokens(spec)
        assert est.is_free
        assert est.prompt_tokens == 0
        assert est.estimate_for_dataset(100) == 0

    def test_subjective_has_tokens(self):
        spec = MetricSpec(
            id="helpfulness", name="help", type="subjective",
            config={"prompt": "Rate the helpfulness of this response", "rubric": ["criteria 1"]},
        )
        est = estimate_metric_tokens(spec)
        assert not est.is_free
        assert est.prompt_tokens > 0
        assert est.total_per_item > 0

    def test_longer_prompt_more_tokens(self):
        short = MetricSpec(id="a", name="a", type="subjective", config={"prompt": "rate"})
        long = MetricSpec(id="b", name="b", type="subjective", config={"prompt": "x" * 1000})
        est_short = estimate_metric_tokens(short)
        est_long = estimate_metric_tokens(long)
        assert est_long.prompt_tokens > est_short.prompt_tokens

    def test_dataset_scaling(self):
        spec = MetricSpec(id="h", name="h", type="subjective", config={"prompt": "rate"})
        est = estimate_metric_tokens(spec)
        assert est.estimate_for_dataset(100) > est.estimate_for_dataset(10)

    def test_as_dict(self):
        spec = MetricSpec(id="test", name="test", type="objective")
        d = estimate_metric_tokens(spec).as_dict()
        assert "metric_id" in d
        assert "is_free" in d
        assert "prompt_tokens" in d


class TestEstimateAllTokens:
    def test_mixed_metrics(self):
        specs = [
            MetricSpec(id="obj", name="obj", type="objective"),
            MetricSpec(id="subj", name="subj", type="subjective", config={"prompt": "rate"}),
        ]
        result = estimate_all_tokens(specs, item_count=50)
        assert result["objective_count"] == 1
        assert result["subjective_count"] == 1
        assert result["total_for_dataset"] > 0

    def test_all_objective(self):
        specs = [MetricSpec(id="a", name="a", type="objective")]
        result = estimate_all_tokens(specs, item_count=100)
        assert result["total_for_dataset"] == 0  # free

    def test_empty(self):
        result = estimate_all_tokens([], item_count=100)
        assert result["total_for_dataset"] == 0
        assert result["objective_count"] == 0

    def test_zero_items(self):
        specs = [MetricSpec(id="s", name="s", type="subjective", config={"prompt": "x"})]
        result = estimate_all_tokens(specs, item_count=0)
        assert result["total_for_dataset"] == 0
