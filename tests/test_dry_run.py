"""Tests for evaluation dry-run estimation."""

import pytest
from evalyn_sdk.evaluation.dry_run import estimate_eval_cost, DryRunEstimate
from evalyn_sdk.models import Metric, MetricSpec, MetricResult


def _make_metric(metric_id: str, metric_type: str = "objective") -> Metric:
    spec = MetricSpec(id=metric_id, name=metric_id, type=metric_type)

    def handler(call, item):
        return MetricResult(
            metric_id=metric_id, item_id=item.id, call_id=call.id,
            score=1.0, passed=True, details={},
        )

    return Metric(spec, handler)


class TestEstimateEvalCost:
    """Test cost estimation."""

    def test_objective_only_is_free(self):
        metrics = [_make_metric("json_valid"), _make_metric("latency")]
        est = estimate_eval_cost(item_count=100, metrics=metrics)
        assert est.estimated_cost_usd == 0.0
        assert est.estimated_input_tokens == 0
        assert est.estimated_output_tokens == 0
        assert est.objective_metrics == 2
        assert est.subjective_metrics == 0

    def test_subjective_has_cost(self):
        metrics = [_make_metric("helpfulness", "subjective")]
        est = estimate_eval_cost(item_count=50, metrics=metrics)
        assert est.estimated_cost_usd > 0
        assert est.estimated_input_tokens > 0
        assert est.estimated_output_tokens > 0
        assert est.subjective_metrics == 1

    def test_mixed_metrics(self):
        metrics = [
            _make_metric("json_valid"),
            _make_metric("helpfulness", "subjective"),
            _make_metric("toxicity", "subjective"),
        ]
        est = estimate_eval_cost(item_count=100, metrics=metrics)
        assert est.objective_metrics == 1
        assert est.subjective_metrics == 2
        assert est.total_metrics == 3
        assert est.total_evaluations == 300

    def test_zero_items(self):
        metrics = [_make_metric("helpfulness", "subjective")]
        est = estimate_eval_cost(item_count=0, metrics=metrics)
        assert est.estimated_cost_usd == 0.0
        assert est.total_evaluations == 0

    def test_empty_metrics(self):
        est = estimate_eval_cost(item_count=100, metrics=[])
        assert est.total_metrics == 0
        assert est.estimated_cost_usd == 0.0

    def test_cost_scales_with_items(self):
        metrics = [_make_metric("helpfulness", "subjective")]
        est_small = estimate_eval_cost(item_count=10, metrics=metrics)
        est_large = estimate_eval_cost(item_count=100, metrics=metrics)
        assert est_large.estimated_cost_usd > est_small.estimated_cost_usd
        # Cost should scale roughly linearly
        ratio = est_large.estimated_cost_usd / est_small.estimated_cost_usd
        assert 9 < ratio < 11

    def test_consistency_confidence_multiplies_cost(self):
        metrics = [_make_metric("helpfulness", "subjective")]
        est_none = estimate_eval_cost(
            item_count=50, metrics=metrics, confidence_method="none",
        )
        est_consistency = estimate_eval_cost(
            item_count=50, metrics=metrics,
            confidence_method="consistency", confidence_samples=5,
        )
        # Consistency with 5 samples should be ~5x the cost
        ratio = est_consistency.estimated_cost_usd / est_none.estimated_cost_usd
        assert 4.5 < ratio < 5.5

    def test_provider_affects_cost(self):
        metrics = [_make_metric("helpfulness", "subjective")]
        est_gemini = estimate_eval_cost(
            item_count=50, metrics=metrics, provider="gemini",
        )
        est_openai = estimate_eval_cost(
            item_count=50, metrics=metrics, provider="openai",
        )
        # Both should have costs but may differ based on pricing
        assert est_gemini.estimated_cost_usd > 0
        assert est_openai.estimated_cost_usd > 0


class TestTimeEstimation:
    """Test time estimation."""

    def test_objective_is_fast(self):
        metrics = [_make_metric("json_valid")]
        est = estimate_eval_cost(item_count=100, metrics=metrics)
        assert est.estimated_time_seconds < 1.0

    def test_subjective_is_slower(self):
        metrics = [_make_metric("helpfulness", "subjective")]
        est = estimate_eval_cost(item_count=100, metrics=metrics)
        assert est.estimated_time_seconds > 10.0

    def test_parallel_reduces_time(self):
        metrics = [_make_metric("helpfulness", "subjective")]
        est_seq = estimate_eval_cost(item_count=100, metrics=metrics, max_workers=1)
        est_par = estimate_eval_cost(item_count=100, metrics=metrics, max_workers=4)
        assert est_par.estimated_time_with_workers < est_seq.estimated_time_with_workers

    def test_sequential_and_parallel_same_when_single_worker(self):
        metrics = [_make_metric("helpfulness", "subjective")]
        est = estimate_eval_cost(item_count=50, metrics=metrics, max_workers=1)
        assert est.estimated_time_seconds == est.estimated_time_with_workers


class TestMetricBreakdown:
    """Test per-metric breakdown in estimates."""

    def test_breakdown_has_all_metrics(self):
        metrics = [
            _make_metric("json_valid"),
            _make_metric("helpfulness", "subjective"),
        ]
        est = estimate_eval_cost(item_count=50, metrics=metrics)
        assert len(est.metric_estimates) == 2
        ids = [m["metric_id"] for m in est.metric_estimates]
        assert "json_valid" in ids
        assert "helpfulness" in ids

    def test_breakdown_objective_free(self):
        metrics = [_make_metric("json_valid")]
        est = estimate_eval_cost(item_count=50, metrics=metrics)
        breakdown = est.metric_estimates[0]
        assert breakdown["estimated_cost_usd"] == 0.0
        assert breakdown["type"] == "objective"

    def test_breakdown_subjective_has_cost(self):
        metrics = [_make_metric("helpfulness", "subjective")]
        est = estimate_eval_cost(item_count=50, metrics=metrics)
        breakdown = est.metric_estimates[0]
        assert breakdown["estimated_cost_usd"] > 0
        assert breakdown["type"] == "subjective"


class TestDryRunEstimate:
    """Test DryRunEstimate formatting."""

    def test_as_dict(self):
        metrics = [_make_metric("json_valid"), _make_metric("help", "subjective")]
        est = estimate_eval_cost(item_count=20, metrics=metrics)
        d = est.as_dict()
        assert "total_items" in d
        assert "estimated_cost_usd" in d
        assert "metric_estimates" in d
        assert isinstance(d["estimated_cost_usd"], float)

    def test_format_text(self):
        metrics = [_make_metric("json_valid"), _make_metric("help", "subjective")]
        est = estimate_eval_cost(item_count=20, metrics=metrics)
        text = est.format_text()
        assert "Evaluation Dry-Run Estimate" in text
        assert "Items:" in text
        assert "Cost estimate:" in text
        assert "Time estimate:" in text
        assert "json_valid" in text
        assert "help" in text

    def test_format_text_empty(self):
        est = estimate_eval_cost(item_count=0, metrics=[])
        text = est.format_text()
        assert "Items:              0" in text
