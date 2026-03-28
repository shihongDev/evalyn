"""Tests for metric category pass rates and rubric preview."""

import pytest
from evalyn_sdk.models import MetricResult, MetricSpec
from evalyn_sdk.analysis.category_stats import (
    compute_category_pass_rates, CategoryStats, CategoryReport,
)
from evalyn_sdk.metrics.preview import (
    preview_metric, preview_all_metrics, preview_from_template,
    estimate_prompt_tokens, MetricPreview,
)


def _result(mid="m", passed=True):
    return MetricResult(
        metric_id=mid, item_id="i1", call_id="c1",
        score=1.0 if passed else 0.0, passed=passed, details={},
    )


# =============================================================================
# Category pass rates tests
# =============================================================================

class TestComputeCategoryPassRates:
    def test_basic(self):
        results = [
            _result("json_valid", True),
            _result("json_valid", True),
            _result("toxicity_safety", True),
            _result("toxicity_safety", False),
        ]
        cat_map = {"json_valid": "structure", "toxicity_safety": "safety"}
        report = compute_category_pass_rates(results, cat_map)
        cats = {c.category: c for c in report.categories}
        assert cats["structure"].pass_rate == 1.0
        assert cats["safety"].pass_rate == 0.5

    def test_weakest_category(self):
        results = [
            _result("a", True), _result("a", True),
            _result("b", True), _result("b", False), _result("b", False),
        ]
        cat_map = {"a": "good", "b": "bad"}
        report = compute_category_pass_rates(results, cat_map)
        assert report.weakest_category == "bad"

    def test_strongest_category(self):
        results = [
            _result("a", True), _result("a", True),
            _result("b", True), _result("b", False),
        ]
        cat_map = {"a": "strong", "b": "weak"}
        report = compute_category_pass_rates(results, cat_map)
        assert report.strongest_category == "strong"

    def test_uncategorized(self):
        results = [_result("unknown_metric", True)]
        report = compute_category_pass_rates(results, {})
        assert report.categories[0].category == "uncategorized"

    def test_empty(self):
        report = compute_category_pass_rates([])
        assert len(report.categories) == 0
        assert report.weakest_category is None

    def test_metric_count(self):
        results = [
            _result("a", True), _result("b", True), _result("c", False),
        ]
        cat_map = {"a": "cat1", "b": "cat1", "c": "cat2"}
        report = compute_category_pass_rates(results, cat_map)
        cats = {c.category: c for c in report.categories}
        assert cats["cat1"].metric_count == 2
        assert cats["cat2"].metric_count == 1

    def test_as_dict(self):
        results = [_result("a", True)]
        d = compute_category_pass_rates(results, {"a": "test"}).as_dict()
        assert "categories" in d
        assert "weakest" in d

    def test_format_text(self):
        results = [_result("a", True), _result("b", False)]
        text = compute_category_pass_rates(results, {"a": "good", "b": "bad"}).format_text()
        assert "Category Pass Rates" in text

    def test_default_categories(self):
        # Uses built-in registry categories
        results = [_result("json_valid", True)]
        report = compute_category_pass_rates(results)
        assert len(report.categories) >= 1

    def test_category_stats_as_dict(self):
        cs = CategoryStats(category="safety", metric_count=3, passed=8, failed=2, total_evaluations=10)
        d = cs.as_dict()
        assert d["pass_rate"] == 0.8
        assert d["metric_count"] == 3


# =============================================================================
# Metric preview tests
# =============================================================================

class TestPreviewMetric:
    def test_objective(self):
        spec = MetricSpec(id="json_valid", name="JSON Valid", type="objective",
                         description="Checks JSON validity")
        preview = preview_metric(spec)
        assert preview.metric_id == "json_valid"
        assert preview.metric_type == "objective"
        assert preview.prompt is None

    def test_subjective_with_prompt(self):
        spec = MetricSpec(
            id="helpfulness", name="Helpfulness", type="subjective",
            description="Evaluate helpfulness",
            config={"prompt": "Rate the helpfulness of this response",
                    "rubric": ["Must be accurate", "Must be helpful"],
                    "threshold": 0.5},
        )
        preview = preview_metric(spec)
        assert preview.prompt is not None
        assert len(preview.rubric) == 2
        assert preview.threshold == 0.5

    def test_as_dict(self):
        spec = MetricSpec(id="test", name="Test", type="objective")
        d = preview_metric(spec).as_dict()
        assert "metric_id" in d
        assert "type" in d

    def test_format_text_objective(self):
        spec = MetricSpec(id="test", name="Test", type="objective",
                         description="A test metric")
        text = preview_metric(spec).format_text()
        assert "test" in text
        assert "objective" in text

    def test_format_text_subjective(self):
        spec = MetricSpec(id="help", name="Help", type="subjective",
                         config={"prompt": "Rate quality", "rubric": ["criteria 1"]})
        text = preview_metric(spec).format_text()
        assert "Rubric" in text
        assert "criteria 1" in text


class TestPreviewAllMetrics:
    def test_multiple(self):
        specs = [
            MetricSpec(id="a", name="A", type="objective"),
            MetricSpec(id="b", name="B", type="subjective", config={"prompt": "test"}),
        ]
        previews = preview_all_metrics(specs)
        assert len(previews) == 2
        assert previews[0].metric_id == "a"
        assert previews[1].prompt == "test"

    def test_empty(self):
        assert preview_all_metrics([]) == []


class TestPreviewFromTemplate:
    def test_known_template(self):
        preview = preview_from_template("toxicity_safety")
        if preview:  # only works if registry is importable
            assert preview.metric_type == "subjective"
            assert preview.prompt is not None

    def test_unknown_template(self):
        assert preview_from_template("nonexistent_metric_xyz") is None


class TestEstimatePromptTokens:
    def test_with_prompt(self):
        preview = MetricPreview(
            metric_id="help", metric_type="subjective",
            description="test",
            prompt="Rate the helpfulness of this response based on accuracy and completeness",
            rubric=["Must be accurate", "Must be helpful and complete"],
        )
        tokens = estimate_prompt_tokens(preview)
        assert tokens > 0

    def test_no_prompt(self):
        preview = MetricPreview(metric_id="obj", metric_type="objective", description="test")
        tokens = estimate_prompt_tokens(preview)
        assert tokens >= 50  # overhead only

    def test_longer_prompt_more_tokens(self):
        short = MetricPreview(metric_id="a", metric_type="subjective",
                             description="", prompt="short")
        long = MetricPreview(metric_id="b", metric_type="subjective",
                            description="", prompt="a" * 1000)
        assert estimate_prompt_tokens(long) > estimate_prompt_tokens(short)
