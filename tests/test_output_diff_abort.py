"""Tests for output diff and abort conditions."""

import pytest
from evalyn_sdk.models import DatasetItem
from evalyn_sdk.analysis.output_diff import (
    compute_output_diff, diff_run_outputs, ItemDiff, OutputDiffReport,
)
from evalyn_sdk.evaluation.abort_conditions import (
    AbortController, AbortCondition, AbortCheckResult,
)


# =============================================================================
# Output diff tests
# =============================================================================

class TestComputeOutputDiff:
    def test_identical(self):
        diff = compute_output_diff("i1", "hello world", "hello world")
        assert diff.is_match is True
        assert diff.similarity == 1.0
        assert len(diff.diff_lines) == 0

    def test_completely_different(self):
        diff = compute_output_diff("i1", "aaa", "zzz")
        assert diff.is_match is False
        assert diff.similarity < 0.5

    def test_partial_match(self):
        diff = compute_output_diff("i1", "hello world", "hello there")
        assert not diff.is_match
        assert 0.3 < diff.similarity < 0.9

    def test_close_match(self):
        diff = compute_output_diff("i1", "The quick brown fox", "The quick brown dog")
        assert diff.is_close  # >80% similar

    def test_empty_strings(self):
        diff = compute_output_diff("i1", "", "")
        assert diff.is_match
        assert diff.similarity == 1.0

    def test_one_empty(self):
        diff = compute_output_diff("i1", "content", "")
        assert not diff.is_match
        assert diff.similarity < 0.5

    def test_none_values(self):
        diff = compute_output_diff("i1", None, None)
        assert diff.is_match  # both convert to ""

    def test_diff_lines_exist(self):
        diff = compute_output_diff("i1", "line1\nline2", "line1\nline3")
        assert len(diff.diff_lines) > 0

    def test_as_dict(self):
        diff = compute_output_diff("i1", "a", "b")
        d = diff.as_dict()
        assert "item_id" in d
        assert "similarity" in d
        assert "is_match" in d

    def test_format_text_match(self):
        diff = compute_output_diff("i1", "same", "same")
        text = diff.format_text()
        assert "MATCH" in text

    def test_format_text_diff(self):
        diff = compute_output_diff("i1", "old", "new")
        text = diff.format_text()
        assert "i1" in text


class TestDiffRunOutputs:
    def test_with_references(self):
        items = [
            DatasetItem(id="i1", input="q", output="actual",
                       human_label={"reference": "expected"}),
            DatasetItem(id="i2", input="q", output="same",
                       human_label={"reference": "same"}),
        ]
        report = diff_run_outputs(items)
        assert report.total_items == 2
        assert report.exact_matches == 1
        assert report.significant_diffs >= 0

    def test_no_references_skipped(self):
        items = [DatasetItem(id="i1", input="q", output="out")]
        report = diff_run_outputs(items)
        assert report.total_items == 0  # no reference = skipped

    def test_empty(self):
        report = diff_run_outputs([])
        assert report.total_items == 0

    def test_custom_results(self):
        items = [
            DatasetItem(id="i1", input="q", output="default",
                       human_label={"reference": "expected"}),
        ]
        report = diff_run_outputs(items, run_results={"i1": "custom_output"})
        assert report.diffs[0].actual == "custom_output"

    def test_as_dict(self):
        items = [DatasetItem(id="i1", input="q", output="a",
                            human_label={"reference": "b"})]
        d = diff_run_outputs(items).as_dict()
        assert "total_items" in d
        assert "avg_similarity" in d

    def test_format_text(self):
        items = [DatasetItem(id="i1", input="q", output="a",
                            human_label={"reference": "b"})]
        text = diff_run_outputs(items).format_text()
        assert "Output Diff" in text


# =============================================================================
# Abort conditions tests
# =============================================================================

class TestAbortController:
    def test_no_conditions_continue(self):
        controller = AbortController()
        result = controller.check({"pass_rate": 0.0})
        assert not result.should_abort

    def test_pass_rate_condition_triggers(self):
        controller = AbortController()
        controller.add_pass_rate_condition(min_rate=0.3)
        result = controller.check({"pass_rate": 0.2})
        assert result.should_abort
        assert len(result.triggered_conditions) == 1

    def test_pass_rate_condition_ok(self):
        controller = AbortController()
        controller.add_pass_rate_condition(min_rate=0.3)
        result = controller.check({"pass_rate": 0.5})
        assert not result.should_abort

    def test_safety_condition_triggers(self):
        controller = AbortController()
        controller.add_safety_condition(
            safety_metrics=["toxicity_safety"],
            min_rate=0.5,
        )
        result = controller.check({
            "metric_pass_rates": {"toxicity_safety": 0.3},
        })
        assert result.should_abort

    def test_safety_condition_ok(self):
        controller = AbortController()
        controller.add_safety_condition(
            safety_metrics=["toxicity_safety"],
            min_rate=0.5,
        )
        result = controller.check({
            "metric_pass_rates": {"toxicity_safety": 0.8},
        })
        assert not result.should_abort

    def test_error_rate_condition(self):
        controller = AbortController()
        controller.add_error_rate_condition(max_error_rate=0.1)
        result = controller.check({"error_count": 5, "items_evaluated": 20})
        assert result.should_abort  # 25% error rate > 10%

    def test_cost_condition(self):
        controller = AbortController()
        controller.add_cost_condition(max_cost_usd=1.0)
        result = controller.check({"cost_usd": 1.5})
        assert result.should_abort

    def test_multiple_conditions_any_triggers(self):
        controller = AbortController()
        controller.add_pass_rate_condition(0.5)
        controller.add_cost_condition(1.0)
        # Only cost triggers (pass rate is fine)
        result = controller.check({"pass_rate": 0.8, "cost_usd": 2.0})
        assert result.should_abort
        assert len(result.triggered_conditions) == 1

    def test_custom_condition(self):
        controller = AbortController()
        controller.add_condition(AbortCondition(
            name="custom",
            description="Custom abort rule",
            check_fn=lambda ctx: ctx.get("custom_flag", False),
        ))
        assert not controller.check({"custom_flag": False}).should_abort
        assert controller.check({"custom_flag": True}).should_abort

    def test_condition_count(self):
        controller = AbortController()
        assert controller.condition_count == 0
        controller.add_pass_rate_condition(0.3)
        controller.add_cost_condition(1.0)
        assert controller.condition_count == 2

    def test_condition_error_safe(self):
        controller = AbortController()
        controller.add_condition(AbortCondition(
            name="bad", description="crashes",
            check_fn=lambda ctx: 1 / 0,  # raises
        ))
        result = controller.check({})
        assert not result.should_abort  # error = don't abort

    def test_as_dict(self):
        controller = AbortController()
        controller.add_pass_rate_condition(0.3)
        result = controller.check({"pass_rate": 0.1})
        d = result.as_dict()
        assert d["should_abort"] is True
        assert len(d["triggered_conditions"]) == 1

    def test_format_text_abort(self):
        controller = AbortController()
        controller.add_pass_rate_condition(0.3)
        result = controller.check({"pass_rate": 0.1})
        assert "ABORT" in result.format_text()

    def test_format_text_continue(self):
        controller = AbortController()
        result = controller.check({})
        assert "CONTINUE" in result.format_text()
