"""Tests for what-if simulator and metric template variables."""

import pytest
from evalyn_sdk.models import DatasetItem
from evalyn_sdk.analysis.what_if import (
    simulate_improvement, find_minimum_improvement,
    find_best_metric_to_improve, WhatIfScenario,
)
from evalyn_sdk.metrics.template_vars import (
    resolve_template, extract_variables, validate_template,
)


# =============================================================================
# What-if simulator tests
# =============================================================================

class TestSimulateImprovement:
    def test_single_metric(self):
        rates = {"a": 0.7, "b": 0.8}
        scenario = simulate_improvement(rates, {"a": 0.1})
        assert scenario.projected_pass_rate > scenario.current_pass_rate
        assert scenario.delta > 0

    def test_multiple_metrics(self):
        rates = {"a": 0.5, "b": 0.6, "c": 0.7}
        scenario = simulate_improvement(rates, {"a": 0.2, "b": 0.1})
        assert scenario.projected_pass_rate > scenario.current_pass_rate

    def test_no_improvement(self):
        rates = {"a": 0.8}
        scenario = simulate_improvement(rates, {})
        assert scenario.delta == 0.0

    def test_capped_at_100(self):
        rates = {"a": 0.95}
        scenario = simulate_improvement(rates, {"a": 0.2})
        # a should be capped at 1.0, not 1.15
        assert scenario.projected_pass_rate <= 1.0

    def test_negative_improvement(self):
        rates = {"a": 0.8}
        scenario = simulate_improvement(rates, {"a": -0.3})
        assert scenario.delta < 0

    def test_as_dict(self):
        rates = {"a": 0.7}
        d = simulate_improvement(rates, {"a": 0.1}).as_dict()
        assert "projected_pass_rate" in d
        assert "delta" in d

    def test_format_text(self):
        rates = {"a": 0.7}
        text = simulate_improvement(rates, {"a": 0.1}, "test").format_text()
        assert "What-If" in text
        assert "test" in text


class TestFindMinimumImprovement:
    def test_finds_improvement(self):
        rates = {"a": 0.6, "b": 0.8}
        result = find_minimum_improvement(rates, target_pass_rate=0.8, metric_id="a")
        assert result is not None
        assert result.projected_pass_rate >= 0.8

    def test_already_at_target(self):
        rates = {"a": 0.9, "b": 0.95}
        result = find_minimum_improvement(rates, target_pass_rate=0.8, metric_id="a")
        assert result is not None
        assert result.delta == 0.0

    def test_not_achievable(self):
        # Even 100% on 'a' can't reach 99% overall if 'b' is at 50%
        rates = {"a": 0.5, "b": 0.5}
        result = find_minimum_improvement(rates, target_pass_rate=0.99, metric_id="a")
        assert result is None

    def test_unknown_metric(self):
        rates = {"a": 0.5}
        result = find_minimum_improvement(rates, target_pass_rate=0.8, metric_id="nonexistent")
        assert result is None

    def test_empty_rates(self):
        result = find_minimum_improvement({}, target_pass_rate=0.8, metric_id="a")
        assert result is None


class TestFindBestMetric:
    def test_finds_best(self):
        rates = {"low": 0.3, "high": 0.9}
        best = find_best_metric_to_improve(rates, improvement=0.1)
        assert best is not None
        # Improving the low metric should give bigger overall boost
        assert "low" in best.description

    def test_empty(self):
        assert find_best_metric_to_improve({}) is None

    def test_single_metric(self):
        best = find_best_metric_to_improve({"only": 0.5})
        assert best is not None
        assert "only" in best.description


# =============================================================================
# Template variables tests
# =============================================================================

class TestResolveTemplate:
    def test_builtin_input(self):
        item = DatasetItem(id="i1", input="hello world", output="response")
        result = resolve_template("Given: {{input}}", item)
        assert result == "Given: hello world"

    def test_builtin_output(self):
        item = DatasetItem(id="i1", input="q", output="answer here")
        result = resolve_template("Output: {{output}}", item)
        assert result == "Output: answer here"

    def test_builtin_item_id(self):
        item = DatasetItem(id="test-123", input="q", output="a")
        result = resolve_template("Item: {{item_id}}", item)
        assert result == "Item: test-123"

    def test_metadata_variable(self):
        item = DatasetItem(id="i1", input="q", output="a", metadata={"domain": "healthcare"})
        result = resolve_template("Domain: {{metadata.domain}}", item)
        assert result == "Domain: healthcare"

    def test_custom_variable(self):
        result = resolve_template(
            "Persona: {{persona}}",
            custom_vars={"persona": "expert clinician"},
        )
        assert result == "Persona: expert clinician"

    def test_multiple_variables(self):
        item = DatasetItem(id="i1", input="question", output="answer")
        result = resolve_template("Q: {{input}} A: {{output}}", item)
        assert result == "Q: question A: answer"

    def test_no_variables(self):
        assert resolve_template("plain text") == "plain text"

    def test_unresolved_kept(self):
        result = resolve_template("{{unknown_var}}")
        assert "{{unknown_var}}" in result

    def test_dict_input(self):
        item = DatasetItem(id="i1", input={"query": "test"}, output="a")
        result = resolve_template("Input: {{input}}", item)
        assert "query" in result  # JSON serialized


class TestExtractVariables:
    def test_basic(self):
        vars = extract_variables("{{input}} and {{output}}")
        assert vars == ["input", "output"]

    def test_nested(self):
        vars = extract_variables("{{metadata.domain}}")
        assert vars == ["metadata.domain"]

    def test_no_variables(self):
        assert extract_variables("plain text") == []

    def test_duplicates_removed(self):
        vars = extract_variables("{{x}} and {{x}}")
        assert vars == ["x"]


class TestValidateTemplate:
    def test_valid_builtins(self):
        result = validate_template("{{input}} -> {{output}}")
        assert result["valid"] is True
        assert len(result["builtin"]) == 2

    def test_unresolved(self):
        result = validate_template("{{unknown_var}}")
        assert result["valid"] is False
        assert "unknown_var" in result["unresolved"]

    def test_custom_vars_resolve(self):
        result = validate_template("{{persona}}", available_vars={"persona": "expert"})
        assert result["valid"] is True
        assert "persona" in result["custom"]

    def test_mixed(self):
        result = validate_template(
            "{{input}} as {{persona}} checking {{unknown}}",
            available_vars={"persona": "x"},
        )
        assert not result["valid"]
        assert "unknown" in result["unresolved"]
        assert "input" in result["builtin"]
