"""Tests for metric weighting profiles and custom metric DSL."""

import pytest
from datetime import datetime, timezone
from evalyn_sdk.analysis.weighting import (
    WeightProfile,
    compute_weighted_pass_rate,
    compute_weighted_score,
    get_weight_profile,
    SAFETY_FIRST,
    EQUAL_WEIGHTS,
    BUILTIN_WEIGHT_PROFILES,
)
from evalyn_sdk.metrics.custom_dsl import (
    build_custom_objective_metric,
    build_custom_subjective_spec,
    load_custom_metrics_from_config,
    _parse_expression,
)
from evalyn_sdk.models import DatasetItem, FunctionCall, MetricResult


def _call(output="test output"):
    now = datetime.now(timezone.utc)
    return FunctionCall(
        id="c1", function_name="test", inputs={}, output=output,
        error=None, started_at=now, ended_at=now, duration_ms=10, session_id=None,
    )


def _item(input_text="test input"):
    return DatasetItem(id="i1", input=input_text, output="test output")


# =============================================================================
# Weighting tests
# =============================================================================

class TestWeightProfile:
    def test_get_weight_listed(self):
        p = WeightProfile(name="test", weights={"help": 2.0})
        assert p.get_weight("help") == 2.0

    def test_get_weight_default(self):
        p = WeightProfile(name="test", weights={"help": 2.0}, default_weight=1.0)
        assert p.get_weight("unlisted") == 1.0

    def test_as_dict_roundtrip(self):
        p = WeightProfile(name="test", weights={"a": 2.0}, description="desc")
        d = p.as_dict()
        restored = WeightProfile.from_dict(d)
        assert restored.name == "test"
        assert restored.weights == {"a": 2.0}
        assert restored.description == "desc"


class TestWeightedPassRate:
    def test_equal_weights(self):
        rates = {"a": 0.8, "b": 0.6}
        result = compute_weighted_pass_rate(rates, EQUAL_WEIGHTS)
        assert result == pytest.approx(0.7)

    def test_weighted(self):
        rates = {"safety": 0.5, "quality": 1.0}
        profile = WeightProfile(name="test", weights={"safety": 3.0, "quality": 1.0})
        result = compute_weighted_pass_rate(rates, profile)
        assert result == pytest.approx(0.625)

    def test_empty(self):
        assert compute_weighted_pass_rate({}, EQUAL_WEIGHTS) == 0.0

    def test_safety_first_profile(self):
        rates = {"toxicity_safety": 0.5, "helpfulness": 1.0}
        result = compute_weighted_pass_rate(rates, SAFETY_FIRST)
        assert result == pytest.approx(0.625)


class TestWeightedScore:
    def test_basic(self):
        scores = {"a": 0.9, "b": 0.3}
        profile = WeightProfile(name="test", weights={"a": 2.0, "b": 1.0})
        result = compute_weighted_score(scores, profile)
        assert result == pytest.approx(0.7)


class TestGetWeightProfile:
    def test_builtin(self):
        p = get_weight_profile("equal")
        assert p.name == "equal"

    def test_safety_first(self):
        p = get_weight_profile("safety-first")
        assert p.get_weight("toxicity_safety") == 3.0

    def test_custom(self):
        customs = {"my-weights": {"weights": {"a": 5.0}}}
        p = get_weight_profile("my-weights", custom_profiles=customs)
        assert p.get_weight("a") == 5.0

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_weight_profile("nonexistent")

    def test_three_builtin(self):
        assert len(BUILTIN_WEIGHT_PROFILES) == 3


# =============================================================================
# Custom DSL tests
# =============================================================================

class TestCustomObjectiveMetric:
    def test_length_check(self):
        m = build_custom_objective_metric("short", 'len(output) < 500')
        result = m.evaluate(_call("short text"), _item())
        assert result.passed is True
        assert result.score == 1.0

    def test_length_check_fails(self):
        m = build_custom_objective_metric("short", 'len(output) < 5')
        result = m.evaluate(_call("long output text"), _item())
        assert result.passed is False

    def test_keyword_in_output(self):
        m = build_custom_objective_metric("has_hello", '"hello" in output')
        assert m.evaluate(_call("hello world"), _item()).passed is True
        assert m.evaluate(_call("goodbye"), _item()).passed is False

    def test_keyword_not_in_output(self):
        m = build_custom_objective_metric("no_error", '"error" not in output')
        assert m.evaluate(_call("success"), _item()).passed is True
        assert m.evaluate(_call("error occurred"), _item()).passed is False

    def test_keyword_in_input(self):
        m = build_custom_objective_metric("has_refund", '"refund" in input')
        assert m.evaluate(_call(), _item("I want a refund")).passed is True
        assert m.evaluate(_call(), _item("hello")).passed is False

    def test_output_equals(self):
        m = build_custom_objective_metric("exact", 'output == "yes"')
        assert m.evaluate(_call("yes"), _item()).passed is True
        assert m.evaluate(_call("no"), _item()).passed is False

    def test_output_not_equals(self):
        m = build_custom_objective_metric("not_empty", 'output != ""')
        assert m.evaluate(_call("content"), _item()).passed is True

    def test_output_length_shorthand(self):
        m = build_custom_objective_metric("len_check", 'output_length > 3')
        assert m.evaluate(_call("long"), _item()).passed is True
        assert m.evaluate(_call("ab"), _item()).passed is False

    def test_input_length_shorthand(self):
        m = build_custom_objective_metric("inp_check", 'input_length >= 5')
        assert m.evaluate(_call(), _item("hello")).passed is True

    def test_case_insensitive_keyword(self):
        m = build_custom_objective_metric("case", '"HELLO" in output')
        assert m.evaluate(_call("Hello World"), _item()).passed is True

    def test_invalid_expression_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            build_custom_objective_metric("bad", 'random garbage !!! ???')

    def test_details_include_expression(self):
        m = build_custom_objective_metric("test", 'len(output) < 100')
        result = m.evaluate(_call("short"), _item())
        assert result.details["expression"] == 'len(output) < 100'


class TestCustomSubjectiveSpec:
    def test_basic_spec(self):
        spec = build_custom_subjective_spec(
            "my_judge", "Rate the quality of {{output}}"
        )
        assert spec.id == "my_judge"
        assert spec.type == "subjective"
        assert "{{output}}" in spec.config["prompt"]

    def test_with_rubric(self):
        spec = build_custom_subjective_spec(
            "quality", "Rate quality",
            rubric=["Must be accurate", "Must be helpful"],
        )
        assert spec.config["rubric"] == ["Must be accurate", "Must be helpful"]

    def test_with_threshold(self):
        spec = build_custom_subjective_spec("q", "prompt", threshold=0.7)
        assert spec.config["threshold"] == 0.7


class TestLoadFromConfig:
    def test_load_objectives(self):
        config = {
            "custom_metrics": [
                {"id": "short_output", "type": "objective", "expression": "len(output) < 500"},
                {"id": "has_json", "type": "objective", "expression": '"json" in output'},
            ]
        }
        metrics = load_custom_metrics_from_config(config)
        assert len(metrics) == 2
        assert metrics[0].spec.id == "short_output"

    def test_skip_subjective(self):
        config = {
            "custom_metrics": [
                {"id": "judge", "type": "subjective", "prompt": "check quality"},
            ]
        }
        metrics = load_custom_metrics_from_config(config)
        assert len(metrics) == 0

    def test_skip_invalid(self):
        config = {
            "custom_metrics": [
                {"id": "bad", "type": "objective", "expression": "completely invalid!!!"},
            ]
        }
        metrics = load_custom_metrics_from_config(config)
        assert len(metrics) == 0

    def test_empty_config(self):
        assert load_custom_metrics_from_config({}) == []

    def test_missing_id_skipped(self):
        config = {"custom_metrics": [{"type": "objective", "expression": "len(output) < 5"}]}
        assert load_custom_metrics_from_config(config) == []


class TestParseExpressionSafety:
    def test_all_comparison_operators(self):
        for op in ["<", "<=", ">", ">=", "==", "!="]:
            fn = _parse_expression(f"len(output) {op} 100")
            assert callable(fn)

    def test_dangerous_strings_rejected(self):
        dangerous = [
            "1 + 1",
            "open('file')",
            "lambda: None",
            "True and False",
            "x = 5",
        ]
        for expr in dangerous:
            with pytest.raises(ValueError):
                _parse_expression(expr)
