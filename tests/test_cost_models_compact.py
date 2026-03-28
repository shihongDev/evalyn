"""Tests for custom cost models and compact output mode."""

import pytest
from evalyn_sdk.trace.instrumentation.providers._shared import (
    register_cost_model,
    clear_custom_cost_models,
    list_custom_cost_models,
    _match_model_costs,
    calculate_cost,
    is_model_pricing_known,
)
from evalyn_sdk.cli.utils.compact import (
    format_run_compact,
    format_compare_compact,
    format_summary_compact,
)


class TestCustomCostModels:
    """Test custom cost model registration."""

    def setup_method(self):
        clear_custom_cost_models()

    def teardown_method(self):
        clear_custom_cost_models()

    def test_register_and_lookup(self):
        register_cost_model("my-model", input_cost=0.5, output_cost=1.0)
        costs = _match_model_costs("my-model")
        assert costs is not None
        assert costs["input"] == 0.5
        assert costs["output"] == 1.0

    def test_custom_overrides_builtin(self):
        # Register a custom price for a model that has built-in pricing
        register_cost_model("gpt-4o", input_cost=0.0, output_cost=0.0)
        costs = _match_model_costs("gpt-4o")
        assert costs["input"] == 0.0  # custom overrides built-in

    def test_case_insensitive(self):
        register_cost_model("My-Custom-Model", input_cost=1.0, output_cost=2.0)
        costs = _match_model_costs("my-custom-model")
        assert costs is not None

    def test_with_cache_costs(self):
        register_cost_model("cached-model", input_cost=1.0, output_cost=2.0,
                          cache_write=1.5, cache_read=0.2)
        costs = _match_model_costs("cached-model")
        assert costs["cache_write"] == 1.5
        assert costs["cache_read"] == 0.2

    def test_zero_cost_for_local(self):
        register_cost_model("ollama/llama3", input_cost=0.0, output_cost=0.0)
        cost = calculate_cost("ollama/llama3", 10000, 5000)
        assert cost == 0.0

    def test_calculate_with_custom(self):
        register_cost_model("my-model", input_cost=2.0, output_cost=4.0)
        cost = calculate_cost("my-model", 1_000_000, 500_000)
        # 1M * $2/1M + 0.5M * $4/1M = $2 + $2 = $4
        assert cost == pytest.approx(4.0)

    def test_is_known_after_register(self):
        assert not is_model_pricing_known("my-unknown-model")
        register_cost_model("my-unknown-model", 1.0, 2.0)
        assert is_model_pricing_known("my-unknown-model")

    def test_clear(self):
        register_cost_model("temp", 1.0, 2.0)
        assert list_custom_cost_models() != {}
        clear_custom_cost_models()
        assert list_custom_cost_models() == {}

    def test_list(self):
        register_cost_model("a", 1.0, 2.0)
        register_cost_model("b", 3.0, 4.0)
        models = list_custom_cost_models()
        assert len(models) == 2
        assert "a" in models
        assert "b" in models

    def test_builtin_still_works(self):
        # Built-in models should still resolve
        costs = _match_model_costs("gpt-4o-mini")
        assert costs is not None
        assert costs["input"] > 0


class TestCompactOutput:
    """Test compact output formatting."""

    def test_basic_run(self):
        line = format_run_compact("abc12345-full-uuid", 0.85, 17, 20)
        assert "RUN abc12345" in line
        assert "PASS" in line
        assert "85%" in line
        assert "17/20" in line

    def test_failing_run(self):
        line = format_run_compact("run-id", 0.3, 6, 20)
        assert "FAIL" in line
        assert "30%" in line

    def test_with_cost(self):
        line = format_run_compact("run-id", 0.85, 17, 20, cost_usd=0.1234)
        assert "COST $0.1234" in line

    def test_with_time_seconds(self):
        line = format_run_compact("run-id", 0.85, 17, 20, duration_seconds=45)
        assert "TIME 45s" in line

    def test_with_time_minutes(self):
        line = format_run_compact("run-id", 0.85, 17, 20, duration_seconds=120)
        assert "TIME 2.0m" in line

    def test_with_name(self):
        line = format_run_compact("run-id", 0.85, 17, 20, name="experiment-v3")
        assert "(experiment-v3)" in line

    def test_zero_cost_omitted(self):
        line = format_run_compact("run-id", 0.85, 17, 20, cost_usd=0.0)
        assert "COST" not in line

    def test_zero_time_omitted(self):
        line = format_run_compact("run-id", 0.85, 17, 20, duration_seconds=0.0)
        assert "TIME" not in line


class TestCompactCompare:
    """Test compact comparison formatting."""

    def test_improvement(self):
        line = format_compare_compact("helpfulness", 0.80, 0.90, 0.10)
        assert "helpfulness" in line
        assert "80%" in line
        assert "90%" in line
        assert "+10.0%" in line
        assert "OK" in line

    def test_regression(self):
        line = format_compare_compact("safety", 0.95, 0.80, -0.15)
        assert "REGRESSION" in line
        assert "-15.0%" in line

    def test_no_change(self):
        line = format_compare_compact("json_valid", 0.90, 0.90, 0.0)
        assert "OK" in line
        assert "+0.0%" in line

    def test_small_regression_ok(self):
        line = format_compare_compact("m", 0.85, 0.82, -0.03)
        assert "OK" in line  # -3% is within 5% tolerance


class TestCompactSummary:
    """Test compact multi-metric summary."""

    def test_basic(self):
        summary = format_summary_compact("run-123", {
            "helpfulness": {"pass_rate": 0.85, "count": 20},
            "json_valid": {"pass_rate": 1.0, "count": 20},
        })
        assert "RUN run-123" in summary
        assert "helpfulness: 85%" in summary
        assert "json_valid: 100%" in summary

    def test_with_name(self):
        summary = format_summary_compact("run-123", {}, name="test-run")
        assert "(test-run)" in summary

    def test_score_only_metric(self):
        summary = format_summary_compact("r", {
            "latency": {"avg_score": 123.5, "count": 10},
        })
        assert "avg=123.50" in summary

    def test_empty_metrics(self):
        summary = format_summary_compact("r", {})
        assert "RUN r" in summary
