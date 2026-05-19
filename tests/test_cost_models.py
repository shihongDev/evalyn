"""Tests for custom cost models."""

import pytest
from evalyn_sdk.trace.instrumentation.providers._shared import (
    register_cost_model,
    clear_custom_cost_models,
    list_custom_cost_models,
    _match_model_costs,
    calculate_cost,
    is_model_pricing_known,
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

