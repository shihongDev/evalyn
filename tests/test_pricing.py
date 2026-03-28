"""Tests for pricing module."""

from __future__ import annotations

from evalyn_sdk.pricing import (
    MODEL_PRICING,
    CostEstimate,
    ModelPrice,
    PricingTable,
    compare_model_costs,
    estimate_cost,
    estimate_run_cost,
    get_default_pricing,
)


# --- get_default_pricing ---


def test_get_default_pricing_has_models():
    table = get_default_pricing()
    models = table.list_models()
    assert len(models) == len(MODEL_PRICING)
    for m in MODEL_PRICING:
        assert m in models


def test_get_default_pricing_values():
    table = get_default_pricing()
    gpt4o = table.get_price("gpt-4o")
    assert gpt4o is not None
    assert gpt4o.input_per_1k_tokens == 0.0025
    assert gpt4o.output_per_1k_tokens == 0.01


def test_get_default_pricing_cached_tokens():
    table = get_default_pricing()
    gemini = table.get_price("gemini-2.5-flash")
    assert gemini is not None
    assert gemini.cached_per_1k_tokens == 0.0000375


# --- estimate_cost ---


def test_estimate_cost_basic():
    est = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=1000)
    # (1000/1000)*0.0025 + (1000/1000)*0.01 = 0.0125
    assert abs(est.estimated_cost_usd - 0.0125) < 1e-9
    assert est.model == "gpt-4o"
    assert est.input_tokens == 1000
    assert est.output_tokens == 1000


def test_estimate_cost_with_cache():
    est = estimate_cost(
        "gemini-2.5-flash",
        input_tokens=1000,
        output_tokens=1000,
        cached_tokens=2000,
    )
    # (1000/1000)*0.00015 + (1000/1000)*0.0006 + (2000/1000)*0.0000375
    expected = 0.00015 + 0.0006 + 0.0000750
    assert abs(est.estimated_cost_usd - expected) < 1e-9
    assert est.cached_tokens == 2000


def test_estimate_cost_unknown_model():
    est = estimate_cost("unknown-model", input_tokens=1000, output_tokens=1000)
    assert est.estimated_cost_usd == 0.0
    assert est.model == "unknown-model"


def test_estimate_cost_zero_tokens():
    est = estimate_cost("gpt-4o", input_tokens=0, output_tokens=0)
    assert est.estimated_cost_usd == 0.0


def test_estimate_cost_custom_pricing():
    table = PricingTable()
    table.add_price(
        "custom-model",
        ModelPrice("custom-model", input_per_1k_tokens=0.001, output_per_1k_tokens=0.002),
    )
    est = estimate_cost("custom-model", 2000, 1000, pricing=table)
    # (2000/1000)*0.001 + (1000/1000)*0.002 = 0.004
    assert abs(est.estimated_cost_usd - 0.004) < 1e-9


# --- estimate_run_cost ---


def test_estimate_run_cost_default_tokens():
    est = estimate_run_cost("gpt-4o-mini", num_items=100)
    # 100 items * 500 input * 200 output
    # total_input=50000, total_output=20000
    # (50000/1000)*0.00015 + (20000/1000)*0.0006 = 0.0075 + 0.012 = 0.0195
    assert abs(est.estimated_cost_usd - 0.0195) < 1e-9
    assert est.input_tokens == 50000
    assert est.output_tokens == 20000


def test_estimate_run_cost_custom_tokens():
    est = estimate_run_cost(
        "gpt-4o", num_items=10, avg_input_tokens=1000, avg_output_tokens=500
    )
    # total_input=10000, total_output=5000
    # (10000/1000)*0.0025 + (5000/1000)*0.01 = 0.025 + 0.05 = 0.075
    assert abs(est.estimated_cost_usd - 0.075) < 1e-9


def test_estimate_run_cost_unknown_model():
    est = estimate_run_cost("nonexistent", num_items=50)
    assert est.estimated_cost_usd == 0.0


# --- compare_model_costs ---


def test_compare_model_costs_sorted():
    models = ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"]
    results = compare_model_costs(models, input_tokens=1000, output_tokens=1000)
    assert len(results) == 3
    # Should be sorted cheapest first
    costs = [r.estimated_cost_usd for r in results]
    assert costs == sorted(costs)


def test_compare_model_costs_includes_unknown():
    models = ["gpt-4o", "nonexistent"]
    results = compare_model_costs(models, input_tokens=1000, output_tokens=1000)
    assert len(results) == 2
    # Unknown model has cost 0, so it comes first
    assert results[0].model == "nonexistent"
    assert results[0].estimated_cost_usd == 0.0


def test_compare_model_costs_single():
    results = compare_model_costs(["gpt-4o"], input_tokens=500, output_tokens=500)
    assert len(results) == 1
    assert results[0].model == "gpt-4o"


# --- PricingTable ---


def test_pricing_table_add_get():
    table = PricingTable()
    price = ModelPrice("test-model", 0.001, 0.002, 0.0005)
    table.add_price("test-model", price)
    got = table.get_price("test-model")
    assert got is not None
    assert got.model == "test-model"
    assert got.input_per_1k_tokens == 0.001


def test_pricing_table_get_missing():
    table = PricingTable()
    assert table.get_price("missing") is None


def test_pricing_table_list_models():
    table = PricingTable()
    table.add_price("b-model", ModelPrice("b-model", 0.001, 0.002))
    table.add_price("a-model", ModelPrice("a-model", 0.003, 0.004))
    assert table.list_models() == ["a-model", "b-model"]


def test_pricing_table_as_dict():
    table = PricingTable()
    table.add_price("m1", ModelPrice("m1", 0.001, 0.002, 0.0003))
    d = table.as_dict()
    assert "m1" in d["prices"]
    assert d["prices"]["m1"]["input_per_1k_tokens"] == 0.001


def test_pricing_table_from_dict():
    data = {
        "prices": {
            "m1": {
                "model": "m1",
                "input_per_1k_tokens": 0.001,
                "output_per_1k_tokens": 0.002,
                "cached_per_1k_tokens": 0.0003,
            }
        }
    }
    table = PricingTable.from_dict(data)
    assert "m1" in table.list_models()
    price = table.get_price("m1")
    assert price is not None
    assert price.cached_per_1k_tokens == 0.0003


def test_pricing_table_roundtrip():
    table = get_default_pricing()
    restored = PricingTable.from_dict(table.as_dict())
    assert table.list_models() == restored.list_models()
    for m in table.list_models():
        orig = table.get_price(m)
        rest = restored.get_price(m)
        assert orig is not None and rest is not None
        assert orig.input_per_1k_tokens == rest.input_per_1k_tokens
        assert orig.output_per_1k_tokens == rest.output_per_1k_tokens


# --- ModelPrice ---


def test_model_price_as_dict():
    p = ModelPrice("m", 0.001, 0.002, 0.0003)
    d = p.as_dict()
    assert d["model"] == "m"
    assert d["input_per_1k_tokens"] == 0.001
    assert d["output_per_1k_tokens"] == 0.002
    assert d["cached_per_1k_tokens"] == 0.0003


def test_model_price_from_dict():
    d = {"model": "m", "input_per_1k_tokens": 0.01, "output_per_1k_tokens": 0.02}
    p = ModelPrice.from_dict(d)
    assert p.model == "m"
    assert p.cached_per_1k_tokens == 0.0  # default


def test_model_price_roundtrip():
    p = ModelPrice("x", 0.005, 0.01, 0.001)
    restored = ModelPrice.from_dict(p.as_dict())
    assert p.model == restored.model
    assert p.input_per_1k_tokens == restored.input_per_1k_tokens


# --- CostEstimate ---


def test_cost_estimate_format_text():
    est = CostEstimate("gpt-4o", 1000, 500, 0, 0.0075)
    text = est.format_text()
    assert "gpt-4o" in text
    assert "1,000" in text
    assert "500" in text
    assert "$0.007500" in text
    assert "Cached" not in text


def test_cost_estimate_format_text_with_cache():
    est = CostEstimate("gemini-2.5-flash", 1000, 500, 2000, 0.001)
    text = est.format_text()
    assert "Cached tokens: 2,000" in text


def test_cost_estimate_as_dict():
    est = CostEstimate("m", 100, 200, 50, 0.123456789)
    d = est.as_dict()
    assert d["model"] == "m"
    assert d["input_tokens"] == 100
    assert d["estimated_cost_usd"] == 0.123457  # rounded to 6 decimals


# --- Edge cases ---


def test_empty_pricing_table():
    table = PricingTable()
    assert table.list_models() == []
    assert table.get_price("anything") is None


def test_estimate_cost_large_tokens():
    est = estimate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=500_000)
    # (1000000/1000)*0.0025 + (500000/1000)*0.01 = 2.5 + 5.0 = 7.5
    assert abs(est.estimated_cost_usd - 7.5) < 0.01
