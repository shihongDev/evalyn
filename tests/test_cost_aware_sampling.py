"""Tests for cost-aware sampling module."""

from __future__ import annotations

import math

import pytest

from evalyn_sdk.cost_aware_sampling import (
    CostAwareResult,
    CostBudget,
    ItemCost,
    compute_item_costs,
    estimate_item_tokens,
    format_cost_report,
    greedy_budget_selection,
    knapsack_selection,
    run_cost_aware_sampling,
)


# ---- ItemCost ----


class TestItemCost:
    def test_basic(self):
        c = ItemCost(item_id="a", estimated_tokens=100, estimated_cost=0.5)
        assert c.item_id == "a"
        assert c.estimated_tokens == 100
        assert c.estimated_cost == 0.5

    def test_as_dict(self):
        c = ItemCost(item_id="b", estimated_tokens=200, estimated_cost=1.0)
        d = c.as_dict()
        assert d == {
            "item_id": "b",
            "estimated_tokens": 200,
            "estimated_cost": 1.0,
        }

    def test_from_dict(self):
        d = {"item_id": "c", "estimated_tokens": 300, "estimated_cost": 1.5}
        c = ItemCost.from_dict(d)
        assert c.item_id == "c"
        assert c.estimated_tokens == 300

    def test_roundtrip(self):
        c = ItemCost(item_id="rt", estimated_tokens=42, estimated_cost=0.123)
        assert ItemCost.from_dict(c.as_dict()) == c


# ---- CostBudget ----


class TestCostBudget:
    def test_defaults(self):
        b = CostBudget()
        assert b.max_tokens == 0
        assert b.max_cost == 0.0
        assert b.cost_per_1k_tokens == 0.075

    def test_custom_values(self):
        b = CostBudget(max_tokens=10000, max_cost=5.0, cost_per_1k_tokens=0.1)
        assert b.max_tokens == 10000
        assert b.max_cost == 5.0

    def test_as_dict(self):
        b = CostBudget(max_tokens=5000, max_cost=2.5)
        d = b.as_dict()
        assert d == {
            "max_tokens": 5000,
            "max_cost": 2.5,
            "cost_per_1k_tokens": 0.075,
        }

    def test_from_dict_full(self):
        d = {"max_tokens": 8000, "max_cost": 3.0, "cost_per_1k_tokens": 0.05}
        b = CostBudget.from_dict(d)
        assert b.max_tokens == 8000
        assert b.cost_per_1k_tokens == 0.05

    def test_from_dict_defaults(self):
        b = CostBudget.from_dict({})
        assert b.max_tokens == 0
        assert b.max_cost == 0.0
        assert b.cost_per_1k_tokens == 0.075

    def test_roundtrip(self):
        b = CostBudget(max_tokens=1000, max_cost=1.0, cost_per_1k_tokens=0.2)
        assert CostBudget.from_dict(b.as_dict()) == b


# ---- CostAwareResult ----


class TestCostAwareResult:
    def test_defaults(self):
        r = CostAwareResult()
        assert r.selected_ids == []
        assert r.total_tokens == 0
        assert r.total_cost == 0.0
        assert r.items_selected == 0
        assert r.budget_utilization == 0.0

    def test_custom_values(self):
        r = CostAwareResult(
            selected_ids=["a", "b"],
            total_tokens=500,
            total_cost=0.5,
            items_selected=2,
            budget_utilization=0.75,
        )
        assert r.items_selected == 2

    def test_as_dict(self):
        r = CostAwareResult(
            selected_ids=["x"],
            total_tokens=100,
            total_cost=0.1,
            items_selected=1,
            budget_utilization=0.5,
        )
        d = r.as_dict()
        assert d["selected_ids"] == ["x"]
        assert d["budget_utilization"] == 0.5

    def test_from_dict(self):
        d = {
            "selected_ids": ["a"],
            "total_tokens": 200,
            "total_cost": 0.2,
            "items_selected": 1,
            "budget_utilization": 0.8,
        }
        r = CostAwareResult.from_dict(d)
        assert r.total_tokens == 200
        assert r.budget_utilization == 0.8

    def test_from_dict_defaults(self):
        r = CostAwareResult.from_dict({})
        assert r.selected_ids == []
        assert r.total_tokens == 0

    def test_roundtrip(self):
        r = CostAwareResult(
            selected_ids=["q"],
            total_tokens=999,
            total_cost=1.23,
            items_selected=1,
            budget_utilization=0.45,
        )
        r2 = CostAwareResult.from_dict(r.as_dict())
        assert r2 == r


# ---- estimate_item_tokens ----


class TestEstimateItemTokens:
    def test_empty_string(self):
        # max(1, ceil(0/4)) = 1, output = min(300, max(50, 1)) = 50, total = 51
        assert estimate_item_tokens("") == 51

    def test_short_text(self):
        # max(1, ceil(2/4)) = 1, output = min(300, max(50, 1)) = 50, total = 51
        assert estimate_item_tokens("hi") == 51

    def test_four_chars(self):
        # max(1, ceil(4/4)) = 1, output = 50, total = 51
        assert estimate_item_tokens("abcd") == 51

    def test_five_chars(self):
        # max(1, ceil(5/4)) = 2, output = 50, total = 52
        assert estimate_item_tokens("abcde") == 52

    def test_longer_text(self):
        text = "a" * 400
        # ceil(400/4) = 100, output = min(300, max(50, 100)) = 100, total = 200
        assert estimate_item_tokens(text) == 200

    def test_includes_output_overhead(self):
        # All estimates include at least 50 output tokens
        assert estimate_item_tokens("x") >= 50

    def test_scales_with_length(self):
        short = estimate_item_tokens("short")
        long = estimate_item_tokens("a" * 1000)
        assert long > short


# ---- compute_item_costs ----


class TestComputeItemCosts:
    def test_basic(self):
        items = {"a": "hello", "b": "world"}
        costs = compute_item_costs(items)
        assert len(costs) == 2
        assert all(isinstance(c, ItemCost) for c in costs)

    def test_sorted_by_id(self):
        items = {"c": "x", "a": "y", "b": "z"}
        costs = compute_item_costs(items)
        assert [c.item_id for c in costs] == ["a", "b", "c"]

    def test_cost_calculation(self):
        items = {"a": "a" * 400}  # 400 chars -> ceil(400/4)=100 input + 100 output = 200
        costs = compute_item_costs(items, cost_per_1k=0.075)
        assert costs[0].estimated_tokens == 200
        expected_cost = 200 * 0.075 / 1000
        assert abs(costs[0].estimated_cost - expected_cost) < 1e-9

    def test_custom_cost_per_1k(self):
        items = {"a": "a" * 400}
        costs = compute_item_costs(items, cost_per_1k=0.1)
        expected_cost = 200 * 0.1 / 1000
        assert abs(costs[0].estimated_cost - expected_cost) < 1e-9

    def test_empty_items(self):
        assert compute_item_costs({}) == []


# ---- greedy_budget_selection ----


class TestGreedyBudgetSelection:
    def test_zero_budget_selects_none(self):
        costs = [
            ItemCost(item_id="a", estimated_tokens=100, estimated_cost=0.1),
            ItemCost(item_id="b", estimated_tokens=200, estimated_cost=0.2),
        ]
        budget = CostBudget()  # zero = no capacity
        result = greedy_budget_selection(costs, budget)
        assert result == []

    def test_large_budget_selects_all(self):
        costs = [
            ItemCost(item_id="a", estimated_tokens=100, estimated_cost=0.1),
            ItemCost(item_id="b", estimated_tokens=200, estimated_cost=0.2),
        ]
        budget = CostBudget(max_tokens=999999, max_cost=999999.0)
        result = greedy_budget_selection(costs, budget)
        assert set(result) == {"a", "b"}

    def test_token_budget(self):
        costs = [
            ItemCost(item_id="a", estimated_tokens=100, estimated_cost=0.01),
            ItemCost(item_id="b", estimated_tokens=200, estimated_cost=0.02),
            ItemCost(item_id="c", estimated_tokens=300, estimated_cost=0.03),
        ]
        budget = CostBudget(max_tokens=250)
        result = greedy_budget_selection(costs, budget)
        # Sorted by cost: a(0.01), b(0.02), c(0.03)
        # a: 100 tokens, b: +200 = 300 > 250, c: +300 > 250
        assert result == ["a"]

    def test_cost_budget(self):
        costs = [
            ItemCost(item_id="a", estimated_tokens=100, estimated_cost=0.5),
            ItemCost(item_id="b", estimated_tokens=100, estimated_cost=0.3),
            ItemCost(item_id="c", estimated_tokens=100, estimated_cost=0.4),
        ]
        budget = CostBudget(max_cost=0.8)
        result = greedy_budget_selection(costs, budget)
        # Sorted by cost: b(0.3), c(0.4), a(0.5)
        # b: 0.3, b+c: 0.7, b+c+a: 1.2 > 0.8
        assert result == ["b", "c"]

    def test_prefers_cheaper_items(self):
        costs = [
            ItemCost(item_id="expensive", estimated_tokens=1000, estimated_cost=10.0),
            ItemCost(item_id="cheap1", estimated_tokens=100, estimated_cost=0.1),
            ItemCost(item_id="cheap2", estimated_tokens=100, estimated_cost=0.1),
        ]
        budget = CostBudget(max_cost=0.5)
        result = greedy_budget_selection(costs, budget)
        assert "expensive" not in result
        assert "cheap1" in result
        assert "cheap2" in result

    def test_empty_costs(self):
        assert greedy_budget_selection([], CostBudget()) == []

    def test_both_budgets(self):
        costs = [
            ItemCost(item_id="a", estimated_tokens=500, estimated_cost=0.1),
            ItemCost(item_id="b", estimated_tokens=500, estimated_cost=0.1),
            ItemCost(item_id="c", estimated_tokens=500, estimated_cost=0.1),
        ]
        budget = CostBudget(max_tokens=900, max_cost=0.25)
        result = greedy_budget_selection(costs, budget)
        # Cost-wise: 0.1 + 0.1 = 0.2 ok, 0.1+0.1+0.1=0.3 > 0.25
        # Token-wise: 500+500=1000 > 900
        # Both limits prevent 2 items: token limit hits at 2
        assert len(result) == 1


# ---- knapsack_selection ----


class TestKnapsackSelection:
    def test_default_value_fn(self):
        costs = [
            ItemCost(item_id="a", estimated_tokens=100, estimated_cost=0.1),
            ItemCost(item_id="b", estimated_tokens=200, estimated_cost=0.2),
        ]
        budget = CostBudget()  # unlimited
        result = knapsack_selection(costs, budget)
        assert set(result) == {"a", "b"}

    def test_custom_value_fn(self):
        costs = [
            ItemCost(item_id="low_value", estimated_tokens=100, estimated_cost=0.5),
            ItemCost(item_id="high_value", estimated_tokens=100, estimated_cost=0.5),
        ]
        # high_value is worth 10, low_value worth 1
        value_fn = lambda c: 10.0 if c.item_id == "high_value" else 1.0
        budget = CostBudget(max_cost=0.6)
        result = knapsack_selection(costs, budget, value_fn=value_fn)
        # high_value has ratio 10/0.5=20, low_value has 1/0.5=2
        assert result == ["high_value"]

    def test_zero_cost_items_first(self):
        costs = [
            ItemCost(item_id="a", estimated_tokens=0, estimated_cost=0.0),
            ItemCost(item_id="b", estimated_tokens=100, estimated_cost=0.1),
        ]
        budget = CostBudget()
        result = knapsack_selection(costs, budget)
        # Zero-cost item has infinite ratio, selected first
        assert result[0] == "a"

    def test_budget_constraint(self):
        costs = [
            ItemCost(item_id="a", estimated_tokens=100, estimated_cost=1.0),
            ItemCost(item_id="b", estimated_tokens=100, estimated_cost=1.0),
            ItemCost(item_id="c", estimated_tokens=100, estimated_cost=1.0),
        ]
        budget = CostBudget(max_cost=2.0)
        result = knapsack_selection(costs, budget)
        assert len(result) == 2

    def test_empty_costs(self):
        assert knapsack_selection([], CostBudget()) == []

    def test_maximizes_value(self):
        # Cheaper items have higher value-to-cost ratio with default fn
        costs = [
            ItemCost(item_id="cheap", estimated_tokens=50, estimated_cost=0.01),
            ItemCost(item_id="expensive", estimated_tokens=500, estimated_cost=1.0),
        ]
        budget = CostBudget(max_cost=0.5)
        result = knapsack_selection(costs, budget)
        assert "cheap" in result
        assert "expensive" not in result


# ---- run_cost_aware_sampling ----


class TestRunCostAwareSampling:
    def test_basic_pipeline(self):
        items = {"a": "short", "b": "a longer text here", "c": "medium text"}
        budget = CostBudget(max_tokens=999999, max_cost=999999.0)
        result = run_cost_aware_sampling(items, budget)
        assert isinstance(result, CostAwareResult)
        assert result.items_selected == 3
        assert len(result.selected_ids) == 3

    def test_total_tokens_computed(self):
        items = {"a": "a" * 400}
        budget = CostBudget(max_tokens=999999, max_cost=999999.0)
        result = run_cost_aware_sampling(items, budget)
        assert result.total_tokens == 200  # ceil(400/4)=100 + output=100

    def test_total_cost_computed(self):
        items = {"a": "a" * 400}
        budget = CostBudget(max_tokens=999999, max_cost=999999.0, cost_per_1k_tokens=0.1)
        result = run_cost_aware_sampling(items, budget)
        expected = 200 * 0.1 / 1000
        assert abs(result.total_cost - expected) < 1e-9

    def test_budget_limits_selection(self):
        items = {str(i): "x" * 100 for i in range(100)}
        # Each item: ceil(100/4)=25, output=min(300,max(50,25))=50, total=75 tokens
        # 75 * 0.075/1000 = 0.005625 per item
        budget = CostBudget(max_cost=0.012)  # fits ~2 items
        result = run_cost_aware_sampling(items, budget)
        assert result.items_selected == 2

    def test_budget_utilization_token_bound(self):
        items = {"a": "a" * 400}  # 200 tokens (100 input + 100 output)
        budget = CostBudget(max_tokens=1000)
        result = run_cost_aware_sampling(items, budget)
        assert abs(result.budget_utilization - 0.2) < 1e-9

    def test_budget_utilization_cost_bound(self):
        items = {"a": "a" * 400}  # 200 tokens, cost = 200*0.075/1000 = 0.015
        budget = CostBudget(max_cost=0.06)
        result = run_cost_aware_sampling(items, budget)
        assert abs(result.budget_utilization - 0.25) < 1e-9

    def test_zero_budget_zero_utilization(self):
        items = {"a": "text"}
        budget = CostBudget()
        result = run_cost_aware_sampling(items, budget)
        assert result.budget_utilization == 0.0

    def test_empty_items(self):
        result = run_cost_aware_sampling({}, CostBudget())
        assert result.items_selected == 0
        assert result.total_tokens == 0
        assert result.total_cost == 0.0


# ---- format_cost_report ----


class TestFormatCostReport:
    def test_contains_header(self):
        result = CostAwareResult()
        report = format_cost_report(result)
        assert "Cost-Aware Sampling Report" in report

    def test_contains_stats(self):
        result = CostAwareResult(
            selected_ids=["a"],
            total_tokens=500,
            total_cost=0.05,
            items_selected=1,
            budget_utilization=0.75,
        )
        report = format_cost_report(result)
        assert "Items selected: 1" in report
        assert "Total tokens: 500" in report
        assert "$0.0500" in report
        assert "75.00%" in report

    def test_lists_selected_ids(self):
        result = CostAwareResult(selected_ids=["item1", "item2"])
        report = format_cost_report(result)
        assert "item1" in report
        assert "item2" in report

    def test_empty_result(self):
        result = CostAwareResult()
        report = format_cost_report(result)
        assert "Items selected: 0" in report
        assert "Total tokens: 0" in report
