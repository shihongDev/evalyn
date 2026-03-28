"""Tests for evaluation.provider_splitting module."""

from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.provider_splitting import (
    ProviderAllocation,
    SplitConfig,
    SplitPlan,
    SplitResult,
    create_split_plan,
    estimate_split_cost,
    split_capacity_weighted,
    split_cost_optimized,
    split_round_robin,
)


# -- helpers -----------------------------------------------------------------

PROVIDERS_2 = [
    {"provider": "openai", "model": "gpt-4o", "capacity": 50, "cost_per_item": 0.01},
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "capacity": 50, "cost_per_item": 0.02},
]

PROVIDERS_3 = [
    {"provider": "openai", "model": "gpt-4o", "capacity": 100, "cost_per_item": 0.03},
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "capacity": 50, "cost_per_item": 0.02},
    {"provider": "gemini", "model": "gemini-2.5-flash", "capacity": 30, "cost_per_item": 0.01},
]

ITEMS_6 = ["a", "b", "c", "d", "e", "f"]
ITEMS_10 = [f"item-{i}" for i in range(10)]


# -- ProviderAllocation ------------------------------------------------------


class TestProviderAllocation:
    def test_as_dict(self):
        a = ProviderAllocation(provider="openai", model="gpt-4o", item_count=10, estimated_cost_usd=0.10)
        d = a.as_dict()
        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4o"
        assert d["item_count"] == 10
        assert d["estimated_cost_usd"] == 0.10

    def test_defaults(self):
        a = ProviderAllocation(provider="x", model="y", item_count=5)
        assert a.estimated_cost_usd == 0.0


# -- SplitConfig -------------------------------------------------------------


class TestSplitConfig:
    def test_defaults(self):
        cfg = SplitConfig()
        assert cfg.providers == []
        assert cfg.strategy == "round_robin"

    def test_as_dict(self):
        cfg = SplitConfig(providers=PROVIDERS_2, strategy="cost_optimized")
        d = cfg.as_dict()
        assert d["strategy"] == "cost_optimized"
        assert len(d["providers"]) == 2

    def test_from_dict(self):
        d = {"providers": PROVIDERS_2, "strategy": "capacity_weighted"}
        cfg = SplitConfig.from_dict(d)
        assert cfg.strategy == "capacity_weighted"
        assert len(cfg.providers) == 2

    def test_from_dict_defaults(self):
        cfg = SplitConfig.from_dict({})
        assert cfg.providers == []
        assert cfg.strategy == "round_robin"

    def test_roundtrip(self):
        cfg = SplitConfig(providers=PROVIDERS_2, strategy="cost_optimized")
        restored = SplitConfig.from_dict(cfg.as_dict())
        assert restored.strategy == cfg.strategy
        assert len(restored.providers) == len(cfg.providers)


# -- SplitPlan ---------------------------------------------------------------


class TestSplitPlan:
    def test_as_dict(self):
        plan = SplitPlan(
            allocations=[
                ProviderAllocation(provider="openai", model="gpt-4o", item_count=5, estimated_cost_usd=0.05),
            ],
            total_items=5,
            total_estimated_cost=0.05,
        )
        d = plan.as_dict()
        assert d["total_items"] == 5
        assert len(d["allocations"]) == 1
        assert d["allocations"][0]["provider"] == "openai"

    def test_from_dict(self):
        d = {
            "allocations": [
                {"provider": "openai", "model": "gpt-4o", "item_count": 5, "estimated_cost_usd": 0.05},
            ],
            "total_items": 5,
            "total_estimated_cost": 0.05,
        }
        plan = SplitPlan.from_dict(d)
        assert plan.total_items == 5
        assert len(plan.allocations) == 1
        assert plan.allocations[0].provider == "openai"
        assert plan.allocations[0].item_count == 5

    def test_roundtrip(self):
        plan = SplitPlan(
            allocations=[
                ProviderAllocation(provider="openai", model="gpt-4o", item_count=5, estimated_cost_usd=0.05),
                ProviderAllocation(provider="anthropic", model="claude-sonnet-4-20250514", item_count=3, estimated_cost_usd=0.06),
            ],
            total_items=8,
            total_estimated_cost=0.11,
        )
        restored = SplitPlan.from_dict(plan.as_dict())
        assert restored.total_items == plan.total_items
        assert len(restored.allocations) == len(plan.allocations)
        assert restored.total_estimated_cost == pytest.approx(plan.total_estimated_cost)

    def test_format_text(self):
        plan = SplitPlan(
            allocations=[
                ProviderAllocation(provider="openai", model="gpt-4o", item_count=5, estimated_cost_usd=0.05),
            ],
            total_items=5,
            total_estimated_cost=0.05,
        )
        text = plan.format_text()
        assert "5 items" in text
        assert "openai" in text
        assert "$" in text


# -- SplitResult -------------------------------------------------------------


class TestSplitResult:
    def test_as_dict(self):
        sr = SplitResult(provider="openai", model="gpt-4o", item_ids=["a", "b"])
        d = sr.as_dict()
        assert d["provider"] == "openai"
        assert d["item_ids"] == ["a", "b"]

    def test_defaults(self):
        sr = SplitResult(provider="x", model="y")
        assert sr.item_ids == []


# -- split_round_robin -------------------------------------------------------


class TestSplitRoundRobin:
    def test_even_distribution(self):
        results = split_round_robin(ITEMS_6, PROVIDERS_2)
        assert len(results) == 2
        assert len(results[0].item_ids) == 3
        assert len(results[1].item_ids) == 3

    def test_uneven_distribution(self):
        items = ["a", "b", "c", "d", "e"]
        results = split_round_robin(items, PROVIDERS_2)
        counts = [len(r.item_ids) for r in results]
        assert sorted(counts) == [2, 3]

    def test_round_robin_order(self):
        results = split_round_robin(ITEMS_6, PROVIDERS_2)
        # Items should alternate: 0->p0, 1->p1, 2->p0, 3->p1, ...
        assert results[0].item_ids == ["a", "c", "e"]
        assert results[1].item_ids == ["b", "d", "f"]

    def test_empty_items(self):
        results = split_round_robin([], PROVIDERS_2)
        # All results should have empty item_ids
        for r in results:
            assert r.item_ids == []

    def test_empty_providers(self):
        results = split_round_robin(ITEMS_6, [])
        assert results == []

    def test_single_provider(self):
        results = split_round_robin(ITEMS_6, [PROVIDERS_2[0]])
        assert len(results) == 1
        assert len(results[0].item_ids) == 6


# -- split_cost_optimized ----------------------------------------------------


class TestSplitCostOptimized:
    def test_cheapest_first(self):
        results = split_cost_optimized(ITEMS_10, PROVIDERS_3)
        # Gemini is cheapest (0.01), capacity 30 - should get items first
        gemini = [r for r in results if r.provider == "gemini"][0]
        assert len(gemini.item_ids) == 10  # all fit in capacity 30

    def test_respects_capacity(self):
        providers = [
            {"provider": "cheap", "model": "m1", "capacity": 3, "cost_per_item": 0.01},
            {"provider": "expensive", "model": "m2", "capacity": 100, "cost_per_item": 0.10},
        ]
        results = split_cost_optimized(ITEMS_6, providers)
        cheap = [r for r in results if r.provider == "cheap"][0]
        expensive = [r for r in results if r.provider == "expensive"][0]
        assert len(cheap.item_ids) == 3
        assert len(expensive.item_ids) == 3

    def test_empty_items(self):
        results = split_cost_optimized([], PROVIDERS_2)
        assert results == []

    def test_empty_providers(self):
        results = split_cost_optimized(ITEMS_6, [])
        assert results == []

    def test_single_provider(self):
        results = split_cost_optimized(ITEMS_6, [PROVIDERS_2[0]])
        assert len(results) == 1
        assert len(results[0].item_ids) == 6


# -- split_capacity_weighted -------------------------------------------------


class TestSplitCapacityWeighted:
    def test_proportional(self):
        # Capacities: 100, 50, 30 = 180 total
        items = [f"i{i}" for i in range(18)]
        results = split_capacity_weighted(items, PROVIDERS_3)
        counts = {r.provider: len(r.item_ids) for r in results}
        # 100/180 * 18 = 10, 50/180 * 18 = 5, remainder = 3
        assert counts["openai"] == 10
        assert counts["anthropic"] == 5
        assert counts["gemini"] == 3

    def test_single_provider(self):
        results = split_capacity_weighted(ITEMS_6, [PROVIDERS_3[0]])
        assert len(results) == 1
        assert len(results[0].item_ids) == 6

    def test_empty_items(self):
        results = split_capacity_weighted([], PROVIDERS_3)
        # Empty results get filtered
        assert all(len(r.item_ids) == 0 for r in results) or len(results) == 0

    def test_zero_capacity_fallback(self):
        providers = [
            {"provider": "a", "model": "m1", "capacity": 0},
            {"provider": "b", "model": "m2", "capacity": 0},
        ]
        # Should fall back to round-robin
        results = split_capacity_weighted(ITEMS_6, providers)
        total_assigned = sum(len(r.item_ids) for r in results)
        assert total_assigned == 6


# -- create_split_plan -------------------------------------------------------


class TestCreateSplitPlan:
    def test_routes_round_robin(self):
        cfg = SplitConfig(providers=PROVIDERS_2, strategy="round_robin")
        plan = create_split_plan(ITEMS_6, cfg)
        assert plan.total_items == 6
        assert len(plan.allocations) == 2

    def test_routes_cost_optimized(self):
        cfg = SplitConfig(providers=PROVIDERS_3, strategy="cost_optimized")
        plan = create_split_plan(ITEMS_10, cfg)
        assert plan.total_items == 10

    def test_routes_capacity_weighted(self):
        cfg = SplitConfig(providers=PROVIDERS_3, strategy="capacity_weighted")
        items = [f"i{i}" for i in range(18)]
        plan = create_split_plan(items, cfg)
        assert plan.total_items == 18

    def test_plan_cost_calculation(self):
        cfg = SplitConfig(providers=PROVIDERS_2, strategy="round_robin")
        plan = create_split_plan(ITEMS_6, cfg)
        # 3 items * 0.01 + 3 items * 0.02 = 0.03 + 0.06 = 0.09
        assert plan.total_estimated_cost == pytest.approx(0.09)

    def test_plan_allocations_have_costs(self):
        cfg = SplitConfig(providers=PROVIDERS_2, strategy="round_robin")
        plan = create_split_plan(ITEMS_6, cfg)
        for alloc in plan.allocations:
            assert alloc.estimated_cost_usd >= 0.0


# -- estimate_split_cost -----------------------------------------------------


class TestEstimateSplitCost:
    def test_basic(self):
        plan = SplitPlan(
            allocations=[
                ProviderAllocation(provider="a", model="m1", item_count=5, estimated_cost_usd=0.05),
                ProviderAllocation(provider="b", model="m2", item_count=5, estimated_cost_usd=0.10),
            ],
            total_items=10,
            total_estimated_cost=0.15,
        )
        assert estimate_split_cost(plan) == pytest.approx(0.15)

    def test_empty_plan(self):
        plan = SplitPlan()
        assert estimate_split_cost(plan) == 0.0

    def test_matches_plan_total(self):
        cfg = SplitConfig(providers=PROVIDERS_2, strategy="round_robin")
        plan = create_split_plan(ITEMS_6, cfg)
        assert estimate_split_cost(plan) == pytest.approx(plan.total_estimated_cost)
