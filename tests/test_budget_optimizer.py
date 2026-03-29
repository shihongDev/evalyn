"""Tests for the simulation budget optimizer module."""

from __future__ import annotations

import pytest

from evalyn_sdk.budget_optimizer import (
    DEFAULT_MODES,
    BudgetAllocation,
    BudgetPlan,
    BudgetReport,
    SimulationMode,
    compute_plan_diversity,
    estimate_mode_cost,
    format_budget_plan,
    format_budget_report,
    optimize_budget,
    update_budget_report,
)


# ---------------------------------------------------------------------------
# SimulationMode
# ---------------------------------------------------------------------------


class TestSimulationMode:
    def test_create_basic(self):
        mode = SimulationMode(
            mode_id="test",
            name="Test",
            description="A test mode",
            estimated_tokens_per_item=100,
        )
        assert mode.mode_id == "test"
        assert mode.diversity_weight == 1.0

    def test_create_with_weight(self):
        mode = SimulationMode(
            mode_id="x",
            name="X",
            description="desc",
            estimated_tokens_per_item=50,
            diversity_weight=2.5,
        )
        assert mode.diversity_weight == 2.5

    def test_as_dict(self):
        mode = SimulationMode(
            mode_id="m1",
            name="Mode 1",
            description="first",
            estimated_tokens_per_item=200,
            diversity_weight=1.5,
        )
        d = mode.as_dict()
        assert d["mode_id"] == "m1"
        assert d["name"] == "Mode 1"
        assert d["description"] == "first"
        assert d["estimated_tokens_per_item"] == 200
        assert d["diversity_weight"] == 1.5

    def test_from_dict(self):
        data = {
            "mode_id": "m2",
            "name": "Mode 2",
            "description": "second",
            "estimated_tokens_per_item": 300,
            "diversity_weight": 0.8,
        }
        mode = SimulationMode.from_dict(data)
        assert mode.mode_id == "m2"
        assert mode.estimated_tokens_per_item == 300
        assert mode.diversity_weight == 0.8

    def test_from_dict_default_weight(self):
        data = {
            "mode_id": "m3",
            "name": "Mode 3",
            "description": "third",
            "estimated_tokens_per_item": 100,
        }
        mode = SimulationMode.from_dict(data)
        assert mode.diversity_weight == 1.0

    def test_roundtrip(self):
        mode = SimulationMode(
            mode_id="rt",
            name="Round",
            description="trip",
            estimated_tokens_per_item=123,
            diversity_weight=0.7,
        )
        assert SimulationMode.from_dict(mode.as_dict()) == mode


# ---------------------------------------------------------------------------
# BudgetAllocation
# ---------------------------------------------------------------------------


class TestBudgetAllocation:
    def test_create_basic(self):
        alloc = BudgetAllocation(
            mode_id="a", item_count=5, estimated_tokens=1000
        )
        assert alloc.actual_tokens == 0

    def test_as_dict(self):
        alloc = BudgetAllocation(
            mode_id="a", item_count=5, estimated_tokens=1000, actual_tokens=950
        )
        d = alloc.as_dict()
        assert d["mode_id"] == "a"
        assert d["item_count"] == 5
        assert d["estimated_tokens"] == 1000
        assert d["actual_tokens"] == 950

    def test_from_dict(self):
        data = {
            "mode_id": "b",
            "item_count": 3,
            "estimated_tokens": 600,
            "actual_tokens": 580,
        }
        alloc = BudgetAllocation.from_dict(data)
        assert alloc.mode_id == "b"
        assert alloc.actual_tokens == 580

    def test_from_dict_default_actual(self):
        data = {"mode_id": "c", "item_count": 1, "estimated_tokens": 100}
        alloc = BudgetAllocation.from_dict(data)
        assert alloc.actual_tokens == 0

    def test_roundtrip(self):
        alloc = BudgetAllocation(
            mode_id="rt", item_count=7, estimated_tokens=700, actual_tokens=720
        )
        assert BudgetAllocation.from_dict(alloc.as_dict()) == alloc


# ---------------------------------------------------------------------------
# BudgetPlan
# ---------------------------------------------------------------------------


class TestBudgetPlan:
    def _make_plan(self):
        return BudgetPlan(
            allocations=[
                BudgetAllocation(mode_id="a", item_count=2, estimated_tokens=400),
                BudgetAllocation(mode_id="b", item_count=3, estimated_tokens=900),
            ],
            total_budget=2000,
            estimated_total_tokens=1300,
            estimated_total_items=5,
            diversity_score=0.85,
        )

    def test_as_dict(self):
        plan = self._make_plan()
        d = plan.as_dict()
        assert len(d["allocations"]) == 2
        assert d["total_budget"] == 2000
        assert d["diversity_score"] == 0.85

    def test_from_dict(self):
        plan = self._make_plan()
        restored = BudgetPlan.from_dict(plan.as_dict())
        assert restored.total_budget == plan.total_budget
        assert len(restored.allocations) == 2
        assert restored.diversity_score == plan.diversity_score

    def test_roundtrip(self):
        plan = self._make_plan()
        restored = BudgetPlan.from_dict(plan.as_dict())
        assert restored == plan


# ---------------------------------------------------------------------------
# BudgetReport
# ---------------------------------------------------------------------------


class TestBudgetReport:
    def _make_report(self):
        plan = BudgetPlan(
            allocations=[
                BudgetAllocation(
                    mode_id="a", item_count=2, estimated_tokens=400, actual_tokens=380
                ),
            ],
            total_budget=1000,
            estimated_total_tokens=400,
            estimated_total_items=2,
            diversity_score=0.5,
        )
        return BudgetReport(
            plan=plan,
            actual_total_tokens=380,
            budget_utilization=0.38,
            items_generated=2,
        )

    def test_as_dict(self):
        report = self._make_report()
        d = report.as_dict()
        assert d["actual_total_tokens"] == 380
        assert d["budget_utilization"] == 0.38
        assert "plan" in d

    def test_from_dict(self):
        report = self._make_report()
        restored = BudgetReport.from_dict(report.as_dict())
        assert restored.actual_total_tokens == 380
        assert restored.items_generated == 2

    def test_roundtrip(self):
        report = self._make_report()
        assert BudgetReport.from_dict(report.as_dict()) == report


# ---------------------------------------------------------------------------
# DEFAULT_MODES
# ---------------------------------------------------------------------------


class TestDefaultModes:
    def test_count(self):
        assert len(DEFAULT_MODES) == 5

    def test_mode_ids(self):
        ids = {m.mode_id for m in DEFAULT_MODES}
        assert ids == {"similar", "diverse", "adversarial", "edge_case", "domain_transfer"}

    def test_similar_tokens(self):
        m = next(m for m in DEFAULT_MODES if m.mode_id == "similar")
        assert m.estimated_tokens_per_item == 200
        assert m.diversity_weight == 0.5

    def test_diverse_tokens(self):
        m = next(m for m in DEFAULT_MODES if m.mode_id == "diverse")
        assert m.estimated_tokens_per_item == 300
        assert m.diversity_weight == 1.5

    def test_adversarial_tokens(self):
        m = next(m for m in DEFAULT_MODES if m.mode_id == "adversarial")
        assert m.estimated_tokens_per_item == 250
        assert m.diversity_weight == 2.0

    def test_edge_case_tokens(self):
        m = next(m for m in DEFAULT_MODES if m.mode_id == "edge_case")
        assert m.estimated_tokens_per_item == 150
        assert m.diversity_weight == 1.0

    def test_domain_transfer_tokens(self):
        m = next(m for m in DEFAULT_MODES if m.mode_id == "domain_transfer")
        assert m.estimated_tokens_per_item == 350
        assert m.diversity_weight == 1.2


# ---------------------------------------------------------------------------
# estimate_mode_cost
# ---------------------------------------------------------------------------


class TestEstimateModeCost:
    def test_basic(self):
        mode = SimulationMode(
            mode_id="x", name="X", description="d", estimated_tokens_per_item=100
        )
        assert estimate_mode_cost(mode, 5) == 500

    def test_zero_count(self):
        mode = SimulationMode(
            mode_id="x", name="X", description="d", estimated_tokens_per_item=200
        )
        assert estimate_mode_cost(mode, 0) == 0

    def test_single(self):
        mode = SimulationMode(
            mode_id="x", name="X", description="d", estimated_tokens_per_item=350
        )
        assert estimate_mode_cost(mode, 1) == 350


# ---------------------------------------------------------------------------
# compute_plan_diversity
# ---------------------------------------------------------------------------


class TestComputePlanDiversity:
    def _modes(self):
        return [
            SimulationMode(
                mode_id="a", name="A", description="d",
                estimated_tokens_per_item=100, diversity_weight=1.0,
            ),
            SimulationMode(
                mode_id="b", name="B", description="d",
                estimated_tokens_per_item=100, diversity_weight=2.0,
            ),
        ]

    def test_empty_plan(self):
        plan = BudgetPlan(
            allocations=[], total_budget=1000,
            estimated_total_tokens=0, estimated_total_items=0,
            diversity_score=0.0,
        )
        assert compute_plan_diversity(plan, self._modes()) == 0.0

    def test_zero_items(self):
        plan = BudgetPlan(
            allocations=[
                BudgetAllocation(mode_id="a", item_count=0, estimated_tokens=0),
            ],
            total_budget=1000, estimated_total_tokens=0,
            estimated_total_items=0, diversity_score=0.0,
        )
        assert compute_plan_diversity(plan, self._modes()) == 0.0

    def test_single_mode(self):
        plan = BudgetPlan(
            allocations=[
                BudgetAllocation(mode_id="a", item_count=10, estimated_tokens=1000),
            ],
            total_budget=1000, estimated_total_tokens=1000,
            estimated_total_items=10, diversity_score=0.0,
        )
        # diversity = 1.0 * 10 / 10 = 1.0
        assert compute_plan_diversity(plan, self._modes()) == 1.0

    def test_two_modes_equal(self):
        plan = BudgetPlan(
            allocations=[
                BudgetAllocation(mode_id="a", item_count=5, estimated_tokens=500),
                BudgetAllocation(mode_id="b", item_count=5, estimated_tokens=500),
            ],
            total_budget=1000, estimated_total_tokens=1000,
            estimated_total_items=10, diversity_score=0.0,
        )
        # diversity = 1.0*5/10 + 2.0*5/10 = 0.5 + 1.0 = 1.5
        assert compute_plan_diversity(plan, self._modes()) == 1.5

    def test_weighted_toward_high_diversity(self):
        plan = BudgetPlan(
            allocations=[
                BudgetAllocation(mode_id="a", item_count=1, estimated_tokens=100),
                BudgetAllocation(mode_id="b", item_count=9, estimated_tokens=900),
            ],
            total_budget=1000, estimated_total_tokens=1000,
            estimated_total_items=10, diversity_score=0.0,
        )
        # diversity = 1.0*1/10 + 2.0*9/10 = 0.1 + 1.8 = 1.9
        score = compute_plan_diversity(plan, self._modes())
        assert abs(score - 1.9) < 1e-9


# ---------------------------------------------------------------------------
# optimize_budget
# ---------------------------------------------------------------------------


class TestOptimizeBudget:
    def test_with_default_modes(self):
        plan = optimize_budget(10000)
        assert plan.total_budget == 10000
        assert plan.estimated_total_tokens <= 10000
        assert plan.estimated_total_items > 0
        assert len(plan.allocations) == 5

    def test_empty_modes(self):
        plan = optimize_budget(1000, modes=[])
        assert plan.estimated_total_items == 0
        assert plan.allocations == []
        assert plan.diversity_score == 0.0

    def test_single_mode(self):
        modes = [
            SimulationMode(
                mode_id="x", name="X", description="d",
                estimated_tokens_per_item=100,
            ),
        ]
        plan = optimize_budget(500, modes=modes)
        assert plan.estimated_total_tokens <= 500
        assert plan.allocations[0].item_count == 5

    def test_respects_budget(self):
        plan = optimize_budget(5000)
        assert plan.estimated_total_tokens <= 5000

    def test_min_per_mode(self):
        modes = [
            SimulationMode(
                mode_id="a", name="A", description="d",
                estimated_tokens_per_item=100, diversity_weight=1.0,
            ),
            SimulationMode(
                mode_id="b", name="B", description="d",
                estimated_tokens_per_item=100, diversity_weight=1.0,
            ),
        ]
        plan = optimize_budget(500, modes=modes, min_per_mode=2)
        for alloc in plan.allocations:
            assert alloc.item_count >= 2

    def test_min_per_mode_exceeds_budget(self):
        modes = [
            SimulationMode(
                mode_id="a", name="A", description="d",
                estimated_tokens_per_item=600,
            ),
        ]
        # Budget is 500, min requires 600 - can't fit min
        plan = optimize_budget(500, modes=modes, min_per_mode=1)
        assert plan.estimated_total_tokens <= 500

    def test_budget_too_small_for_any(self):
        modes = [
            SimulationMode(
                mode_id="a", name="A", description="d",
                estimated_tokens_per_item=1000,
            ),
        ]
        plan = optimize_budget(100, modes=modes)
        assert plan.estimated_total_items == 0

    def test_diversity_score_computed(self):
        plan = optimize_budget(10000)
        assert plan.diversity_score > 0.0

    def test_allocations_match_totals(self):
        plan = optimize_budget(10000)
        total_items = sum(a.item_count for a in plan.allocations)
        total_tokens = sum(a.estimated_tokens for a in plan.allocations)
        assert total_items == plan.estimated_total_items
        assert total_tokens == plan.estimated_total_tokens

    def test_greedy_favors_high_ratio(self):
        # Mode with higher diversity_weight / tokens ratio should get more items
        modes = [
            SimulationMode(
                mode_id="cheap_diverse", name="CD", description="d",
                estimated_tokens_per_item=100, diversity_weight=2.0,
            ),
            SimulationMode(
                mode_id="expensive_similar", name="ES", description="d",
                estimated_tokens_per_item=500, diversity_weight=0.5,
            ),
        ]
        plan = optimize_budget(5000, modes=modes, min_per_mode=1)
        cd = next(a for a in plan.allocations if a.mode_id == "cheap_diverse")
        es = next(a for a in plan.allocations if a.mode_id == "expensive_similar")
        assert cd.item_count > es.item_count

    def test_zero_budget(self):
        plan = optimize_budget(0)
        assert plan.estimated_total_items == 0
        assert plan.estimated_total_tokens == 0


# ---------------------------------------------------------------------------
# update_budget_report
# ---------------------------------------------------------------------------


class TestUpdateBudgetReport:
    def _make_plan(self):
        return BudgetPlan(
            allocations=[
                BudgetAllocation(mode_id="a", item_count=5, estimated_tokens=500),
                BudgetAllocation(mode_id="b", item_count=3, estimated_tokens=900),
            ],
            total_budget=2000,
            estimated_total_tokens=1400,
            estimated_total_items=8,
            diversity_score=1.0,
        )

    def test_basic_report(self):
        plan = self._make_plan()
        report = update_budget_report(plan, {"a": 480, "b": 870})
        assert report.actual_total_tokens == 1350
        assert report.items_generated == 8

    def test_utilization(self):
        plan = self._make_plan()
        report = update_budget_report(plan, {"a": 1000, "b": 500})
        assert abs(report.budget_utilization - 0.75) < 1e-9

    def test_zero_budget_utilization(self):
        plan = BudgetPlan(
            allocations=[],
            total_budget=0,
            estimated_total_tokens=0,
            estimated_total_items=0,
            diversity_score=0.0,
        )
        report = update_budget_report(plan, {})
        assert report.budget_utilization == 0.0

    def test_updates_actual_tokens_on_report_allocations(self):
        plan = self._make_plan()
        report = update_budget_report(plan, {"a": 480, "b": 870})
        # Report's plan has updated tokens, original plan is not mutated
        assert report.plan.allocations[0].actual_tokens == 480
        assert report.plan.allocations[1].actual_tokens == 870
        # Original plan unchanged
        assert plan.allocations[0].actual_tokens == 0
        assert plan.allocations[1].actual_tokens == 0

    def test_missing_mode_in_actual(self):
        plan = self._make_plan()
        report = update_budget_report(plan, {"a": 480})
        assert report.plan.allocations[1].actual_tokens == 0
        assert report.actual_total_tokens == 480


# ---------------------------------------------------------------------------
# format_budget_plan
# ---------------------------------------------------------------------------


class TestFormatBudgetPlan:
    def test_contains_header(self):
        plan = optimize_budget(5000)
        text = format_budget_plan(plan)
        assert "Budget Plan" in text

    def test_contains_budget(self):
        plan = optimize_budget(5000)
        text = format_budget_plan(plan)
        assert "5,000" in text

    def test_contains_mode_ids(self):
        plan = optimize_budget(5000)
        text = format_budget_plan(plan)
        for alloc in plan.allocations:
            assert alloc.mode_id in text

    def test_contains_diversity(self):
        plan = optimize_budget(5000)
        text = format_budget_plan(plan)
        assert "Diversity score" in text


# ---------------------------------------------------------------------------
# format_budget_report
# ---------------------------------------------------------------------------


class TestFormatBudgetReport:
    def test_contains_header(self):
        plan = optimize_budget(5000)
        report = update_budget_report(plan, {"similar": 100})
        text = format_budget_report(report)
        assert "Budget Report" in text

    def test_contains_utilization(self):
        plan = optimize_budget(5000)
        report = update_budget_report(plan, {"similar": 100})
        text = format_budget_report(report)
        assert "utilization" in text.lower()

    def test_contains_per_mode_comparison(self):
        plan = optimize_budget(5000)
        report = update_budget_report(
            plan, {a.mode_id: a.estimated_tokens for a in plan.allocations}
        )
        text = format_budget_report(report)
        assert "planned" in text
        assert "actual" in text
