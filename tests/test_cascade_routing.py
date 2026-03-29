"""Tests for the cascade_routing module."""

from __future__ import annotations

import pytest

from evalyn_sdk.cascade_routing import (
    DEFAULT_TIERS,
    CascadeConfig,
    CascadeReport,
    ModelTier,
    RoutingDecision,
    compute_cost_savings,
    default_config,
    estimate_difficulty,
    format_cascade_report,
    route_batch,
    route_item,
    should_escalate,
)


# ---------------------------------------------------------------------------
# ModelTier
# ---------------------------------------------------------------------------


class TestModelTier:
    def test_create(self):
        t = ModelTier(tier_name="fast", model_id="m1", cost_per_1k_tokens=0.1)
        assert t.tier_name == "fast"
        assert t.quality_ceiling == 1.0

    def test_custom_ceiling(self):
        t = ModelTier(tier_name="fast", model_id="m1", cost_per_1k_tokens=0.1, quality_ceiling=0.7)
        assert t.quality_ceiling == 0.7

    def test_as_dict(self):
        t = ModelTier(tier_name="premium", model_id="o3", cost_per_1k_tokens=3.0, quality_ceiling=1.0)
        d = t.as_dict()
        assert d["tier_name"] == "premium"
        assert d["cost_per_1k_tokens"] == 3.0

    def test_from_dict(self):
        t = ModelTier.from_dict({"tier_name": "fast", "model_id": "m", "cost_per_1k_tokens": 0.5})
        assert t.quality_ceiling == 1.0

    def test_roundtrip(self):
        orig = ModelTier(tier_name="standard", model_id="gpt-4o", cost_per_1k_tokens=0.3, quality_ceiling=0.9)
        rebuilt = ModelTier.from_dict(orig.as_dict())
        assert rebuilt.tier_name == orig.tier_name
        assert rebuilt.cost_per_1k_tokens == orig.cost_per_1k_tokens


# ---------------------------------------------------------------------------
# RoutingDecision
# ---------------------------------------------------------------------------


class TestRoutingDecision:
    def test_create(self):
        d = RoutingDecision(item_id="1", assigned_tier="fast", difficulty_score=0.2, confidence=0.8)
        assert d.escalated is False

    def test_escalated(self):
        d = RoutingDecision(
            item_id="1", assigned_tier="fast", difficulty_score=0.2, confidence=0.8, escalated=True
        )
        assert d.escalated is True

    def test_as_dict(self):
        d = RoutingDecision(item_id="x", assigned_tier="premium", difficulty_score=0.9, confidence=0.5, escalated=True)
        dd = d.as_dict()
        assert dd["escalated"] is True
        assert dd["item_id"] == "x"

    def test_from_dict_defaults(self):
        d = RoutingDecision.from_dict(
            {"item_id": "a", "assigned_tier": "fast", "difficulty_score": 0.1, "confidence": 0.9}
        )
        assert d.escalated is False

    def test_roundtrip(self):
        orig = RoutingDecision(item_id="q", assigned_tier="standard", difficulty_score=0.5, confidence=0.6, escalated=True)
        rebuilt = RoutingDecision.from_dict(orig.as_dict())
        assert rebuilt.escalated == orig.escalated
        assert rebuilt.difficulty_score == orig.difficulty_score


# ---------------------------------------------------------------------------
# CascadeConfig
# ---------------------------------------------------------------------------


class TestCascadeConfig:
    def test_create(self):
        c = CascadeConfig(tiers=DEFAULT_TIERS, difficulty_thresholds=[0.3, 0.7])
        assert c.escalation_threshold == 0.3
        assert len(c.tiers) == 3

    def test_as_dict(self):
        c = default_config()
        d = c.as_dict()
        assert len(d["tiers"]) == 3
        assert len(d["difficulty_thresholds"]) == 2

    def test_from_dict(self):
        c = default_config()
        rebuilt = CascadeConfig.from_dict(c.as_dict())
        assert len(rebuilt.tiers) == 3
        assert rebuilt.escalation_threshold == c.escalation_threshold

    def test_roundtrip(self):
        c = CascadeConfig(
            tiers=[ModelTier("t", "m", 1.0)],
            difficulty_thresholds=[0.5],
            escalation_threshold=0.4,
        )
        rebuilt = CascadeConfig.from_dict(c.as_dict())
        assert rebuilt.escalation_threshold == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# CascadeReport
# ---------------------------------------------------------------------------


class TestCascadeReport:
    def _make_report(self):
        decisions = [
            RoutingDecision("1", "fast", 0.1, 0.9),
            RoutingDecision("2", "premium", 0.8, 0.7),
        ]
        return CascadeReport(
            decisions=decisions,
            tier_distribution={"fast": 1, "premium": 1},
            estimated_cost_savings=0.5,
            total_items=2,
        )

    def test_as_dict(self):
        r = self._make_report()
        d = r.as_dict()
        assert d["total_items"] == 2
        assert len(d["decisions"]) == 2

    def test_from_dict(self):
        r = self._make_report()
        rebuilt = CascadeReport.from_dict(r.as_dict())
        assert rebuilt.total_items == 2
        assert rebuilt.tier_distribution["fast"] == 1

    def test_roundtrip(self):
        r = self._make_report()
        rebuilt = CascadeReport.from_dict(r.as_dict())
        assert rebuilt.estimated_cost_savings == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# DEFAULT_TIERS
# ---------------------------------------------------------------------------


class TestDefaultTiers:
    def test_count(self):
        assert len(DEFAULT_TIERS) == 3

    def test_tier_names(self):
        names = [t.tier_name for t in DEFAULT_TIERS]
        assert names == ["fast", "standard", "premium"]

    def test_costs_ascending(self):
        costs = [t.cost_per_1k_tokens for t in DEFAULT_TIERS]
        assert costs == sorted(costs)

    def test_ceilings(self):
        assert DEFAULT_TIERS[0].quality_ceiling == 0.7
        assert DEFAULT_TIERS[1].quality_ceiling == 0.9
        assert DEFAULT_TIERS[2].quality_ceiling == 1.0


# ---------------------------------------------------------------------------
# estimate_difficulty
# ---------------------------------------------------------------------------


class TestEstimateDifficulty:
    def test_empty(self):
        assert estimate_difficulty("") == 0.0

    def test_whitespace_only(self):
        assert estimate_difficulty("   ") == 0.0

    def test_simple_short(self):
        score = estimate_difficulty("hi")
        assert 0.0 <= score <= 0.5  # short = easy

    def test_complex_long(self):
        text = (
            "Explain and analyze the fundamental differences between "
            "supervised and unsupervised learning approaches. Compare and "
            "contrast their strengths. Evaluate which is better for anomaly "
            "detection in production systems. Justify your reasoning with "
            "concrete examples and prove your claims."
        )
        score = estimate_difficulty(text)
        assert score > 0.3  # should be harder

    def test_bounded_zero_one(self):
        for text in ["a", "hello world", "x " * 500]:
            score = estimate_difficulty(text)
            assert 0.0 <= score <= 1.0

    def test_negations_increase_difficulty(self):
        base = "Write a function that returns values"
        with_neg = "Write a function that does not return values and never raises exceptions"
        s_base = estimate_difficulty(base)
        s_neg = estimate_difficulty(with_neg)
        assert s_neg >= s_base

    def test_complex_markers_increase_difficulty(self):
        simple = "Write a hello world program"
        complex_text = "Analyze and evaluate the design then optimize the implementation"
        s_simple = estimate_difficulty(simple)
        s_complex = estimate_difficulty(complex_text)
        assert s_complex > s_simple


# ---------------------------------------------------------------------------
# route_item
# ---------------------------------------------------------------------------


class TestRouteItem:
    def test_easy_item_routes_to_fast(self):
        config = default_config()
        decision = route_item("1", "hi", config)
        assert decision.assigned_tier == "fast"
        assert decision.difficulty_score < 0.35

    def test_hard_item_routes_to_premium(self):
        config = default_config()
        hard = (
            "Analyze and compare and evaluate and critique and synthesize "
            "the fundamental differences between multiple complex distributed systems. "
            "Justify each claim with concrete evidence. Consider edge cases and prove correctness rigorously. "
            "Do not skip any reasoning steps and never make assumptions. "
            "You should not oversimplify and do not ignore the trade-offs. "
            "Design an optimal architecture that balances consistency and availability. "
            "Derive the theoretical bounds and optimize for both latency and throughput. "
            "Implement a proof of concept and debug any issues that arise during evaluation. "
            "Explain why certain approaches cannot work in this context."
        )
        decision = route_item("2", hard, config)
        assert decision.assigned_tier == "premium"

    def test_decision_fields(self):
        config = default_config()
        d = route_item("abc", "hello", config)
        assert d.item_id == "abc"
        assert 0.0 <= d.confidence <= 1.0
        assert 0.0 <= d.difficulty_score <= 1.0
        assert d.escalated is False

    def test_single_tier_config(self):
        config = CascadeConfig(
            tiers=[ModelTier("only", "m", 1.0)],
            difficulty_thresholds=[],
        )
        d = route_item("1", "anything at all", config)
        assert d.assigned_tier == "only"
        assert d.confidence == 1.0  # no thresholds, max confidence

    def test_two_tier_config(self):
        config = CascadeConfig(
            tiers=[ModelTier("low", "m1", 0.1), ModelTier("high", "m2", 1.0)],
            difficulty_thresholds=[0.5],
        )
        easy = route_item("1", "hi", config)
        assert easy.assigned_tier == "low"


# ---------------------------------------------------------------------------
# route_batch
# ---------------------------------------------------------------------------


class TestRouteBatch:
    def test_basic(self):
        config = default_config()
        items = [
            {"id": "a", "input": "hello"},
            {"id": "b", "input": "Analyze and evaluate a complex distributed system design"},
        ]
        report = route_batch(items, config)
        assert report.total_items == 2
        assert sum(report.tier_distribution.values()) == 2

    def test_empty(self):
        config = default_config()
        report = route_batch([], config)
        assert report.total_items == 0
        assert report.estimated_cost_savings == 0.0

    def test_custom_text_field(self):
        config = default_config()
        items = [{"id": "1", "prompt": "hello world"}]
        report = route_batch(items, config, text_field="prompt")
        assert report.total_items == 1

    def test_auto_id(self):
        config = default_config()
        items = [{"input": "test"}]
        report = route_batch(items, config)
        assert report.decisions[0].item_id == "0"

    def test_savings_positive_with_easy_items(self):
        config = default_config()
        items = [{"id": str(i), "input": "simple"} for i in range(10)]
        report = route_batch(items, config)
        assert report.estimated_cost_savings > 0.0

    def test_all_premium_no_savings(self):
        # If all items go to premium, savings = 0
        config = CascadeConfig(
            tiers=[ModelTier("premium", "o3", 3.0, 1.0)],
            difficulty_thresholds=[],
        )
        items = [{"id": "1", "input": "x"}]
        report = route_batch(items, config)
        assert report.estimated_cost_savings == pytest.approx(0.0)

    def test_tier_distribution_sums_to_total(self):
        config = default_config()
        items = [
            {"id": "1", "input": "hi"},
            {"id": "2", "input": "Analyze and evaluate complex systems with reasoning"},
            {"id": "3", "input": "simple question"},
        ]
        report = route_batch(items, config)
        assert sum(report.tier_distribution.values()) == report.total_items

    def test_missing_text_field_uses_empty(self):
        config = default_config()
        items = [{"id": "1"}]  # no "input" field
        report = route_batch(items, config)
        assert report.total_items == 1
        # Empty text should be easy
        assert report.decisions[0].difficulty_score == 0.0


# ---------------------------------------------------------------------------
# should_escalate
# ---------------------------------------------------------------------------


class TestShouldEscalate:
    def test_low_quality_escalates(self):
        config = default_config()
        decision = RoutingDecision("1", "fast", 0.2, 0.8)
        assert should_escalate(decision, 0.1, config) is True

    def test_high_quality_does_not_escalate(self):
        config = default_config()
        decision = RoutingDecision("1", "fast", 0.2, 0.8)
        assert should_escalate(decision, 0.9, config) is False

    def test_at_threshold_does_not_escalate(self):
        config = default_config()
        decision = RoutingDecision("1", "fast", 0.2, 0.8)
        assert should_escalate(decision, 0.3, config) is False

    def test_just_below_threshold(self):
        config = default_config()
        decision = RoutingDecision("1", "fast", 0.2, 0.8)
        assert should_escalate(decision, 0.299, config) is True

    def test_custom_threshold(self):
        config = CascadeConfig(
            tiers=DEFAULT_TIERS,
            difficulty_thresholds=[0.3, 0.7],
            escalation_threshold=0.5,
        )
        decision = RoutingDecision("1", "fast", 0.2, 0.8)
        assert should_escalate(decision, 0.4, config) is True
        assert should_escalate(decision, 0.6, config) is False


# ---------------------------------------------------------------------------
# compute_cost_savings
# ---------------------------------------------------------------------------


class TestComputeCostSavings:
    def test_zero_baseline(self):
        report = CascadeReport([], {}, 0.5, 0)
        assert compute_cost_savings(report, 0.0) == 0.0

    def test_returns_estimated(self):
        report = CascadeReport([], {}, 0.75, 0)
        assert compute_cost_savings(report, 100.0) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# format_cascade_report
# ---------------------------------------------------------------------------


class TestFormatCascadeReport:
    def test_contains_header(self):
        report = CascadeReport([], {}, 0.0, 0)
        text = format_cascade_report(report)
        assert "Cascade Routing Report" in text

    def test_contains_savings(self):
        report = CascadeReport([], {}, 0.42, 0)
        text = format_cascade_report(report)
        assert "42.0%" in text

    def test_contains_tier_distribution(self):
        report = CascadeReport(
            decisions=[RoutingDecision("1", "fast", 0.1, 0.9)],
            tier_distribution={"fast": 1},
            estimated_cost_savings=0.9,
            total_items=1,
        )
        text = format_cascade_report(report)
        assert "fast" in text
        assert "Tier Distribution" in text

    def test_contains_decisions(self):
        decisions = [
            RoutingDecision("a", "fast", 0.1, 0.9),
            RoutingDecision("b", "premium", 0.8, 0.7, escalated=True),
        ]
        report = CascadeReport(
            decisions=decisions,
            tier_distribution={"fast": 1, "premium": 1},
            estimated_cost_savings=0.5,
            total_items=2,
        )
        text = format_cascade_report(report)
        assert "ESCALATED" in text
        assert "Decisions:" in text

    def test_empty_report(self):
        report = CascadeReport([], {}, 0.0, 0)
        text = format_cascade_report(report)
        assert "Total items:       0" in text
