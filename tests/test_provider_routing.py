"""Tests for evaluation.provider_routing module."""

from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.provider_routing import (
    RoutingConfig,
    RoutingDecision,
    RoutingReport,
    RoutingRule,
    create_cost_optimized_routing,
    create_safety_routing,
    match_metric_to_rule,
    route_all_metrics,
    route_metric,
)


# -- RoutingRule -------------------------------------------------------------


class TestRoutingRule:
    def test_defaults(self):
        rule = RoutingRule(metric_pattern="safety", provider="gemini")
        assert rule.model == ""
        assert rule.priority == 0

    def test_as_dict(self):
        rule = RoutingRule(metric_pattern="safety", provider="gemini", priority=10)
        d = rule.as_dict()
        assert d["metric_pattern"] == "safety"
        assert d["provider"] == "gemini"
        assert d["priority"] == 10

    def test_from_dict(self):
        d = {"metric_pattern": "quality", "provider": "openai", "model": "gpt-4o"}
        rule = RoutingRule.from_dict(d)
        assert rule.metric_pattern == "quality"
        assert rule.model == "gpt-4o"
        assert rule.priority == 0

    def test_roundtrip(self):
        rule = RoutingRule(
            metric_pattern="coherence", provider="anthropic", model="claude-sonnet-4-20250514", priority=5
        )
        restored = RoutingRule.from_dict(rule.as_dict())
        assert restored.metric_pattern == rule.metric_pattern
        assert restored.provider == rule.provider
        assert restored.model == rule.model
        assert restored.priority == rule.priority


# -- RoutingDecision ---------------------------------------------------------


class TestRoutingDecision:
    def test_as_dict(self):
        d = RoutingDecision(
            metric_id="safety_score", provider="gemini", model="gemini-2.5-flash"
        )
        out = d.as_dict()
        assert out["metric_id"] == "safety_score"
        assert out["rule_name"] == "default"

    def test_custom_rule_name(self):
        d = RoutingDecision(
            metric_id="x", provider="y", model="z", rule_name="safety"
        )
        assert d.rule_name == "safety"


# -- RoutingConfig -----------------------------------------------------------


class TestRoutingConfig:
    def test_defaults(self):
        cfg = RoutingConfig()
        assert cfg.default_provider == "gemini"
        assert cfg.default_model == "gemini-2.5-flash"
        assert cfg.rules == []

    def test_as_dict(self):
        cfg = RoutingConfig(
            rules=[RoutingRule(metric_pattern="safety", provider="gemini")],
            default_provider="openai",
        )
        d = cfg.as_dict()
        assert len(d["rules"]) == 1
        assert d["default_provider"] == "openai"

    def test_from_dict(self):
        d = {
            "rules": [{"metric_pattern": "safety", "provider": "gemini"}],
            "default_provider": "openai",
            "default_model": "gpt-4o",
        }
        cfg = RoutingConfig.from_dict(d)
        assert len(cfg.rules) == 1
        assert cfg.default_provider == "openai"
        assert cfg.default_model == "gpt-4o"

    def test_roundtrip(self):
        cfg = RoutingConfig(
            rules=[
                RoutingRule(metric_pattern="safety", provider="gemini", priority=10),
                RoutingRule(metric_pattern="quality", provider="openai", model="gpt-4o"),
            ],
            default_provider="gemini",
            default_model="gemini-2.5-flash",
        )
        restored = RoutingConfig.from_dict(cfg.as_dict())
        assert len(restored.rules) == 2
        assert restored.rules[0].metric_pattern == "safety"
        assert restored.default_provider == "gemini"


# -- RoutingReport -----------------------------------------------------------


class TestRoutingReport:
    def test_empty(self):
        report = RoutingReport()
        assert report.decisions == []
        assert report.providers_used == []
        assert report.estimated_cost_savings_pct == 0.0

    def test_as_dict(self):
        report = RoutingReport(
            decisions=[
                RoutingDecision(metric_id="x", provider="gemini", model="m")
            ],
            providers_used=["gemini"],
        )
        d = report.as_dict()
        assert len(d["decisions"]) == 1

    def test_from_dict(self):
        d = {
            "decisions": [
                {"metric_id": "x", "provider": "gemini", "model": "m"}
            ],
            "providers_used": ["gemini"],
            "estimated_cost_savings_pct": 25.0,
        }
        report = RoutingReport.from_dict(d)
        assert len(report.decisions) == 1
        assert report.estimated_cost_savings_pct == 25.0

    def test_roundtrip(self):
        report = RoutingReport(
            decisions=[
                RoutingDecision(metric_id="safety", provider="gemini", model="m"),
                RoutingDecision(
                    metric_id="quality", provider="openai", model="gpt-4o", rule_name="quality"
                ),
            ],
            providers_used=["gemini", "openai"],
            estimated_cost_savings_pct=50.0,
        )
        restored = RoutingReport.from_dict(report.as_dict())
        assert len(restored.decisions) == 2
        assert restored.estimated_cost_savings_pct == 50.0

    def test_format_text_basic(self):
        report = RoutingReport(
            decisions=[
                RoutingDecision(
                    metric_id="safety",
                    provider="gemini",
                    model="gemini-2.5-flash",
                    rule_name="safety",
                )
            ],
            providers_used=["gemini"],
        )
        text = report.format_text()
        assert "safety" in text
        assert "gemini" in text
        assert "Routing Report" in text

    def test_format_text_with_savings(self):
        report = RoutingReport(
            decisions=[],
            providers_used=["gemini", "openai"],
            estimated_cost_savings_pct=33.3,
        )
        text = report.format_text()
        assert "33.3%" in text

    def test_format_text_no_model(self):
        report = RoutingReport(
            decisions=[
                RoutingDecision(metric_id="x", provider="gemini", model="")
            ],
            providers_used=["gemini"],
        )
        text = report.format_text()
        # No parenthesized model should appear
        assert "()" not in text


# -- match_metric_to_rule ----------------------------------------------------


class TestMatchMetricToRule:
    def test_exact_match(self):
        rules = [
            RoutingRule(metric_pattern="safety_score", provider="gemini"),
            RoutingRule(metric_pattern="safety", provider="openai"),
        ]
        result = match_metric_to_rule("safety_score", rules)
        assert result is not None
        assert result.provider == "gemini"

    def test_substring_match(self):
        rules = [
            RoutingRule(metric_pattern="safety", provider="gemini"),
        ]
        result = match_metric_to_rule("content_safety_check", rules)
        assert result is not None
        assert result.provider == "gemini"

    def test_no_match(self):
        rules = [
            RoutingRule(metric_pattern="safety", provider="gemini"),
        ]
        result = match_metric_to_rule("coherence_score", rules)
        assert result is None

    def test_priority_wins(self):
        rules = [
            RoutingRule(metric_pattern="safety", provider="openai", priority=1),
            RoutingRule(metric_pattern="safety", provider="gemini", priority=10),
        ]
        result = match_metric_to_rule("safety_check", rules)
        assert result is not None
        assert result.provider == "gemini"

    def test_exact_beats_substring(self):
        rules = [
            RoutingRule(metric_pattern="safe", provider="openai", priority=100),
            RoutingRule(metric_pattern="safety", provider="gemini", priority=0),
        ]
        result = match_metric_to_rule("safety", rules)
        assert result is not None
        assert result.provider == "gemini"

    def test_empty_rules(self):
        result = match_metric_to_rule("anything", [])
        assert result is None


# -- route_metric ------------------------------------------------------------


class TestRouteMetric:
    def test_matching_rule(self):
        cfg = RoutingConfig(
            rules=[RoutingRule(metric_pattern="safety", provider="gemini", model="gemini-2.5-flash")],
        )
        decision = route_metric("safety", cfg)
        assert decision.provider == "gemini"
        assert decision.rule_name == "safety"

    def test_falls_back_to_default(self):
        cfg = RoutingConfig(
            rules=[RoutingRule(metric_pattern="safety", provider="gemini")],
            default_provider="openai",
            default_model="gpt-4o",
        )
        decision = route_metric("coherence", cfg)
        assert decision.provider == "openai"
        assert decision.model == "gpt-4o"
        assert decision.rule_name == "default"

    def test_rule_model_used(self):
        cfg = RoutingConfig(
            rules=[RoutingRule(metric_pattern="quality", provider="openai", model="gpt-4o")],
        )
        decision = route_metric("quality", cfg)
        assert decision.model == "gpt-4o"

    def test_rule_without_model_uses_default(self):
        cfg = RoutingConfig(
            rules=[RoutingRule(metric_pattern="quality", provider="openai")],
            default_model="gemini-2.5-flash",
        )
        decision = route_metric("quality", cfg)
        assert decision.model == "gemini-2.5-flash"


# -- route_all_metrics -------------------------------------------------------


class TestRouteAllMetrics:
    def test_providers_used(self):
        cfg = RoutingConfig(
            rules=[
                RoutingRule(metric_pattern="safety", provider="gemini"),
            ],
            default_provider="openai",
        )
        report = route_all_metrics(["safety", "quality"], cfg)
        assert "gemini" in report.providers_used
        assert "openai" in report.providers_used

    def test_empty_metrics(self):
        cfg = RoutingConfig()
        report = route_all_metrics([], cfg)
        assert report.decisions == []
        assert report.providers_used == []
        assert report.estimated_cost_savings_pct == 0.0

    def test_all_default(self):
        cfg = RoutingConfig(default_provider="gemini")
        report = route_all_metrics(["a", "b", "c"], cfg)
        assert len(report.decisions) == 3
        assert report.providers_used == ["gemini"]
        # No savings when single provider
        assert report.estimated_cost_savings_pct == 0.0

    def test_savings_with_mixed_routing(self):
        cfg = RoutingConfig(
            rules=[RoutingRule(metric_pattern="safety", provider="openai")],
            default_provider="gemini",
        )
        report = route_all_metrics(["safety", "quality"], cfg)
        assert report.estimated_cost_savings_pct > 0


# -- create_safety_routing ---------------------------------------------------


class TestCreateSafetyRouting:
    def test_preset(self):
        cfg = create_safety_routing()
        assert len(cfg.rules) == 3
        assert cfg.default_provider == "openai"

    def test_safety_metric_routed(self):
        cfg = create_safety_routing()
        decision = route_metric("safety_score", cfg)
        assert decision.provider == "gemini"

    def test_non_safety_uses_default(self):
        cfg = create_safety_routing()
        decision = route_metric("coherence", cfg)
        assert decision.provider == "openai"


# -- create_cost_optimized_routing -------------------------------------------


class TestCreateCostOptimizedRouting:
    def test_preset(self):
        cfg = create_cost_optimized_routing()
        assert len(cfg.rules) > 0
        assert cfg.default_provider == "gemini"

    def test_simple_metric_cheap_model(self):
        cfg = create_cost_optimized_routing()
        decision = route_metric("response_length", cfg)
        assert "lite" in decision.model.lower() or "flash" in decision.model.lower()

    def test_complex_metric_expensive_model(self):
        cfg = create_cost_optimized_routing()
        decision = route_metric("reasoning_quality", cfg)
        assert decision.provider == "anthropic"
