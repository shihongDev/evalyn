"""Tests for the judge routing module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.judge_routing import (
    BUILTIN_JUDGES,
    JudgeModel,
    JudgeRoute,
    JudgeRoutingConfig,
    JudgeRoutingResult,
    estimate_routing_cost,
    format_routing_plan,
    route_all_metrics,
    route_metric,
)


# ---------------------------------------------------------------------------
# JudgeModel
# ---------------------------------------------------------------------------


class TestJudgeModel:
    def test_basic(self):
        m = JudgeModel(
            model_id="gpt-4o",
            provider="openai",
            quality_tier="standard",
            specializations=["general"],
        )
        assert m.model_id == "gpt-4o"
        assert m.provider == "openai"
        assert m.quality_tier == "standard"

    def test_default_specializations(self):
        m = JudgeModel(model_id="m", provider="p", quality_tier="fast")
        assert m.specializations == []

    def test_as_dict(self):
        m = JudgeModel(
            model_id="claude-sonnet",
            provider="anthropic",
            quality_tier="standard",
            specializations=["safety", "quality"],
        )
        d = m.as_dict()
        assert d["model_id"] == "claude-sonnet"
        assert d["provider"] == "anthropic"
        assert d["specializations"] == ["safety", "quality"]

    def test_from_dict_minimal(self):
        m = JudgeModel.from_dict(
            {"model_id": "x", "provider": "p", "quality_tier": "fast"}
        )
        assert m.specializations == []

    def test_from_dict_full(self):
        data = {
            "model_id": "m",
            "provider": "p",
            "quality_tier": "premium",
            "specializations": ["a", "b"],
        }
        m = JudgeModel.from_dict(data)
        assert m.quality_tier == "premium"
        assert m.specializations == ["a", "b"]

    def test_roundtrip(self):
        m = JudgeModel(
            model_id="gemini-2.0-flash",
            provider="google",
            quality_tier="fast",
            specializations=["general"],
        )
        assert JudgeModel.from_dict(m.as_dict()) == m

    def test_specializations_copied(self):
        original = ["a"]
        m = JudgeModel.from_dict(
            {"model_id": "x", "provider": "p", "quality_tier": "fast", "specializations": original}
        )
        original.append("b")
        assert m.specializations == ["a"]


# ---------------------------------------------------------------------------
# JudgeRoute
# ---------------------------------------------------------------------------


class TestJudgeRoute:
    def test_basic(self):
        r = JudgeRoute(metric_id="accuracy", assigned_model="gpt-4o")
        assert r.metric_id == "accuracy"
        assert r.assigned_model == "gpt-4o"
        assert r.reason == ""

    def test_with_reason(self):
        r = JudgeRoute(metric_id="safety", assigned_model="claude-sonnet", reason="safety specialist")
        assert r.reason == "safety specialist"

    def test_as_dict(self):
        r = JudgeRoute(metric_id="m", assigned_model="a", reason="r")
        d = r.as_dict()
        assert d == {"metric_id": "m", "assigned_model": "a", "reason": "r"}

    def test_from_dict_minimal(self):
        r = JudgeRoute.from_dict({"metric_id": "m", "assigned_model": "a"})
        assert r.reason == ""

    def test_from_dict_full(self):
        r = JudgeRoute.from_dict(
            {"metric_id": "m", "assigned_model": "a", "reason": "because"}
        )
        assert r.reason == "because"

    def test_roundtrip(self):
        r = JudgeRoute(metric_id="helpfulness", assigned_model="gpt-4o", reason="test")
        assert JudgeRoute.from_dict(r.as_dict()) == r


# ---------------------------------------------------------------------------
# JudgeRoutingConfig
# ---------------------------------------------------------------------------


class TestJudgeRoutingConfig:
    def test_defaults(self):
        cfg = JudgeRoutingConfig()
        assert cfg.default_model == "gemini-2.0-flash"
        assert cfg.routes == {}
        assert cfg.category_routes == {}

    def test_custom(self):
        cfg = JudgeRoutingConfig(
            default_model="gpt-4o",
            routes={"accuracy": "claude-sonnet"},
            category_routes={"safety": "claude-sonnet"},
        )
        assert cfg.default_model == "gpt-4o"
        assert cfg.routes == {"accuracy": "claude-sonnet"}
        assert cfg.category_routes == {"safety": "claude-sonnet"}

    def test_as_dict(self):
        cfg = JudgeRoutingConfig(routes={"a": "b"})
        d = cfg.as_dict()
        assert d["default_model"] == "gemini-2.0-flash"
        assert d["routes"] == {"a": "b"}
        assert d["category_routes"] == {}

    def test_from_dict_minimal(self):
        cfg = JudgeRoutingConfig.from_dict({})
        assert cfg.default_model == "gemini-2.0-flash"
        assert cfg.routes == {}

    def test_from_dict_full(self):
        data = {
            "default_model": "gpt-4o",
            "routes": {"x": "y"},
            "category_routes": {"c": "d"},
        }
        cfg = JudgeRoutingConfig.from_dict(data)
        assert cfg.default_model == "gpt-4o"
        assert cfg.routes == {"x": "y"}
        assert cfg.category_routes == {"c": "d"}

    def test_roundtrip(self):
        cfg = JudgeRoutingConfig(
            default_model="claude-sonnet",
            routes={"a": "b"},
            category_routes={"c": "d"},
        )
        assert JudgeRoutingConfig.from_dict(cfg.as_dict()) == cfg

    def test_routes_are_copied(self):
        original = {"a": "b"}
        cfg = JudgeRoutingConfig.from_dict({"routes": original})
        original["c"] = "d"
        assert cfg.routes == {"a": "b"}


# ---------------------------------------------------------------------------
# JudgeRoutingResult
# ---------------------------------------------------------------------------


class TestJudgeRoutingResult:
    def test_defaults(self):
        r = JudgeRoutingResult()
        assert r.routes == []
        assert r.models_used == []
        assert r.total_metrics == 0

    def test_as_dict(self):
        route = JudgeRoute(metric_id="m", assigned_model="a")
        r = JudgeRoutingResult(routes=[route], models_used=["a"], total_metrics=1)
        d = r.as_dict()
        assert len(d["routes"]) == 1
        assert d["models_used"] == ["a"]
        assert d["total_metrics"] == 1

    def test_from_dict(self):
        data = {
            "routes": [{"metric_id": "m", "assigned_model": "a", "reason": ""}],
            "models_used": ["a"],
            "total_metrics": 1,
        }
        r = JudgeRoutingResult.from_dict(data)
        assert len(r.routes) == 1
        assert r.routes[0].metric_id == "m"

    def test_roundtrip(self):
        route = JudgeRoute(metric_id="x", assigned_model="y", reason="z")
        r = JudgeRoutingResult(routes=[route], models_used=["y"], total_metrics=1)
        restored = JudgeRoutingResult.from_dict(r.as_dict())
        assert restored.routes == r.routes
        assert restored.models_used == r.models_used
        assert restored.total_metrics == r.total_metrics


# ---------------------------------------------------------------------------
# BUILTIN_JUDGES
# ---------------------------------------------------------------------------


class TestBuiltinJudges:
    def test_count(self):
        assert len(BUILTIN_JUDGES) == 4

    def test_ids(self):
        ids = {j.model_id for j in BUILTIN_JUDGES}
        assert ids == {"gemini-2.0-flash", "gemini-2.0-pro", "claude-sonnet", "gpt-4o"}

    def test_gemini_flash_is_fast(self):
        flash = [j for j in BUILTIN_JUDGES if j.model_id == "gemini-2.0-flash"][0]
        assert flash.quality_tier == "fast"
        assert flash.provider == "google"

    def test_gemini_pro_is_premium(self):
        pro = [j for j in BUILTIN_JUDGES if j.model_id == "gemini-2.0-pro"][0]
        assert pro.quality_tier == "premium"

    def test_claude_sonnet_specializations(self):
        cs = [j for j in BUILTIN_JUDGES if j.model_id == "claude-sonnet"][0]
        assert "safety" in cs.specializations
        assert "quality" in cs.specializations
        assert cs.provider == "anthropic"

    def test_gpt4o_is_standard(self):
        g = [j for j in BUILTIN_JUDGES if j.model_id == "gpt-4o"][0]
        assert g.quality_tier == "standard"
        assert g.provider == "openai"


# ---------------------------------------------------------------------------
# route_metric
# ---------------------------------------------------------------------------


class TestRouteMetric:
    def test_default_route(self):
        cfg = JudgeRoutingConfig()
        r = route_metric("accuracy", "quality", cfg)
        assert r.metric_id == "accuracy"
        assert r.assigned_model == "gemini-2.0-flash"
        assert "default" in r.reason

    def test_specific_override(self):
        cfg = JudgeRoutingConfig(routes={"safety_check": "claude-sonnet"})
        r = route_metric("safety_check", "safety", cfg)
        assert r.assigned_model == "claude-sonnet"
        assert "explicit" in r.reason

    def test_category_route(self):
        cfg = JudgeRoutingConfig(category_routes={"safety": "claude-sonnet"})
        r = route_metric("toxicity", "safety", cfg)
        assert r.assigned_model == "claude-sonnet"
        assert "category" in r.reason

    def test_specific_overrides_category(self):
        cfg = JudgeRoutingConfig(
            routes={"toxicity": "gpt-4o"},
            category_routes={"safety": "claude-sonnet"},
        )
        r = route_metric("toxicity", "safety", cfg)
        assert r.assigned_model == "gpt-4o"
        assert "explicit" in r.reason

    def test_category_overrides_default(self):
        cfg = JudgeRoutingConfig(
            default_model="gemini-2.0-flash",
            category_routes={"quality": "gemini-2.0-pro"},
        )
        r = route_metric("helpfulness", "quality", cfg)
        assert r.assigned_model == "gemini-2.0-pro"

    def test_unmatched_category_falls_to_default(self):
        cfg = JudgeRoutingConfig(category_routes={"safety": "claude-sonnet"})
        r = route_metric("helpfulness", "quality", cfg)
        assert r.assigned_model == "gemini-2.0-flash"

    def test_custom_default(self):
        cfg = JudgeRoutingConfig(default_model="gpt-4o")
        r = route_metric("any_metric", "any_cat", cfg)
        assert r.assigned_model == "gpt-4o"


# ---------------------------------------------------------------------------
# route_all_metrics
# ---------------------------------------------------------------------------


class TestRouteAllMetrics:
    def test_single_metric(self):
        metrics = [{"id": "accuracy", "category": "quality"}]
        cfg = JudgeRoutingConfig()
        result = route_all_metrics(metrics, cfg)
        assert result.total_metrics == 1
        assert len(result.routes) == 1
        assert result.models_used == ["gemini-2.0-flash"]

    def test_multiple_metrics(self):
        metrics = [
            {"id": "accuracy", "category": "quality"},
            {"id": "toxicity", "category": "safety"},
            {"id": "helpfulness", "category": "quality"},
        ]
        cfg = JudgeRoutingConfig(
            category_routes={"safety": "claude-sonnet"},
        )
        result = route_all_metrics(metrics, cfg)
        assert result.total_metrics == 3
        assert "gemini-2.0-flash" in result.models_used
        assert "claude-sonnet" in result.models_used

    def test_empty_metrics(self):
        result = route_all_metrics([], JudgeRoutingConfig())
        assert result.total_metrics == 0
        assert result.routes == []
        assert result.models_used == []

    def test_models_used_unique(self):
        metrics = [
            {"id": "a", "category": "quality"},
            {"id": "b", "category": "quality"},
        ]
        result = route_all_metrics(metrics, JudgeRoutingConfig())
        assert result.models_used == ["gemini-2.0-flash"]

    def test_models_used_preserves_order(self):
        metrics = [
            {"id": "a", "category": "safety"},
            {"id": "b", "category": "quality"},
        ]
        cfg = JudgeRoutingConfig(
            category_routes={"safety": "claude-sonnet"},
        )
        result = route_all_metrics(metrics, cfg)
        assert result.models_used == ["claude-sonnet", "gemini-2.0-flash"]

    def test_all_routed(self):
        metrics = [
            {"id": "accuracy", "category": "quality"},
            {"id": "safety", "category": "safety"},
        ]
        cfg = JudgeRoutingConfig(
            routes={"accuracy": "gpt-4o"},
            category_routes={"safety": "claude-sonnet"},
        )
        result = route_all_metrics(metrics, cfg)
        routes_map = {r.metric_id: r.assigned_model for r in result.routes}
        assert routes_map["accuracy"] == "gpt-4o"
        assert routes_map["safety"] == "claude-sonnet"


# ---------------------------------------------------------------------------
# estimate_routing_cost
# ---------------------------------------------------------------------------


class TestEstimateRoutingCost:
    def test_basic_cost(self):
        result = JudgeRoutingResult(
            routes=[
                JudgeRoute(metric_id="a", assigned_model="gpt-4o"),
                JudgeRoute(metric_id="b", assigned_model="gpt-4o"),
            ],
            models_used=["gpt-4o"],
            total_metrics=2,
        )
        cost = estimate_routing_cost(result, {"gpt-4o": 0.10})
        assert abs(cost - 0.20) < 1e-9

    def test_mixed_models(self):
        result = JudgeRoutingResult(
            routes=[
                JudgeRoute(metric_id="a", assigned_model="gpt-4o"),
                JudgeRoute(metric_id="b", assigned_model="gemini-2.0-flash"),
            ],
            models_used=["gpt-4o", "gemini-2.0-flash"],
            total_metrics=2,
        )
        cost = estimate_routing_cost(
            result, {"gpt-4o": 0.10, "gemini-2.0-flash": 0.01}
        )
        assert abs(cost - 0.11) < 1e-9

    def test_unknown_model_zero_cost(self):
        result = JudgeRoutingResult(
            routes=[JudgeRoute(metric_id="a", assigned_model="unknown-model")],
            models_used=["unknown-model"],
            total_metrics=1,
        )
        cost = estimate_routing_cost(result, {"gpt-4o": 0.10})
        assert cost == 0.0

    def test_empty_result(self):
        result = JudgeRoutingResult()
        cost = estimate_routing_cost(result, {"gpt-4o": 0.10})
        assert cost == 0.0

    def test_empty_cost_dict(self):
        result = JudgeRoutingResult(
            routes=[JudgeRoute(metric_id="a", assigned_model="gpt-4o")],
            models_used=["gpt-4o"],
            total_metrics=1,
        )
        cost = estimate_routing_cost(result, {})
        assert cost == 0.0


# ---------------------------------------------------------------------------
# format_routing_plan
# ---------------------------------------------------------------------------


class TestFormatRoutingPlan:
    def test_empty(self):
        result = JudgeRoutingResult()
        text = format_routing_plan(result)
        assert "0 metrics" in text

    def test_single_route(self):
        result = JudgeRoutingResult(
            routes=[JudgeRoute(metric_id="accuracy", assigned_model="gpt-4o", reason="default model")],
            models_used=["gpt-4o"],
            total_metrics=1,
        )
        text = format_routing_plan(result)
        assert "accuracy -> gpt-4o" in text
        assert "default model" in text
        assert "1 metrics" in text

    def test_multiple_routes(self):
        result = JudgeRoutingResult(
            routes=[
                JudgeRoute(metric_id="accuracy", assigned_model="gpt-4o"),
                JudgeRoute(metric_id="safety", assigned_model="claude-sonnet"),
            ],
            models_used=["gpt-4o", "claude-sonnet"],
            total_metrics=2,
        )
        text = format_routing_plan(result)
        assert "gpt-4o" in text
        assert "claude-sonnet" in text
        assert "2 metrics" in text

    def test_models_listed(self):
        result = JudgeRoutingResult(
            routes=[],
            models_used=["gpt-4o", "claude-sonnet"],
            total_metrics=0,
        )
        text = format_routing_plan(result)
        assert "Models: gpt-4o, claude-sonnet" in text

    def test_reason_in_parentheses(self):
        result = JudgeRoutingResult(
            routes=[JudgeRoute(metric_id="x", assigned_model="y", reason="specific")],
            models_used=["y"],
            total_metrics=1,
        )
        text = format_routing_plan(result)
        assert "(specific)" in text

    def test_empty_reason_no_parens(self):
        result = JudgeRoutingResult(
            routes=[JudgeRoute(metric_id="x", assigned_model="y", reason="")],
            models_used=["y"],
            total_metrics=1,
        )
        text = format_routing_plan(result)
        assert "()" not in text
