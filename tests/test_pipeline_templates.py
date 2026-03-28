from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.pipeline_templates import (
    BUILTIN_TEMPLATES,
    TemplateConfig,
    TemplateLibrary,
    get_default_library,
    get_template,
    list_templates,
    suggest_template,
)


# ---------------------------------------------------------------------------
# TemplateConfig
# ---------------------------------------------------------------------------


class TestTemplateConfig:
    def test_as_dict(self):
        t = TemplateConfig("t1", "safety", ["m1", "m2"], "gemini", "standard", "desc")
        d = t.as_dict()
        assert d["name"] == "t1"
        assert d["goal"] == "safety"
        assert d["metrics"] == ["m1", "m2"]
        assert d["provider"] == "gemini"
        assert d["profile"] == "standard"
        assert d["description"] == "desc"

    def test_from_dict_roundtrip(self):
        t = TemplateConfig("t1", "quality", ["helpfulness"], "openai", "thorough", "d")
        restored = TemplateConfig.from_dict(t.as_dict())
        assert restored == t

    def test_from_dict_defaults(self):
        t = TemplateConfig.from_dict({"name": "x"})
        assert t.goal == ""
        assert t.metrics == []
        assert t.provider == "gemini"
        assert t.profile == "standard"
        assert t.description == ""

    def test_as_dict_copies_metrics(self):
        original = ["a", "b"]
        t = TemplateConfig("t", metrics=original)
        d = t.as_dict()
        d["metrics"].append("c")
        assert t.metrics == ["a", "b"]

    def test_defaults(self):
        t = TemplateConfig("minimal")
        assert t.goal == ""
        assert t.metrics == []
        assert t.provider == "gemini"
        assert t.profile == "standard"
        assert t.description == ""


# ---------------------------------------------------------------------------
# TemplateLibrary
# ---------------------------------------------------------------------------


class TestTemplateLibrary:
    def test_add_and_get(self):
        lib = TemplateLibrary()
        cfg = TemplateConfig("my-template", "safety")
        lib.add(cfg)
        assert lib.get("my-template") is cfg

    def test_get_missing_returns_none(self):
        lib = TemplateLibrary()
        assert lib.get("nope") is None

    def test_list_all(self):
        lib = TemplateLibrary()
        lib.add(TemplateConfig("a"))
        lib.add(TemplateConfig("b"))
        assert len(lib.list_all()) == 2

    def test_list_by_goal(self):
        lib = TemplateLibrary()
        lib.add(TemplateConfig("a", "safety"))
        lib.add(TemplateConfig("b", "quality"))
        lib.add(TemplateConfig("c", "safety"))
        result = lib.list_by_goal("safety")
        assert len(result) == 2
        assert all(t.goal == "safety" for t in result)

    def test_list_by_goal_empty(self):
        lib = TemplateLibrary()
        lib.add(TemplateConfig("a", "safety"))
        assert lib.list_by_goal("cost") == []

    def test_remove_existing(self):
        lib = TemplateLibrary()
        lib.add(TemplateConfig("a"))
        assert lib.remove("a") is True
        assert lib.get("a") is None

    def test_remove_missing(self):
        lib = TemplateLibrary()
        assert lib.remove("nope") is False

    def test_add_overwrites(self):
        lib = TemplateLibrary()
        lib.add(TemplateConfig("x", "old"))
        lib.add(TemplateConfig("x", "new"))
        assert lib.get("x").goal == "new"

    def test_as_dict(self):
        lib = TemplateLibrary()
        lib.add(TemplateConfig("t1", "safety"))
        d = lib.as_dict()
        assert "t1" in d["templates"]

    def test_from_dict_roundtrip(self):
        lib = TemplateLibrary()
        lib.add(TemplateConfig("t1", "safety", ["m1"]))
        lib.add(TemplateConfig("t2", "quality"))
        restored = TemplateLibrary.from_dict(lib.as_dict())
        assert len(restored.list_all()) == 2
        assert restored.get("t1").metrics == ["m1"]

    def test_empty_library(self):
        lib = TemplateLibrary()
        assert lib.list_all() == []
        assert lib.as_dict() == {"templates": {}}

    def test_from_dict_empty(self):
        lib = TemplateLibrary.from_dict({})
        assert lib.list_all() == []


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


class TestGetTemplate:
    def test_builtin_safety_gate(self):
        t = get_template("safety-gate")
        assert t.name == "safety-gate"
        assert t.goal == "safety"

    def test_builtin_quality_check(self):
        t = get_template("quality-check")
        assert t.goal == "quality"

    def test_builtin_regression_test(self):
        t = get_template("regression-test")
        assert t.profile == "thorough"

    def test_builtin_cost_audit(self):
        t = get_template("cost-audit")
        assert t.profile == "cost-optimized"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Template not found"):
            get_template("nonexistent")


class TestListTemplates:
    def test_has_four_builtins(self):
        result = list_templates()
        assert len(result) == 4

    def test_names(self):
        names = {t.name for t in list_templates()}
        assert names == {"safety-gate", "quality-check", "regression-test", "cost-audit"}


class TestGetDefaultLibrary:
    def test_contains_builtins(self):
        lib = get_default_library()
        assert lib.get("safety-gate") is not None
        assert lib.get("quality-check") is not None
        assert len(lib.list_all()) == 4


class TestSuggestTemplate:
    def test_known_goal_safety(self):
        t = suggest_template("safety")
        assert t is not None
        assert t.goal == "safety"

    def test_known_goal_quality(self):
        t = suggest_template("quality")
        assert t is not None
        assert t.goal == "quality"

    def test_known_goal_regression(self):
        t = suggest_template("regression")
        assert t is not None
        assert t.name == "regression-test"

    def test_known_goal_cost(self):
        t = suggest_template("cost")
        assert t is not None
        assert t.name == "cost-audit"

    def test_unknown_goal_returns_none(self):
        assert suggest_template("unknown-goal") is None
