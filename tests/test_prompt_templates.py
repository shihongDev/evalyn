"""Tests for calibration prompt templates."""

from __future__ import annotations

import pytest

from evalyn_sdk.calibration.prompt_templates import (
    BUILTIN_TEMPLATES,
    PromptTemplate,
    TemplateLibrary,
    get_builtin_templates,
    render_template,
)


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    def test_as_dict(self):
        t = PromptTemplate(
            name="test",
            template="Hello {{name}}",
            variables=["name"],
            description="A test template",
            category="testing",
        )
        d = t.as_dict()
        assert d == {
            "name": "test",
            "template": "Hello {{name}}",
            "variables": ["name"],
            "description": "A test template",
            "category": "testing",
        }

    def test_from_dict(self):
        d = {
            "name": "test",
            "template": "Hello {{name}}",
            "variables": ["name"],
            "description": "A test",
            "category": "testing",
        }
        t = PromptTemplate.from_dict(d)
        assert t.name == "test"
        assert t.template == "Hello {{name}}"
        assert t.variables == ["name"]
        assert t.description == "A test"
        assert t.category == "testing"

    def test_from_dict_defaults(self):
        d = {"name": "minimal", "template": "no vars"}
        t = PromptTemplate.from_dict(d)
        assert t.variables == []
        assert t.description == ""
        assert t.category == "general"

    def test_roundtrip(self):
        original = PromptTemplate(
            name="rt",
            template="{{a}} and {{b}}",
            variables=["a", "b"],
            description="roundtrip test",
            category="test",
        )
        restored = PromptTemplate.from_dict(original.as_dict())
        assert restored.name == original.name
        assert restored.template == original.template
        assert restored.variables == original.variables
        assert restored.description == original.description
        assert restored.category == original.category


# ---------------------------------------------------------------------------
# TemplateLibrary
# ---------------------------------------------------------------------------


class TestTemplateLibrary:
    def test_add_and_get(self):
        lib = TemplateLibrary()
        t = PromptTemplate(name="foo", template="bar")
        lib.add(t)
        assert lib.get("foo") is t

    def test_get_missing_returns_none(self):
        lib = TemplateLibrary()
        assert lib.get("nonexistent") is None

    def test_list_by_category(self):
        lib = TemplateLibrary()
        lib.add(PromptTemplate(name="a", template="", category="eval"))
        lib.add(PromptTemplate(name="b", template="", category="eval"))
        lib.add(PromptTemplate(name="c", template="", category="other"))
        evals = lib.list_by_category("eval")
        assert len(evals) == 2
        names = {t.name for t in evals}
        assert names == {"a", "b"}

    def test_list_by_category_empty(self):
        lib = TemplateLibrary()
        assert lib.list_by_category("missing") == []

    def test_list_all(self):
        lib = TemplateLibrary()
        lib.add(PromptTemplate(name="a", template=""))
        lib.add(PromptTemplate(name="b", template=""))
        assert len(lib.list_all()) == 2

    def test_list_all_empty(self):
        lib = TemplateLibrary()
        assert lib.list_all() == []

    def test_remove_existing(self):
        lib = TemplateLibrary()
        lib.add(PromptTemplate(name="a", template=""))
        assert lib.remove("a") is True
        assert lib.get("a") is None

    def test_remove_nonexistent(self):
        lib = TemplateLibrary()
        assert lib.remove("nope") is False

    def test_add_overwrites(self):
        lib = TemplateLibrary()
        lib.add(PromptTemplate(name="a", template="v1"))
        lib.add(PromptTemplate(name="a", template="v2"))
        assert lib.get("a").template == "v2"

    def test_as_dict_roundtrip(self):
        lib = TemplateLibrary()
        lib.add(PromptTemplate(name="x", template="{{v}}", variables=["v"], category="cat"))
        lib.add(PromptTemplate(name="y", template="plain", category="other"))
        d = lib.as_dict()
        restored = TemplateLibrary.from_dict(d)
        assert restored.get("x").template == "{{v}}"
        assert restored.get("y").category == "other"
        assert len(restored.list_all()) == 2

    def test_from_dict_empty(self):
        lib = TemplateLibrary.from_dict({})
        assert lib.list_all() == []


# ---------------------------------------------------------------------------
# render_template
# ---------------------------------------------------------------------------


class TestRenderTemplate:
    def test_basic_render(self):
        t = PromptTemplate(name="t", template="Hello {{name}}, welcome to {{place}}")
        result = render_template(t, {"name": "Alice", "place": "Wonderland"})
        assert result == "Hello Alice, welcome to Wonderland"

    def test_missing_variable_raises(self):
        t = PromptTemplate(name="t", template="Hello {{name}}")
        with pytest.raises(ValueError, match="Missing variables.*name"):
            render_template(t, {})

    def test_extra_variables_ignored(self):
        t = PromptTemplate(name="t", template="Hello {{name}}")
        result = render_template(t, {"name": "Bob", "extra": "ignored"})
        assert result == "Hello Bob"

    def test_no_variables(self):
        t = PromptTemplate(name="t", template="No placeholders here")
        result = render_template(t, {})
        assert result == "No placeholders here"

    def test_repeated_variable(self):
        t = PromptTemplate(name="t", template="{{x}} and {{x}}")
        result = render_template(t, {"x": "val"})
        assert result == "val and val"

    def test_partial_missing_raises(self):
        t = PromptTemplate(name="t", template="{{a}} {{b}}")
        with pytest.raises(ValueError, match="Missing variables.*b"):
            render_template(t, {"a": "yes"})


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------


class TestBuiltinTemplates:
    def test_has_three_builtins(self):
        assert len(BUILTIN_TEMPLATES) == 3

    def test_get_builtin_templates_returns_library(self):
        lib = get_builtin_templates()
        assert isinstance(lib, TemplateLibrary)
        assert len(lib.list_all()) == 3

    def test_binary_judge_renders(self):
        lib = get_builtin_templates()
        t = lib.get("binary_judge")
        result = render_template(t, {"metric_name": "accuracy", "output": "42"})
        assert "accuracy" in result
        assert "42" in result
        assert "PASS or FAIL" in result

    def test_likert_judge_renders(self):
        lib = get_builtin_templates()
        t = lib.get("likert_judge")
        result = render_template(t, {"metric_name": "fluency", "output": "text"})
        assert "fluency" in result
        assert "1-5" in result

    def test_comparison_judge_renders(self):
        lib = get_builtin_templates()
        t = lib.get("comparison_judge")
        result = render_template(t, {
            "metric_name": "clarity",
            "output_a": "first",
            "output_b": "second",
        })
        assert "clarity" in result
        assert "first" in result
        assert "second" in result

    def test_builtin_categories(self):
        lib = get_builtin_templates()
        evals = lib.list_by_category("evaluation")
        comps = lib.list_by_category("comparison")
        assert len(evals) == 2
        assert len(comps) == 1
