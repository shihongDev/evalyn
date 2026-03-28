"""Tests for quickstart templates."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.quickstart_templates import (
    TEMPLATES,
    QuickstartResult,
    QuickstartTemplate,
    get_quickstart,
    list_quickstarts,
    plan_quickstart,
    render_quickstart_guide,
    suggest_quickstart,
)

import pytest


# ---------------------------------------------------------------------------
# get_quickstart
# ---------------------------------------------------------------------------


class TestGetQuickstart:
    def test_langchain(self):
        t = get_quickstart("langchain")
        assert t.name == "langchain"
        assert t.framework == "langchain"

    def test_langgraph(self):
        t = get_quickstart("langgraph")
        assert t.name == "langgraph"
        assert t.framework == "langgraph"

    def test_crewai(self):
        t = get_quickstart("crewai")
        assert t.name == "crewai"
        assert t.framework == "crewai"

    def test_vanilla(self):
        t = get_quickstart("vanilla")
        assert t.name == "vanilla"
        assert t.framework == "python"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown quickstart"):
            get_quickstart("nonexistent")

    def test_unknown_lists_available(self):
        with pytest.raises(ValueError, match="crewai"):
            get_quickstart("nope")


# ---------------------------------------------------------------------------
# list_quickstarts
# ---------------------------------------------------------------------------


class TestListQuickstarts:
    def test_has_four_templates(self):
        templates = list_quickstarts()
        assert len(templates) == 4

    def test_all_are_quickstart_templates(self):
        for t in list_quickstarts():
            assert isinstance(t, QuickstartTemplate)

    def test_names_present(self):
        names = {t.name for t in list_quickstarts()}
        assert names == {"langchain", "langgraph", "crewai", "vanilla"}


# ---------------------------------------------------------------------------
# plan_quickstart
# ---------------------------------------------------------------------------


class TestPlanQuickstart:
    def test_lists_files(self):
        result = plan_quickstart("langchain")
        assert "evalyn.yaml" in result.files_created
        assert "agent.py" in result.files_created

    def test_lists_commands(self):
        result = plan_quickstart("langchain")
        assert len(result.commands_to_run) >= 1
        assert any("install" in cmd for cmd in result.commands_to_run)

    def test_template_name_set(self):
        result = plan_quickstart("vanilla")
        assert result.template_name == "vanilla"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            plan_quickstart("unknown")

    def test_crewai_files(self):
        result = plan_quickstart("crewai")
        assert "crew.py" in result.files_created

    def test_langgraph_files(self):
        result = plan_quickstart("langgraph")
        assert "graph.py" in result.files_created


# ---------------------------------------------------------------------------
# render_quickstart_guide
# ---------------------------------------------------------------------------


class TestRenderQuickstartGuide:
    def test_contains_template_name(self):
        t = get_quickstart("langchain")
        guide = render_quickstart_guide(t)
        assert "langchain" in guide

    def test_contains_gemini_api_key(self):
        t = get_quickstart("vanilla")
        guide = render_quickstart_guide(t)
        assert "GEMINI_API_KEY" in guide

    def test_contains_install_step(self):
        t = get_quickstart("crewai")
        guide = render_quickstart_guide(t)
        assert "Install dependencies" in guide

    def test_contains_run_step(self):
        t = get_quickstart("langgraph")
        guide = render_quickstart_guide(t)
        assert "evalyn run" in guide

    def test_contains_description(self):
        t = get_quickstart("langchain")
        guide = render_quickstart_guide(t)
        assert "LangChain agent evaluation" in guide

    def test_lists_files(self):
        t = get_quickstart("vanilla")
        guide = render_quickstart_guide(t)
        assert "main.py" in guide
        assert "evalyn.yaml" in guide


# ---------------------------------------------------------------------------
# suggest_quickstart
# ---------------------------------------------------------------------------


class TestSuggestQuickstart:
    def test_exact_match(self):
        t = suggest_quickstart("langchain")
        assert t is not None
        assert t.name == "langchain"

    def test_case_insensitive(self):
        t = suggest_quickstart("LangChain")
        assert t is not None
        assert t.name == "langchain"

    def test_framework_match(self):
        t = suggest_quickstart("python")
        assert t is not None
        assert t.name == "vanilla"

    def test_no_match(self):
        t = suggest_quickstart("tensorflow")
        assert t is None

    def test_empty_string(self):
        t = suggest_quickstart("")
        assert t is None

    def test_whitespace_only(self):
        t = suggest_quickstart("   ")
        assert t is None

    def test_substring_match(self):
        t = suggest_quickstart("crew")
        assert t is not None
        assert t.name == "crewai"


# ---------------------------------------------------------------------------
# QuickstartTemplate as_dict / from_dict
# ---------------------------------------------------------------------------


class TestQuickstartTemplateRoundtrip:
    def test_roundtrip(self):
        t = QuickstartTemplate(
            name="test",
            framework="test-fw",
            description="A test template",
            files={"a.py": "# code\n"},
            dependencies=["dep1"],
            setup_commands=["cmd1"],
        )
        d = t.as_dict()
        restored = QuickstartTemplate.from_dict(d)
        assert restored.name == t.name
        assert restored.framework == t.framework
        assert restored.description == t.description
        assert restored.files == t.files
        assert restored.dependencies == t.dependencies
        assert restored.setup_commands == t.setup_commands

    def test_from_dict_defaults(self):
        t = QuickstartTemplate.from_dict({})
        assert t.name == ""
        assert t.framework == ""
        assert t.files == {}
        assert t.dependencies == []

    def test_builtin_template_roundtrip(self):
        for name, template in TEMPLATES.items():
            d = template.as_dict()
            restored = QuickstartTemplate.from_dict(d)
            assert restored.name == template.name
            assert restored.framework == template.framework


# ---------------------------------------------------------------------------
# QuickstartResult
# ---------------------------------------------------------------------------


class TestQuickstartResult:
    def test_as_dict(self):
        r = QuickstartResult(
            template_name="test",
            files_created=["a.py"],
            commands_to_run=["cmd"],
        )
        d = r.as_dict()
        assert d["template_name"] == "test"
        assert d["files_created"] == ["a.py"]
        assert d["commands_to_run"] == ["cmd"]

    def test_format_text_basic(self):
        r = QuickstartResult(
            template_name="langchain",
            files_created=["evalyn.yaml", "agent.py"],
            commands_to_run=["uv pip install langchain"],
        )
        text = r.format_text()
        assert "langchain" in text
        assert "evalyn.yaml" in text
        assert "agent.py" in text
        assert "uv pip install" in text

    def test_format_text_empty(self):
        r = QuickstartResult(template_name="empty")
        text = r.format_text()
        assert "empty" in text

    def test_format_text_files_section(self):
        r = QuickstartResult(
            template_name="t",
            files_created=["f1.py"],
        )
        text = r.format_text()
        assert "Files to create" in text

    def test_format_text_commands_section(self):
        r = QuickstartResult(
            template_name="t",
            commands_to_run=["echo hello"],
        )
        text = r.format_text()
        assert "Commands to run" in text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_templates_dict_not_empty(self):
        assert len(TEMPLATES) > 0

    def test_all_templates_have_files(self):
        for name, t in TEMPLATES.items():
            assert len(t.files) > 0, f"Template {name} has no files"

    def test_all_templates_have_dependencies(self):
        for name, t in TEMPLATES.items():
            assert len(t.dependencies) > 0, f"Template {name} has no deps"

    def test_all_templates_have_setup_commands(self):
        for name, t in TEMPLATES.items():
            assert len(t.setup_commands) > 0, f"Template {name} has no cmds"

    def test_all_templates_have_evalyn_yaml(self):
        for name, t in TEMPLATES.items():
            assert "evalyn.yaml" in t.files, f"Template {name} missing evalyn.yaml"
